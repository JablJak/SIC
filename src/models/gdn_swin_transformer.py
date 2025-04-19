from functools import partial
from typing import Optional, Callable, Any

from compressai.layers import GDN, GDN1
from torch import nn, Tensor
from torchvision.models._api import register_model, WeightsEnum
from torchvision.models._utils import handle_legacy_interface, _ovewrite_named_param
from torchvision.models.swin_transformer import PatchMergingV2, SwinTransformerBlockV2, Swin_S_Weights, \
    _patch_merging_pad
from torchvision.ops import Permute, MLP


class VariableDepthPatchMerging(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduction = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = norm_layer(out_dim)  # difference

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C
        x = self.norm(x)
        return x



class GDNSwinTransformer(nn.Module):
    def __init__(
        self,
        patch_size: list[int],
        embed_dim: int,
        stage_dims: list[int],
        depths: list[int],
        num_heads: list[int],
        window_size: list[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 1000,
        block = SwinTransformerBlockV2,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = VariableDepthPatchMerging,
    ):
        super().__init__()
        self.num_classes = num_classes

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        layers: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(
                    3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
                ),
                Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )
        ]

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        for i_stage in range(len(depths)):
            stage: list[nn.Module] = []
            dim = stage_dims[i_stage]
            for i_layer in range(depths[i_stage]):
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                swin_block = block(
                    dim,
                    num_heads[i_stage],
                    window_size=window_size,
                    shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    stochastic_depth_prob=sd_prob,
                    norm_layer=norm_layer,
                )
                stage.append(swin_block)
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(4 * stage_dims[i_stage], stage_dims[i_stage + 1], norm_layer))
                layers.append(permute_and_gdn(stage_dims[i_stage + 1], inverse=False))
        self.features = nn.Sequential(*layers)

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)
        self.permute = Permute([0, 3, 1, 2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.head = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        return x



@register_model()
@handle_legacy_interface(weights=("pretrained", Swin_S_Weights.IMAGENET1K_V1))
def gdn_swin_v2_s(
        *,
        weights: Optional[Swin_S_Weights] = None,
        embed_dim=96,
        stage_dims=[96, 192, 288, 384],
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.3,
        mlp_ratio=4.0,
        progress: bool = True, **kwargs: Any) -> GDNSwinTransformer:
    weights = Swin_S_Weights.verify(weights)

    return _gdn_swin_transformer(
        patch_size=[4, 4],
        embed_dim=embed_dim,
        stage_dims=stage_dims,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        mlp_ratio=mlp_ratio,
        weights=weights,
        progress=progress,
        block=SwinTransformerBlockV2,
        downsample_layer=VariableDepthPatchMerging,
        **kwargs,
    )

def _gdn_swin_transformer(
    patch_size: list[int],
    embed_dim: int,
    stage_dims: list[int],
    depths: list[int],
    num_heads: list[int],
    window_size: list[int],
    stochastic_depth_prob: float,
    mlp_ratio: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> GDNSwinTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = GDNSwinTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        stage_dims=stage_dims,
        mlp_ratio=mlp_ratio,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model

def permute_and_gdn(dim: int, inverse: bool) -> nn.Sequential:
    return nn.Sequential(
        Permute([0, 3, 1, 2]),
        GDN1(dim, inverse=inverse),
        Permute([0, 2, 3, 1])
    )