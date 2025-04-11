from functools import partial
from typing import Optional, Callable, Any

from compressai.layers import GDN
from torch import nn
from torchvision.models._api import register_model, WeightsEnum
from torchvision.models._utils import handle_legacy_interface, _ovewrite_named_param
from torchvision.models.swin_transformer import PatchMergingV2, SwinTransformerBlockV2, Swin_S_Weights
from torchvision.ops import Permute, MLP


class GDNSwinTransformer(nn.Module):
    def __init__(
        self,
        patch_size: list[int],
        embed_dim: int,
        depths: list[int],
        num_heads: list[int],
        window_size: list[int],
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 1000,
        block = SwinTransformerBlockV2,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMergingV2,
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
            dim = embed_dim * 2**i_stage
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
                swin_block.mlp = MLP(
                    dim,
                    [int(dim * mlp_ratio), dim],
                    activation_layer=partial(permute_and_gdn, dim=(dim * int(mlp_ratio)), inverse=False),
                    inplace=None,
                    dropout=dropout
                )
                stage.append(swin_block)
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
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
def gdn_swin_v2_s(*, weights: Optional[Swin_S_Weights] = None, progress: bool = True, **kwargs: Any) -> GDNSwinTransformer:
    weights = Swin_S_Weights.verify(weights)

    return _gdn_swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.3,
        weights=weights,
        progress=progress,
        block=SwinTransformerBlockV2,
        downsample_layer=PatchMergingV2,
        **kwargs,
    )

def _gdn_swin_transformer(
    patch_size: list[int],
    embed_dim: int,
    depths: list[int],
    num_heads: list[int],
    window_size: list[int],
    stochastic_depth_prob: float,
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
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model

def permute_and_gdn(dim: int, inverse: bool) -> nn.Sequential:
    return nn.Sequential(
        Permute([0, 3, 1, 2]),
        GDN(dim, inverse=inverse),
        Permute([0, 2, 3, 1])
    )