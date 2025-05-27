from functools import partial
from typing import Optional, Callable, Any

import torch
from compressai.layers import GDN, GDN1
from torch import nn, Tensor
from torchvision.models._api import register_model, WeightsEnum
from torchvision.models._utils import handle_legacy_interface, _ovewrite_named_param
from torchvision.models.swin_transformer import PatchMergingV2, SwinTransformerBlockV2, Swin_S_Weights, \
    _patch_merging_pad, ShiftedWindowAttentionV2, SwinTransformerBlock, Swin_V2_S_Weights
from torchvision.ops import Permute, MLP

from src.utils.initializers import initialize_weights


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


class SwinTransformerBlockMixed(SwinTransformerBlock):
    """
    Swin Transformer V2 Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttentionV2.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: list[int],
        shift_size: list[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_layer: Callable[..., nn.Module] = ShiftedWindowAttentionV2,
    ):
        super().__init__(
            dim,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
            attn_layer=attn_layer,
        )
        # self.norm3 = norm_layer(dim)
        # self.cnn_post_attn = CNNPostAttn(dim)

    def forward(self, x: Tensor):
        # x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        # cnn = self.cnn_inv_post_attn(x)
        # mlp = self.mlp(x)
        # x = x + self.stochastic_depth(self.norm2(0.5 * cnn + 0.5 * mlp))
        # return x
        x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        x = x + self.stochastic_depth(self.norm2(self.mlp(x)))
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
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        stochastic_depth_prob: float = 0.1,
        block = SwinTransformerBlockMixed,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = VariableDepthPatchMerging,
    ):
        super().__init__()

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                3, embed_dim, kernel_size=(patch_size[0], patch_size[1]), stride=(patch_size[0], patch_size[1])
            ),
            Permute([0, 2, 3, 1]),
            norm_layer(embed_dim),
        )

        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.gdn_layers = nn.ModuleList()
        self.residual_norms = nn.ModuleList()

        total_stage_blocks = sum(depths)
        stage_block_id = 0

        for i_stage in range(len(depths)):
            stage_module_list: list[nn.Module] = []
            dim = stage_dims[i_stage]
            for i_layer in range(depths[i_stage]):
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1) if total_stage_blocks > 1 else 0.0
                swin_block = block(
                    dim, num_heads[i_stage], window_size=window_size,
                    shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                    mlp_ratio=mlp_ratio, dropout=dropout, attention_dropout=attention_dropout,
                    stochastic_depth_prob=sd_prob, norm_layer=norm_layer,
                )
                stage_module_list.append(swin_block)
                stage_block_id += 1

            self.stages.append(nn.Sequential(*stage_module_list))
            self.residual_norms.append(norm_layer(dim))

            if i_stage < (len(depths) - 1):
                next_dim = stage_dims[i_stage+1]
                self.downsamplers.append(downsample_layer(4 * dim, next_dim, norm_layer))
                self.gdn_layers.append(permute_and_gdn(next_dim, inverse=False))

        initialize_weights(self)


    def forward(self, x):
        x = self.patch_embed(x)

        for i in range(len(self.stages)):
            residual = x
            x = self.stages[i](x)
            x = x + residual
            x = self.residual_norms[i](x)
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)
                x = self.gdn_layers[i](x)

        return x


@register_model()
@handle_legacy_interface(weights=("pretrained", Swin_V2_S_Weights.IMAGENET1K_V1))
def gdn_swin_v2_s(
        *,
        weights: Optional[Swin_V2_S_Weights] = None,
        embed_dim=96,
        stage_dims=[96, 192, 288, 384],
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.3,
        mlp_ratio=4.0,
        progress: bool = True, **kwargs: Any) -> GDNSwinTransformer:
    weights = Swin_V2_S_Weights.verify(weights)

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

class LinearScheduler:
    def __init__(self, total_steps, initial_value=0.0, final_value=1.0):
        self.total_steps = total_steps
        self.initial_value = initial_value
        self.final_value = final_value
        self.current_step = 0

    def step(self):
        value = self.get_value()
        self.current_step += 1
        return value

    def get_value(self):
        if self.current_step >= self.total_steps:
            return self.final_value
        progress = self.current_step / self.total_steps
        return self.initial_value + progress * (self.final_value - self.initial_value)

    def reset(self):
         self.current_step = 0

class GradualIntroductionLayer(nn.Module):
    def __init__(self, layer_to_introduce, alpha=1.0, identity_path=None):
        super().__init__()
        self.alpha = 0.0
        self.set_alpha(alpha)
        self.layer = layer_to_introduce
        self.identity = identity_path if identity_path is not None else nn.Identity()

    def forward(self, x):
        layer_output = self.layer(x)
        identity_output = self.identity(x)

        if layer_output.shape != identity_output.shape:
             raise RuntimeError(f"Shape mismatch between layer output ({layer_output.shape}) "
                                f"and identity path output ({identity_output.shape}). "
                                "Weighted sum requires matching shapes.")

        if self.training:
            alpha_tensor = torch.tensor(self.alpha, dtype=x.dtype, device=x.device)

            output = alpha_tensor * layer_output + (1.0 - alpha_tensor) * identity_output
        else:
            output = layer_output

        return output

    def set_alpha(self, alpha):
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0.")
        self.alpha = alpha
        
class CNNPostAttn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        # self.gdn1 = GDN1(in_channels=dim, inverse=False)
        # self.gdn2 = GDN1(in_channels=dim, inverse=False)
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.gelu2(x)
        return x.permute(0, 2, 3, 1)


class CNNInvPostAttn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.ConvTranspose2d(dim, dim, kernel_size=3, stride=1, padding=1)
        # self.gdn1 = GDN1(in_channels=dim, inverse=True)
        # self.gdn2 = GDN1(in_channels=dim, inverse=True)
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.gelu1 = nn.GELU()
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.gelu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.gelu2(x)
        return x.permute(0, 2, 3, 1)

def permute_and_gdn(dim: int, inverse: bool) -> nn.Sequential:
    return nn.Sequential(
        Permute([0, 3, 1, 2]),
        GradualIntroductionLayer(
            layer_to_introduce=GDN1(in_channels=dim, inverse=inverse),
        ),
        Permute([0, 2, 3, 1])
    )