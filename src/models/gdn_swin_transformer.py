from functools import partial
from typing import Optional, Callable, Any, cast

import torch
from compressai.layers import GDN1
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint_sequential
from torchvision.models._api import register_model, WeightsEnum
from torchvision.models._utils import handle_legacy_interface
from torchvision.models.swin_transformer import SwinTransformerBlockV2, \
    ShiftedWindowAttentionV2, SwinTransformerBlock, Swin_V2_S_Weights, Swin_V2_B_Weights, _patch_merging_pad
from torchvision.ops import Permute
from src.utils.initializers import initialize_weights
from src.utils.torch_utils import get_boundary_mask


class VariableDepthPatchMerging(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, norm: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm = norm(out_dim)

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x



class MaskedSwinTransformerBlock(SwinTransformerBlock):
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

    def forward(self, x: Tensor):
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
        checkpointing: bool = False,
        bottleneck_dim: int = 384,
        block = SwinTransformerBlockV2,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = VariableDepthPatchMerging,
    ):
        super().__init__()
        self.checkpointing = checkpointing
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                3, embed_dim, kernel_size=(patch_size[0]+1, patch_size[1]+1),
                stride=(patch_size[0], patch_size[1]), padding=((patch_size[0]+1)//2, (patch_size[1]+1)//2),
            ),
            Permute([0, 2, 3, 1]),
            norm_layer(embed_dim),
        )

        self.stages = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        self.gdns = nn.ModuleList()
        self.a_proj = nn.Conv2d(stage_dims[-1], bottleneck_dim, kernel_size=1)
        self.mask_fusions = nn.ModuleList()

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
                    stochastic_depth_prob=sd_prob, norm_layer=norm_layer
                )
                stage_module_list.append(swin_block)
                stage_block_id += 1

            self.stages.append(nn.Sequential(*stage_module_list))
            self.gdns.append(nn.Sequential(
                Permute([0, 3, 1, 2]),
                GDN1(stage_dims[i_stage + 1] if i_stage + 1 < len(depths) else stage_dims[i_stage]),
                Permute([0, 2, 3, 1])
            ))
            self.mask_fusions.append(nn.Sequential(
                Permute([0, 3, 1, 2]),
                nn.Conv2d(dim + 1, dim, kernel_size=1),
                Permute([0, 2, 3, 1])
            ))
            if i_stage < (len(depths) - 1):
                next_dim = stage_dims[i_stage+1]
                self.downsamplers.append(downsample_layer(dim, next_dim))
        initialize_weights(self)


    def forward(self, x):
        x = 2 * x - 1
        x = self.patch_embed(x)
        for i in range(len(self.stages)):
            x = x.to(next(self.stages[i].parameters()).device)

            B, H, W, C = x.shape
            mask = get_boundary_mask(H, W, x.device)
            mask = mask.expand(B, -1, -1, -1)
            x = torch.concat([x, mask], dim=3)
            x = self.mask_fusions[i](x)

            if self.checkpointing:
                x = checkpoint_sequential(self.stages[i], int(len(cast(nn.Sequential, self.stages[i]))), x, use_reentrant=False)
            else:
                x = self.stages[i](x)
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)
            x = self.gdns[i](x)

        x = x.permute(0, 3, 1, 2)
        x = self.a_proj(x)
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
        checkpointing=False,
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
        checkpointing=checkpointing,
        **kwargs,
    )
@register_model()
@handle_legacy_interface(weights=("pretrained", Swin_V2_B_Weights.IMAGENET1K_V1))
def gdn_swin_v2_b(
        *,
        weights: Optional[Swin_V2_B_Weights] = None,
        embed_dim=128,
        stage_dims=[128, 256, 512, 1024],
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8],
        stochastic_depth_prob=0.3,
        mlp_ratio=4.0,
        checkpointing=False,
        bottleneck_dim=384,
        dropout=0,
        attention_dropout=0,
        progress: bool = True, **kwargs: Any) -> GDNSwinTransformer:
    weights = Swin_V2_B_Weights.verify(weights)

    return _gdn_swin_transformer(
        patch_size=[4, 4],
        embed_dim=embed_dim,
        stage_dims=stage_dims,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        attention_dropout=attention_dropout,
        weights=weights,
        progress=progress,
        block=SwinTransformerBlockV2,
        downsample_layer=VariableDepthPatchMerging,
        checkpointing=checkpointing,
        bottleneck_dim=bottleneck_dim,
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
    dropout: float,
    attention_dropout: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    checkpointing: bool,
    bottleneck_dim: int,
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
        dropout=dropout,
        attention_dropout=attention_dropout,
        checkpointing=checkpointing,
        bottleneck_dim=bottleneck_dim,
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