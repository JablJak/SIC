import torch
from compressai.layers import GDN1
from torch.utils.checkpoint import checkpoint
from torch import nn, Tensor
from torchvision.models.swin_transformer import ShiftedWindowAttentionV2
from torchvision.ops import StochasticDepth, MLP, Permute

from src.utils.initializers import initialize_weights
from src.utils.torch_utils import get_boundary_mask


class PatchReconstruction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.upscale = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(dim, 3, kernel_size=1)
        )
        self.leaky_clamp = LeakyClamp(0.0, 1.0, 0.01)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.upscale(x)

        if self.training:
            x = self.leaky_clamp(x)
        else:
            x = torch.clamp(x, 0, 1)
        return x


class SwinTransformerDecoder(nn.Module):
    def __init__(
            self,
            stage_dims,
            num_heads,
            windows_sizes,
            mlp_ratios,
            depths,
            sd_factor,
            checkpointing,
            dropout,
            attention_dropout,
            bottleneck_dim,
        ):
        super().__init__()
        self.num_stages = n = len(num_heads)
        blocks_total = sum(depths)
        blocks_remaining = [blocks_total - sum(depths[:i]) for i in range(len(depths))]
        stages = [
                SwinTransformerDecoderStage(
                    in_dim=stage_dims[i],
                    out_dim=stage_dims[i + 1],
                    num_heads=num_heads[i],
                    window_size=windows_sizes[i],
                    mlp_ratio=mlp_ratios[i],
                    depth=depths[i],
                    sd_factor=sd_factor,
                    blocks_remaining=blocks_remaining[i],
                    blocks_total=blocks_total,
                    checkpointing=checkpointing,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
               for i in range(n)
        ]
        self.stages = nn.ModuleList(stages)
        self.reconstruction = PatchReconstruction(stage_dims[-1])
        self.s_proj = nn.Conv2d(bottleneck_dim, stage_dims[0], kernel_size=1)
        self.mask_fusions = nn.ModuleList(
            [nn.Sequential(
                Permute([0, 3, 1, 2]),
                nn.Conv2d(stage_dims[i] + 1, stage_dims[i], kernel_size=1),
                Permute([0, 2, 3, 1])
            ) for i in range(n)]
        )
        initialize_weights(self)

    def forward(self, x):
        x = self.s_proj(x)
        x = x.permute(0, 2, 3, 1)
        for i in range(len(self.stages)):
            B, H, W, C = x.shape
            mask = get_boundary_mask(H, W, x.device)
            mask = mask.expand(B, -1, -1, -1)
            x = torch.concat([x, mask], dim=3)
            x = self.mask_fusions[i](x)
            x = self.stages[i](x)

        x = self.reconstruction(x)
        return x


class SwinTransformerDecoderStage(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        window_size,
        mlp_ratio,
        depth,
        sd_factor,
        dropout,
        attention_dropout,
        blocks_remaining,
        blocks_total,
        checkpointing,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth = depth
        self.blocks = nn.ModuleList(
            [SwinTransformerDecoderBlock(
                dim=out_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0 if i_block % 2 == 0 else w // 2 for w in window_size],
                mlp_ratio=mlp_ratio,
                sd_factor=sd_factor * ((blocks_remaining - i_block) / blocks_total),
                dropout=dropout,
                attention_dropout=attention_dropout,
            ) for i_block in range(depth)]
        )

        self.checkpointing = checkpointing
        self.igdn = nn.Sequential(
            Permute([0, 3, 1, 2]),
            GDN1(in_channels=in_dim, inverse=True),
            Permute([0, 2, 3, 1])
        )
        self.upscale = nn.Sequential(
            Permute([0, 3, 1, 2]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            Permute([0, 2, 3, 1]),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        x = self.igdn(x)
        if self.in_dim != self.out_dim:
            x = self.upscale(x)
        for i in range(self.depth):
            if self.checkpointing:
                x = checkpoint(self.blocks[i], x, use_reentrant=False)
            else:
                x = self.blocks[i](x)
        return x


class SwinTransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        shift_size,
        sd_factor,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = ShiftedWindowAttentionV2(dim=dim, window_size=window_size, num_heads=num_heads,
                                             shift_size=shift_size, attention_dropout=attention_dropout)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim],
                       activation_layer=nn.GELU,
                       inplace=None, dropout=dropout)
        self.stochastic_depth = StochasticDepth(sd_factor, "row")

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        x = x + self.stochastic_depth(self.norm2(self.mlp(x)))
        return x

class LeakyClamp(nn.Module):
    def __init__(self, min_value, max_value, eps=1e-3):
        super().__init__()
        self.min_val = min_value
        self.max_val = max_value
        self.eps = eps

    def forward(self, x):
        x = torch.where(x < self.min_val, self.eps * (x - self.min_val) + self.min_val, x)
        x = torch.where(x > self.max_val, self.eps * (x - self.max_val) + self.max_val, x)
        return x