from functools import partial

from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
from timm.layers import DropPath
from torch.nn import init
from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b, Swin_V2_T_Weights
from torch import nn, Tensor
from torchvision.models.swin_transformer import ShiftedWindowAttentionV2
from torchvision.ops import StochasticDepth, MLP

from src.models.gdn_swin_transformer import permute_and_gdn
from src.utils.activation import LearnableTempSigmoid


class PatchReconstruction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_trans = nn.ConvTranspose2d(dim, 3, kernel_size=2, stride=2, padding=0)
        self.activation = LearnableTempSigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv_trans(x)
        x = self.activation(x)
        return x


class SwinTransformerDecoder(nn.Module):
    def __init__(
            self,
            stage_dims,
            num_heads,
            windows_sizes,
            mlp_ratios,
            depths,
            sd_factor
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
                )
               for i in range(n)
        ]
        self.stages = nn.ModuleList(stages)
        self.reconstruction = PatchReconstruction(stage_dims[-1])

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
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
        blocks_remaining,
        blocks_total
    ):
        super().__init__()
        self.depth = depth

        self.linear = nn.Linear(in_dim, out_dim * 4)
        self.blocks = nn.ModuleList(
            [SwinTransformerDecoderBlock(
                dim=in_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0 if i_block % 2 == 0 else w // 2 for w in window_size],
                mlp_ratio=mlp_ratio,
                sd_factor=sd_factor * ((blocks_remaining - i_block) / blocks_total),
            ) for i_block in range(depth)]
        )
        self.norm = nn.LayerNorm(in_dim)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.activation = nn.GELU()
        self.gdn = permute_and_gdn(in_dim, inverse=True)


    def forward(self, x):
        x = self.gdn(x)
        for i in range(self.depth):
            x = self.blocks[i](x)
        x = self.norm(x)
        x = self.linear(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pixel_shuffle(x)
        x = x.permute(0, 2, 3, 1)
        return x


class SwinTransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        shift_size,
        sd_factor,
        dropout: float = 0.0,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = ShiftedWindowAttentionV2(dim=dim, window_size=window_size, num_heads=num_heads, shift_size=shift_size)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim],
                       activation_layer=nn.GELU,
                       inplace=None, dropout=dropout)
        self.stochastic_depth = StochasticDepth(sd_factor, "row")

        # TODO: Initialize rest of the layers

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        x = x + self.stochastic_depth(self.norm2(self.mlp(x)))
        return x