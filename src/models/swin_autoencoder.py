from functools import partial

from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
from timm.layers import DropPath
from torch.nn import init
from torch.utils.checkpoint import checkpoint
from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b, Swin_V2_T_Weights
from torch import nn, Tensor
from torchvision.models.swin_transformer import ShiftedWindowAttentionV2
from torchvision.ops import StochasticDepth, MLP

from src.models.gdn_swin_transformer import permute_and_gdn, CNNPostAttn, CNNInvPostAttn
from src.utils.activation import LearnableTempSigmoid, LearnableTempScaledTanh
from src.utils.initializers import initialize_weights


class PatchReconstruction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_trans_48 = nn.ConvTranspose2d(dim, 3, kernel_size=2, stride=2, padding=0)
        self.activation = LearnableTempScaledTanh()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv_trans_48(x)
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
            sd_factor,
            checkpointing,
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
                    checkpointing=checkpointing
                )
               for i in range(n)
        ]
        self.stages = nn.ModuleList(stages)
        self.reconstruction = PatchReconstruction(stage_dims[-1])

        initialize_weights(self)

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
        blocks_total,
        checkpointing,
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
        self.norms = nn.ModuleList([nn.LayerNorm(in_dim) for _ in range(depth)])
        self.norm_1 = nn.LayerNorm(in_dim)
        self.norm_2 = nn.LayerNorm(out_dim)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.checkpointing = checkpointing


    def forward(self, x):
        residual = x
        for i in range(self.depth):
            if self.checkpointing:
                x = checkpoint(self.blocks[i], x, use_reentrant=False)
            else:
                x = self.blocks[i](x)
            x = self.norms[i](x)
        x = x + residual
        x = self.norm_1(x)
        x = self.linear(x)
        x = x.permute(0, 3, 1, 2)
        x = self.pixel_shuffle(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm_2(x)
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
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # self.norm3 = nn.LayerNorm(dim)
        self.attn = ShiftedWindowAttentionV2(dim=dim, window_size=window_size, num_heads=num_heads, shift_size=shift_size, attention_dropout=0.1)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim],
                       activation_layer=nn.GELU,
                       inplace=None, dropout=dropout)
        self.stochastic_depth = StochasticDepth(sd_factor, "row")
        # self.cnn_inv_post_attn = CNNInvPostAttn(dim)

    def forward(self, x: Tensor):
        # x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        # cnn = self.cnn_inv_post_attn(x)
        # mlp = self.mlp(x)
        # x = x + self.stochastic_depth(self.norm2(0.5 * cnn + 0.5 * mlp))
        # return x
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x