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

class PatchReconstruction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_trans = nn.ConvTranspose2d(48, 3, kernel_size=4, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(3)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv_trans(x)
        x = self.norm(x)
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
        self.reconstruction = PatchReconstruction()

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

    def forward(self, x):
        for i in range(self.depth):
            x = self.blocks[i](x)
        x = self.norm(x)
        x = self.linear(x)
        x = self.activation(x)
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
        mlp_ratio=2.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = ShiftedWindowAttentionV2(dim=dim, window_size=window_size, num_heads=num_heads, shift_size=shift_size)
        self.mlp = MLP(dim, [int(dim * mlp_ratio), dim],
                       activation_layer=partial(permute_and_gdn, dim=(dim * int(mlp_ratio)), inverse=True),
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

class SwinTransformerAutoencoder(nn.Module):
    ENCODER_MAP = {
        "swin_v2_t": swin_v2_t,
        "swin_v2_s": swin_v2_s,
        "swin_v2_b": swin_v2_b,
    }

    def __init__(
            self,
            encoder_type = "swin_v2_t",
            encoder_weights = Swin_V2_T_Weights.DEFAULT,
            decoder_dims = (240, 192, 144, 96, 48),
            decoder_num_heads = (24, 12, 6, 3),
            decoder_window_size = ((7, 7), (7, 7), (7, 7), (7, 7)),
            decoder_mlp_ratio = (4, 4, 4, 4),
            decoder_depths = (2, 6, 2, 2),
            decoder_sd_factor = 0.1
    ):
        super().__init__()
        self.encoder_weights = encoder_weights
        self.decoder_num_heads = decoder_num_heads
        self.decoder_window_size = decoder_window_size
        self.decoder_mlp_ratio = decoder_mlp_ratio
        self.decoder_depths = decoder_depths
        self.encoder = self._create_encoder(encoder_type, encoder_weights)
        self.entropy_model = EntropyBottleneck(channels=768) # TODO: this can't be hardcoded
        # TODO: Maybe introduce intermediate linear layer to enhance compression
        self.decoder = SwinTransformerDecoder(
            stage_dims=decoder_dims,
            num_heads=decoder_num_heads,
            windows_sizes=decoder_window_size,
            mlp_ratios=decoder_mlp_ratio,
            depths=decoder_depths,
            sd_factor=decoder_sd_factor
        )

    def _create_encoder(self, encoder_name, weights):
        if encoder_name not in self.ENCODER_MAP:
            raise ValueError(f"Invalid encoder type: {encoder_name}")
        if len({len(field) for field in [
            self.decoder_num_heads,
            self.decoder_window_size,
            self.decoder_mlp_ratio,
            self.decoder_depths
        ]}) != 1:
            raise ValueError(f"All decoder properties must equal number of decoder stages")
        return self.ENCODER_MAP[encoder_name](weights=weights).features

    def forward(self, x):
        y = self.encoder(x)
        y_hat, y_likelihoods = self.entropy_model(y.permute(0, 3, 1, 2))
        x_hat = self.decoder(y_hat.permute(0, 2, 3, 1), x.shape[-2:])
        return x_hat, y_likelihoods