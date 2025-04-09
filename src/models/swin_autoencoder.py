from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
from timm.layers import DropPath
from torch.nn import init
from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b, Swin_V2_T_Weights
from torch import nn, Tensor
from torchvision.models.swin_transformer import ShiftedWindowAttentionV2
from torchvision.ops import StochasticDepth


class PatchReconstruction(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_trans = nn.ConvTranspose2d(48, 3, kernel_size=4, stride=2, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv_trans(x)
        x = self.activation(x)
        return x


class SwinTransformerDecoder(nn.Module):
    def __init__(
            self,
            dim,
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
                    dim=dim * 2 ** (n - i - 1),
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
        dim,
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
        self.split_norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim * 2)
        self.blocks = nn.ModuleList(
            [SwinTransformerDecoderBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0 if i_block % 2 == 0 else w // 2 for w in window_size],
                mlp_ratio=mlp_ratio,
                sd_factor=sd_factor * ((blocks_remaining - i_block) / blocks_total),
            ) for i_block in range(depth)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.igdn = GDN(dim // 2, inverse=True)

    def forward(self, x):
        for i in range(self.depth):
            x = x + self.blocks[i](self.norms[i](x))
        x = self.linear(self.split_norm(x))
        x = x.permute(0, 3, 1, 2)
        x = self.pixel_shuffle(x)
        x = self.igdn(x)
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
        mlp_ratio=4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = ShiftedWindowAttentionV2(dim=dim, window_size=window_size, num_heads=num_heads, shift_size=shift_size)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.stochastic_depth = StochasticDepth(sd_factor, "row")

        # TODO: Initialize rest of the layers

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
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
            decoder_dim = 96,
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
            dim=decoder_dim,
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