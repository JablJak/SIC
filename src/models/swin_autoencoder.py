from compressai.entropy_models import EntropyBottleneck
from timm.layers import DropPath
from torch.nn import init
from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b, Swin_V2_T_Weights
from torch import nn, Tensor
from torchvision.models.swin_transformer import ShiftedWindowAttentionV2


class PatchReconstruction(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.conv_trans = nn.ConvTranspose2d(96, 3, kernel_size=patch_size, stride=patch_size, padding=0)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv_trans(x)
        return x


class SwinTransformerDecoder(nn.Module):
    def __init__(
            self,
            dim,
            patch_size,
            num_heads,
            window_size,
            mlp_ratio,
            depths,
        ):
        super().__init__()
        self.stage1 = SwinTransformerDecoderStage(dim * 8, num_heads[0], window_size[0], mlp_ratio[0], depths[0])
        self.stage2 = SwinTransformerDecoderStage(dim * 4, num_heads[1], window_size[1], mlp_ratio[1], depths[1])
        self.stage3 = SwinTransformerDecoderStage(dim * 2, num_heads[2], window_size[2], mlp_ratio[2], depths[2])
        self.stage4 = SwinTransformerDecoderStage(dim, num_heads[3], window_size[3], mlp_ratio[3], depths[3], split=False)
        self.reconstruction = PatchReconstruction(patch_size)

    def forward(self, x, img_size: tuple[int, int]):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        H, W = img_size
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
        split = True
    ):
        super().__init__()
        self.depth = depth
        self.split = split
        self.upsample = nn.PixelShuffle(2)
        self.split_norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim * 2)
        self.blocks = nn.ModuleList(
            [SwinTransformerDecoderBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0 if i_block % 2 == 0 else w // 2 for w in window_size],
                mlp_ratio=mlp_ratio,
            ) for i_block in (range(depth) if split else range(depth - 1))]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in (range(depth) if split else range(depth - 1))])
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        for i in (range(self.depth) if self.split else range(self.depth - 1)):
            x = x + self.blocks[i](self.norms[i](x))
        if self.split:
            x = self.linear(self.split_norm(x))
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
        self.drop_path = DropPath(0.1)

        # TODO: Initialize rest of the layers

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x: Tensor):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
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
            decoder_patch_size = 4,
            decoder_num_heads = (24, 12, 6, 3),
            decoder_window_size = ((7, 7), (7, 7), (7, 7), (7, 7)),
            decoder_mlp_ratio = (4, 4, 4, 4),
            decoder_depths = (2, 6, 2, 2),
    ):
        super().__init__()
        self.encoder_weights = encoder_weights
        self.encoder = self._create_encoder(encoder_type, encoder_weights)
        self.entropy_model = EntropyBottleneck(channels=768) # TODO: this can't be hardcoded
        # TODO: Maybe introduce intermediate linear layer to enhance compression

        self.decoder = SwinTransformerDecoder(
            dim=decoder_dim,
            patch_size=decoder_patch_size,
            num_heads=decoder_num_heads,
            window_size=decoder_window_size,
            mlp_ratio=decoder_mlp_ratio,
            depths=decoder_depths,
        )

    def _create_encoder(self, encoder_name, weights):
        if encoder_name not in self.ENCODER_MAP:
            raise ValueError(f"Invalid encoder type: {encoder_name}")
        return self.ENCODER_MAP[encoder_name](weights=weights).features

    def forward(self, x):
        y = self.encoder(x)
        y_hat, y_likelihoods = self.entropy_model(y.permute(0, 3, 1, 2))
        x_hat = self.decoder(y_hat.permute(0, 2, 3, 1), x.shape[-2:])
        return x_hat, y_likelihoods