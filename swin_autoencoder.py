from timm.layers import DropPath
from torch.nn import init
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
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
            dim = 96,
            patch_size = 4,
            num_heads = (24, 12, 6, 3),
            window_size = ((7, 7), (7, 7), (7, 7), (7, 7)),
            mlp_ratio = (4, 4, 4, 4),
            depths = (2, 6, 2, 2),
        ):
        super().__init__()
        self.stage1 = SwinTransformerDecoderStage(dim * 8, num_heads[0], window_size[0], mlp_ratio[0])
        self.stage2 = SwinTransformerDecoderStage(dim * 4, num_heads[1], window_size[1], mlp_ratio[1])
        self.stage3 = SwinTransformerDecoderStage(dim * 2, num_heads[2], window_size[2], mlp_ratio[2])
        self.stage4 = SwinTransformerDecoderStage(dim, num_heads[3], window_size[3], mlp_ratio[3], split=False)
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
        split = True
    ):
        super().__init__()
        self.upsample = nn.PixelShuffle(2)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim * 2)
        self.block = SwinTransformerDecoderBlock(
            dim,
            num_heads,
            window_size,
            mlp_ratio
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.split = split

    def forward(self, x):
        x = self.block(self.norm1(x))
        if self.split:
            x = self.linear(self.norm2(x))
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
        mlp_ratio=4,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = ShiftedWindowAttentionV2(dim=dim, window_size=window_size, num_heads=num_heads, shift_size=[0, 0])
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.drop_path = DropPath(0.1)
        # Inicjalizacja wag
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
    def __init__(self, encoder_weights=Swin_V2_T_Weights.DEFAULT):
        super().__init__()
        self.encoder_weights = encoder_weights
        self.encoder = swin_v2_t(weights=encoder_weights).features
        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        # self.linear TODO: introduce immediate linear layer to enhance compression
        self.decoder = SwinTransformerDecoder()
        # for param in self.decoder.parameters():
        #     param.requires_grad = True

    def forward(self, x):
        encoder_features = self.encoder(x)
        decoder_features = self.decoder(encoder_features, x.shape[-2:])
        return decoder_features