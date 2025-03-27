from compressai.latent_codecs import EntropyBottleneckLatentCodec
from compressai.models import SimpleVAECompressionModel
from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b, Swin_V2_T_Weights

from src.models.swin_autoencoder import SwinTransformerDecoder


class SwinTransformerCompressionAutoencoder(SimpleVAECompressionModel):
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
        self.encoder_type = encoder_type
        self.encoder_weights = encoder_weights
        # TODO: Maybe introduce intermediate linear layer to enhance compression
        self.decoder_num_heads = decoder_num_heads
        self.decoder_window_size = decoder_window_size
        self.decoder_mlp_ratio = decoder_mlp_ratio
        self.decoder_depths = decoder_depths
        self.encoder = self.g_a = self._create_encoder(encoder_type, encoder_weights)
        self.decoder = self.g_s = SwinTransformerDecoder(
            dim=decoder_dim,
            patch_sizes=decoder_patch_size,
            num_heads=decoder_num_heads,
            windows_sizes=decoder_window_size,
            mlp_ratios=decoder_mlp_ratio,
            depths=decoder_depths,
        )
        self.latent_codec = EntropyBottleneckLatentCodec(channels=768)

    def _create_encoder(self, encoder_name, weights):
        return self.ENCODER_MAP[encoder_name](weights=weights).features

    def _validate_args(self):
        if self.encoder_type not in self.ENCODER_MAP:
            raise ValueError(f"Invalid encoder type: {self.encoder_type}")
        if len({len(field) for field in [
            self.decoder_num_heads,
            self.decoder_window_size,
            self.decoder_mlp_ratio,
            self.decoder_depths
        ]}) != 1:
            raise ValueError(f"All decoder properties must equal number of decoder stages")

    def forward(self, x):
        y = self.g_a(x)
        y = y.permute(0, 3, 1, 2)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]
        y_hat = y_hat.permute(0, 2, 3, 1)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": y_out["likelihoods"],
        }
