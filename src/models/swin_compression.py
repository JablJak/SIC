from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import EntropyBottleneckLatentCodec, HyperpriorLatentCodec, HyperLatentCodec, \
    GaussianConditionalLatentCodec, CheckerboardLatentCodec
from compressai.layers import conv3x3, subpel_conv3x3, CheckerboardMaskedConv2d
from compressai.models import SimpleVAECompressionModel
from torch import nn
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
            decoder_window_size = ((8, 8), (8, 8), (8, 8), (8, 8)),
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
            num_heads=decoder_num_heads,
            windows_sizes=decoder_window_size,
            mlp_ratios=decoder_mlp_ratio,
            depths=decoder_depths,
        )
        N = 768
        h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )
        self.latent_codec = HyperpriorLatentCodec(
            latent_codec={
                "y": CheckerboardLatentCodec(
                    latent_codec={
                        "y": GaussianConditionalLatentCodec(quantizer="ste"),
                    },
                    entropy_parameters=nn.Sequential(
                        nn.Conv2d(N * 12 // 3, N * 10 // 3, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(N * 10 // 3, N * 8 // 3, 1),
                        nn.LeakyReLU(inplace=True),
                        nn.Conv2d(N * 8 // 3, N * 6 // 3, 1),
                    ),
                    context_prediction=CheckerboardMaskedConv2d(
                        N, 2 * N, kernel_size=5, stride=1, padding=2
                    ),
                ),
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            }
        )

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

    def compress(self, x):
        y = self.g_a(x)
        y = y.permute(0, 3, 1, 2)
        outputs = self.latent_codec.compress(y)
        return outputs

    def decompress(self, *args, **kwargs):
        y_out = self.latent_codec.decompress(*args, **kwargs)
        y_hat = y_out["y_hat"]
        y_hat = y_hat.permute(0, 2, 3, 1)
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
        }

    def a_s_parameters(self):
        aux_params_list = list(self.latent_codec.parameters())
        aux_params_ids = {id(p) for p in aux_params_list}
        main_params = [p for p in self.parameters() if id(p) not in aux_params_ids]
        return main_params