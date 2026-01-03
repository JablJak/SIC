import torch
from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import EntropyBottleneckLatentCodec, HyperpriorLatentCodec, HyperLatentCodec, \
    GaussianConditionalLatentCodec, CheckerboardLatentCodec, ChannelGroupsLatentCodec
from compressai.layers import conv3x3, subpel_conv3x3, CheckerboardMaskedConv2d, sequential_channel_ramp
from compressai.models import SimpleVAECompressionModel
from compressai.models.utils import conv
from matplotlib import pyplot as plt
from networkx.classes import selfloop_edges
from torch import nn, autocast, Tensor
from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b, Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights

from src.models.gdn_swin_transformer import gdn_swin_v2_s, gdn_swin_v2_b
from src.models.swin_autoencoder import SwinTransformerDecoder

class SwinTransformerCompressionAutoencoder(SimpleVAECompressionModel):
    ENCODER_MAP = {
        "swin_v2_t": swin_v2_t,
        "swin_v2_s": swin_v2_s,
        "swin_v2_b": swin_v2_b,
        "gdn_swin_v2_s": gdn_swin_v2_s,
        "gdn_swin_v2_b": gdn_swin_v2_b,
    }

    def __init__(
            self,
            encoder_type = "swin_v2_t",
            encoder_pretrained = False,
            encoder_embed_dim = 96,
            encoder_dims=(96, 192, 288, 384),
            encoder_depths=(2, 2, 18, 2),
            encoder_num_heads=(3, 6, 12, 24),
            encoder_window_size=(8, 8),
            encoder_sd_factor=0.3,
            encoder_mlp_ratio=4,
            encoder_dropout=0,
            encoder_attention_dropout=0,
            decoder_dims = (384, 288, 192, 96, 48),
            decoder_num_heads = (24, 12, 6, 3),
            decoder_window_size = ((8, 8), (8, 8), (8, 8), (8, 8)),
            decoder_mlp_ratio = (4, 4, 4, 4),
            decoder_depths = (2, 6, 2, 2),
            decoder_sd_factor=0.1,
            decoder_dropout=0,
            decoder_attention_dropout=0,
            bottleneck_dim=256,
            no_compress=False,
            checkpointing=False,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.encoder_pretrained = encoder_pretrained
        self.decoder_num_heads = decoder_num_heads
        self.decoder_window_size = decoder_window_size
        self.decoder_mlp_ratio = decoder_mlp_ratio
        self.decoder_depths = decoder_depths
        self.decoder_dropout = decoder_dropout
        self.decoder_attention_dropout = decoder_attention_dropout
        self.g_a = self._create_encoder(
            encoder_name=encoder_type,
            pretrained=encoder_pretrained,
            embed_dim=encoder_embed_dim,
            stage_dims=encoder_dims,
            depths=encoder_depths,
            num_heads=encoder_num_heads,
            window_size=encoder_window_size,
            stochastic_depth_prob=encoder_sd_factor,
            mlp_ratio=encoder_mlp_ratio,
            dropout=encoder_dropout,
            attention_dropout=encoder_attention_dropout,
            checkpointing=checkpointing,
            bottleneck_dim=bottleneck_dim,
        )
        self.g_s = SwinTransformerDecoder(
            stage_dims=decoder_dims,
            num_heads=decoder_num_heads,
            windows_sizes=decoder_window_size,
            mlp_ratios=decoder_mlp_ratio,
            depths=decoder_depths,
            sd_factor=decoder_sd_factor,
            dropout=decoder_dropout,
            attention_dropout=decoder_attention_dropout,
            checkpointing=checkpointing,
            bottleneck_dim=bottleneck_dim,
        )
        self.no_compress = no_compress
        self.checkpointing = checkpointing
        self.bottleneck_dim = bottleneck_dim
        if not no_compress:
            N = bottleneck_dim
            M = bottleneck_dim // 2
            h_a = nn.Sequential(
                conv3x3(N, N),
                nn.GELU(),
                conv3x3(N, N),
                nn.GELU(),
                conv3x3(N, N, stride=2),
                nn.GELU(),
                conv3x3(N, N),
                nn.GELU(),
                conv3x3(N, M, stride=2),
            )

            h_s = nn.Sequential(
                conv3x3(M, N),
                nn.GELU(),
                subpel_conv3x3(N, N, 2),
                nn.GELU(),
                conv3x3(N, N * 3 // 2),
                nn.GELU(),
                subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
                nn.GELU(),
                conv3x3(N * 3 // 2, N * 2),
            )
            self.latent_codec = HyperpriorLatentCodec(
                latent_codec={
                    "y": CheckerboardLatentCodec(
                        latent_codec={
                            "y": GaussianConditionalLatentCodec(quantizer="noise"),
                        },
                        entropy_parameters=nn.Sequential(
                            nn.Conv2d(N * 12 // 3, N * 10 // 3, 1),
                            nn.GELU(),
                            nn.Conv2d(N * 10 // 3, N * 8 // 3, 1),
                            nn.GELU(),
                            nn.Conv2d(N * 8 // 3, N * 6 // 3, 1),
                        ),
                        context_prediction=CheckerboardMaskedConv2d(
                            N, 2 * N, kernel_size=5, stride=1, padding=2
                        ),
                    ),
                    "hyper": HyperLatentCodec(
                        entropy_bottleneck=EntropyBottleneck(M),
                        h_a=h_a,
                        h_s=h_s,
                        quantizer="noise",
                    ),
                }
            )

    def _create_encoder(self,
        encoder_name, 
        pretrained,
        embed_dim,
        stage_dims,
        depths,
        num_heads,
        window_size,
        stochastic_depth_prob,
        mlp_ratio,
        dropout,
        attention_dropout,
        checkpointing,
        bottleneck_dim,
    ):
        match encoder_name:
            case "swin_v2_t":
                weights = Swin_V2_T_Weights.DEFAULT
            case "swin_v2_s":
                weights = Swin_V2_S_Weights.DEFAULT
            case "swin_v2_b":
                weights = Swin_V2_B_Weights.DEFAULT
            case "gdn_swin_v2_s":
                weights = Swin_V2_S_Weights.DEFAULT
            case "gdn_swin_v2_b":
                weights = Swin_V2_B_Weights.DEFAULT
        return self.ENCODER_MAP[encoder_name](
            weights=(weights if pretrained else None),
            embed_dim=embed_dim,
            stage_dims=stage_dims,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            stochastic_depth_prob=stochastic_depth_prob,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            checkpointing=checkpointing,
            bottleneck_dim=bottleneck_dim,
        ).features if not encoder_name.startswith("gdn") else \
            self.ENCODER_MAP[encoder_name](
            weights=(weights if pretrained else None),
            embed_dim=embed_dim,
            stage_dims=stage_dims,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            stochastic_depth_prob=stochastic_depth_prob,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            checkpointing=checkpointing,
            bottleneck_dim=bottleneck_dim,
            )

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

    def forward(self, x: Tensor):
        y = self.g_a(x)
        if not self.no_compress:
            y = y.permute(0, 3, 1, 2)
            y_out = self.latent_codec(y)
            y_hat = y_out["y_hat"]
            y_hat = y_hat.permute(0, 2, 3, 1)
            x_hat = self.g_s(y_hat)
            return {
                "x_hat": x_hat,
                "likelihoods": y_out["likelihoods"],
            }
        else:
            x_hat = self.g_s(y)
            return {
                "x_hat": x_hat,
                "likelihoods": None,
            }


    def compress(self, x):
        y = self.g_a(x)
        plt.hist(y.cpu().numpy(), bins=100, density=True, alpha=0.6, label='Faktyczne y (Encoder)')
        plt.show()
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

    def parameters(self, aux=False, named=False):
        main_parameters = [
            (name, param) if named else param
            for name, param in self.named_parameters()
            if param.requires_grad and not name.endswith(".quantiles")
        ]

        aux_parameters = [
            (name, param) if named else param
            for name, param in self.named_parameters()
            if param.requires_grad and name.endswith(".quantiles")
        ]
        if aux:
            return iter(aux_parameters)
        else:
            return iter(main_parameters)

