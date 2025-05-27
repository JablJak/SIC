import torch
from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import EntropyBottleneckLatentCodec, HyperpriorLatentCodec, HyperLatentCodec, \
    GaussianConditionalLatentCodec, CheckerboardLatentCodec, ChannelGroupsLatentCodec
from compressai.layers import conv3x3, subpel_conv3x3, CheckerboardMaskedConv2d, sequential_channel_ramp
from compressai.models import SimpleVAECompressionModel
from compressai.models.utils import conv
from networkx.classes import selfloop_edges
from torch import nn, autocast, Tensor
from torchvision.models import swin_v2_t, swin_v2_s, swin_v2_b, Swin_V2_T_Weights, Swin_V2_S_Weights, Swin_V2_B_Weights

from src.models.gdn_swin_transformer import gdn_swin_v2_s
from src.models.swin_autoencoder import SwinTransformerDecoder

class SwinTransformerCompressionAutoencoder(SimpleVAECompressionModel):
    ENCODER_MAP = {
        "swin_v2_t": swin_v2_t,
        "swin_v2_s": swin_v2_s,
        "swin_v2_b": swin_v2_b,
        "gdn_swin_v2_s": gdn_swin_v2_s,
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
            decoder_dims = (384, 288, 192, 96, 48),
            decoder_num_heads = (24, 12, 6, 3),
            decoder_window_size = ((8, 8), (8, 8), (8, 8), (8, 8)),
            decoder_mlp_ratio = (4, 4, 4, 4),
            decoder_depths = (2, 6, 2, 2),
            decoder_sd_factor=0.1,
            no_compress=False,

    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.encoder_pretrained = encoder_pretrained
        self.decoder_num_heads = decoder_num_heads
        self.decoder_window_size = decoder_window_size
        self.decoder_mlp_ratio = decoder_mlp_ratio
        self.decoder_depths = decoder_depths
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
        )
        self.g_s = SwinTransformerDecoder(
            stage_dims=decoder_dims,
            num_heads=decoder_num_heads,
            windows_sizes=decoder_window_size,
            mlp_ratios=decoder_mlp_ratio,
            depths=decoder_depths,
            sd_factor=decoder_sd_factor,
        )
        self.no_compress = no_compress
        if not no_compress:
            N = encoder_dims[-1]
            M = N
            groups = [48, 48, 96, 192, M - 384]

            self.groups = list(groups)
            assert sum(self.groups) == M
            # h_a = nn.Sequential(
            #     conv3x3(N, N),
            #     nn.GELU(),
            #     conv3x3(N, N),
            #     nn.GELU(),
            #     conv3x3(N, N, stride=2),
            #     nn.GELU(),
            #     conv3x3(N, N),
            #     nn.GELU(),
            #     conv3x3(N, N, stride=2),
            # )
            #
            # h_s = nn.Sequential(
            #     conv3x3(N, N),
            #     nn.GELU(),
            #     subpel_conv3x3(N, N, 2),
            #     nn.GELU(),
            #     conv3x3(N, N * 3 // 2),
            #     nn.GELU(),
            #     subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            #     nn.GELU(),
            #     conv3x3(N * 3 // 2, N * 2),
            # )
            # # In [He2022], this is labeled "g_ch^(k)".
            # channel_context = {
            #     f"y{k}": nn.Sequential(
            #         conv(sum(self.groups[:k]), 224, kernel_size=5, stride=1),
            #         nn.ReLU(inplace=True),
            #         conv(224, 128, kernel_size=5, stride=1),
            #         nn.ReLU(inplace=True),
            #         conv(128, self.groups[k] * 2, kernel_size=5, stride=1),
            #     )
            #     for k in range(1, len(self.groups))
            # }
            #
            # # In [He2022], this is labeled "g_sp^(k)".
            # spatial_context = [
            #     CheckerboardMaskedConv2d(
            #         self.groups[k],
            #         self.groups[k] * 2,
            #         kernel_size=5,
            #         stride=1,
            #         padding=2,
            #     )
            #     for k in range(len(self.groups))
            # ]
            #
            # # In [He2022], this is labeled "Param Aggregation".
            # param_aggregation = [
            #     sequential_channel_ramp(
            #         # Input: spatial context, channel context, and hyper params.
            #         self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + N * 2,
            #         self.groups[k] * 2,
            #         min_ch=N * 2,
            #         num_layers=3,
            #         interp="linear",
            #         make_layer=nn.Conv2d,
            #         make_act=lambda: nn.ReLU(inplace=True),
            #         kernel_size=1,
            #         stride=1,
            #         padding=0,
            #     )
            #     for k in range(len(self.groups))
            # ]
            #
            # # In [He2022], this is labeled the space-channel context model (SCCTX).
            # # The side params and channel context params are computed externally.
            # scctx_latent_codec = {
            #     f"y{k}": CheckerboardLatentCodec(
            #         latent_codec={
            #             "y": GaussianConditionalLatentCodec(quantizer="ste"),
            #         },
            #         context_prediction=spatial_context[k],
            #         entropy_parameters=param_aggregation[k],
            #     )
            #     for k in range(len(self.groups))
            # }
            #
            # # [He2022] uses a "hyperprior" architecture, which reconstructs y using z.
            # self.latent_codec = HyperpriorLatentCodec(
            #     latent_codec={
            #         # Channel groups with space-channel context model (SCCTX):
            #         "y": ChannelGroupsLatentCodec(
            #             groups=self.groups,
            #             channel_context=channel_context,
            #             latent_codec=scctx_latent_codec,
            #         ),
            #         # Side information branch containing z:
            #         "hyper": HyperLatentCodec(
            #             entropy_bottleneck=EntropyBottleneck(N),
            #             h_a=h_a,
            #             h_s=h_s,
            #             quantizer="ste",
            #         ),
            #     },
            # )
            N = encoder_dims[-1]
            h_a = nn.Sequential(
                conv3x3(N, N),
                nn.GELU(),
                conv3x3(N, N),
                nn.GELU(),
                conv3x3(N, N, stride=2),
                nn.GELU(),
                conv3x3(N, N),
                nn.GELU(),
                conv3x3(N, N, stride=2),
            )

            h_s = nn.Sequential(
                conv3x3(N, N),
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
                            "y": GaussianConditionalLatentCodec(quantizer="ste"),
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
                        entropy_bottleneck=EntropyBottleneck(N),
                        h_a=h_a,
                        h_s=h_s,
                        quantizer="ste",
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
        return self.ENCODER_MAP[encoder_name](
            weights=(weights if pretrained else None),
            embed_dim=embed_dim,
            stage_dims=stage_dims,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            stochastic_depth_prob=stochastic_depth_prob,
            mlp_ratio=mlp_ratio,
        ).features if encoder_name != "gdn_swin_v2_s" else \
            self.ENCODER_MAP[encoder_name](
            weights=(weights if pretrained else None),
            embed_dim=embed_dim,
            stage_dims=stage_dims,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            stochastic_depth_prob=stochastic_depth_prob,
            mlp_ratio=mlp_ratio,
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
            with autocast(device_type=x.device.type, enabled=False):
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

