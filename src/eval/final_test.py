from abc import ABC, abstractmethod
from typing import Iterable

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from io import BytesIO

from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from torchmetrics.functional.image import peak_signal_noise_ratio as torch_psnr

from src.data.base_dataset import BaseDataset
from src.data.kodak_dataset import KodakDataset
from src.data.transforms import RGBCompression
from src.models.swin_compression import SwinTransformerCompressionAutoencoder
from src.utils.initializers import model_from_config


class Codec(ABC):
    def __init__(self):
        self.qualities: Iterable | None = None
        self.label: str | None = None
        self.bpp = []
        self.psnr = []

    @abstractmethod
    def encode(self, ref_img: PIL.Image.Image, quality: int, **kwargs) -> tuple[float, float]:
        pass

    def eval_codec(self, dataset: BaseDataset):
        for quality in self.qualities:
            avg_bpp = 0
            avg_psnr = 0
            for ref_img, _ in dataset:
                bpp, psnr = self.encode(ref_img, quality)
                avg_bpp += bpp
                avg_psnr += psnr
            avg_bpp /= len(dataset)
            avg_psnr /= len(dataset)
            self.bpp.append(avg_bpp)
            self.psnr.append(avg_psnr)

class SwinLicCodec(Codec):
    def __init__(self):
        super().__init__()
        self.qualities = [
                "SWIN-B-IC_0.16.0",
                "SWIN-B-IC_0.16.0.1",
                "SWIN-B-IC_0.16.0.2",
                "SWIN-B-IC_0.16.0.3",
                "SWIN-B-IC_0.16.0.4",
                "SWIN-B-IC_0.16.0.5",
                "SWIN-B-IC_0.16.0.6",
                "SWIN-B-IC_0.16.0.7"
            ]

    def encode(self, ref_img: PIL.Image.Image, quality: int, **kwargs) -> tuple[float, float]:
        raise NotImplementedError

    def eval_codec(self, dataset: BaseDataset):
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        dataloader = torch.utils.data.DataLoader(swin_test_dataset, batch_size=1, shuffle=False)

        for quality in self.qualities:
            avg_bpp = torch.tensor(0, dtype=torch.float32).to(device)
            avg_psnr = torch.tensor(0, dtype=torch.float32).to(device)
            model = model_from_config(
                {
                    "module": "src.models.swin_compression.SwinTransformerCompressionAutoencoder",
                    "weights": quality,
                    "args": {
                        "pretrained_encoder": False,
                        "encoder_type": "gdn_swin_v2_b",
                        "encoder_embed_dim": 128,
                        "encoder_dims": [128, 256, 512],
                        "encoder_depths": [2, 6, 24],
                        "encoder_num_heads": [4, 8, 16],
                        "encoder_window_size": [8, 8],
                        "encoder_sd_factor": 0.1,
                        "encoder_mlp_ratio": 4,
                        "decoder_depths": [24, 6, 2],
                        "decoder_dims": [512, 256, 128, 64],
                        "decoder_num_heads": [16, 8, 4],
                        "decoder_window_size": [[8, 8], [8, 8], [8, 8], [8, 8]],
                        "decoder_mlp_ratio": [4, 4, 4, 4],
                        "decoder_sd_factor": 0.05,
                        "bottleneck_dim": 384,
                        "no_compress": False,
                        "checkpointing": False,
                    }
                })
            model.update()
            model.eval()
            model.to(device)
            with torch.no_grad():
                for x_test_in, x_test in dataloader:
                    x_test_in, x_test = x_test_in.to(device).detach(), x_test.to(device).detach()
                    compress_output = model.compress(x_test_in)
                    b_repr, shape = compress_output['strings'], compress_output['shape']
                    x_hat_test = model.decompress(b_repr, shape)['x_hat']

                    psnr = torch_psnr(x_test, x_hat_test, data_range=(0.0, 1.0), dim=(2, 3))
                    avg_psnr += psnr

                    batch_size = len(b_repr[0])
                    bits = sum(len(stream) * 8 for group in b_repr for stream in group)
                    _, _, H, W = x_hat_test.shape
                    bpp = bits / (batch_size * H * W)
                    avg_bpp += bpp

            avg_bpp /= len(dataloader)
            avg_psnr /= len(dataloader)
            self.bpp.append(avg_bpp.to("cpu").numpy())
            self.psnr.append(avg_psnr.to("cpu").numpy())


class PILCodec(Codec, ABC):
    def __init__(self):
        super().__init__()
        self.qualities = range(0, 100, 5)

    def encode(self, ref_img: PIL.Image.Image, quality: int, **kwargs) -> tuple[float, float]:
        with BytesIO() as img_bytes:
            ref_img.save(fp=img_bytes, quality=quality, **kwargs)
            raw_bytes = img_bytes.getvalue()
            bpp = (len(raw_bytes) * 8) / (ref_img.size[0] * ref_img.size[1])
            comp_img = Image.open(BytesIO(raw_bytes))
            ref_data = np.array(ref_img, dtype=np.float64)
            comp_data = np.array(comp_img, dtype=np.float64)
            psnr = sk_psnr(ref_data, comp_data, data_range=255)
        return bpp, psnr

class JPEGCodec(PILCodec):
    def __init__(self):
        super().__init__()
        self.label = 'JPEG'

    def encode(self, ref_img: PIL.Image.Image, quality: int, **kwargs) -> tuple[float, float]:
        return super().encode(ref_img=ref_img, quality=quality, subsampling=2, format='JPEG', **kwargs)

class WebPCodec(PILCodec):
    def __init__(self):
        super().__init__()
        self.label = 'WebP'

    def encode(self, ref_img: PIL.Image.Image, quality: int, **kwargs) -> tuple[float, float]:
        return super().encode(ref_img=ref_img, quality=quality, format='WebP', alpha_quality=0, **kwargs)

class AVIFCodec(PILCodec):
    def __init__(self):
        super().__init__()
        self.label = 'AVIF'

    def encode(self, ref_img: PIL.Image.Image, quality: int, **kwargs) -> tuple[float, float]:
        return super().encode(ref_img=ref_img, quality=quality, format='AVIF', **kwargs)

codecs = [
    JPEGCodec(),
    WebPCodec(),
    AVIFCodec()
]

pil_test_dataset = KodakDataset(None, None, None, None)
swin_test_dataset = KodakDataset(RGBCompression(noresize=True, normalize=True),None, None, None)

for codec in codecs:
    if isinstance(codec, PILCodec):
        codec.eval_codec(pil_test_dataset)
        plt.plot(codec.bpp, codec.psnr, label=codec.label, marker='o')
    else:
        pass
        # codec.eval_codec(pil_test_dataset)

plt.grid(True)
plt.legend()
plt.xlim(0, 1.2)
plt.ylim(20, 38)
plt.show()

