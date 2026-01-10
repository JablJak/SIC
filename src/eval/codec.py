import os
import subprocess
from abc import ABC
from datetime import datetime
from typing import Iterable

import PIL.Image
import numpy as np
import torch
from io import BytesIO

from PIL import Image
from PIL.ImageFile import ImageFile
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from torchmetrics.functional.image import peak_signal_noise_ratio as torch_psnr
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure as torch_msssim
from torchvision.transforms.v2.functional import to_pil_image, to_image, to_dtype

from src.data.base_dataset import BaseDataset
from src.data.kodak_dataset import KodakDataset
from src.data.transforms import RGBCompression, RGB_MIXED_LIC_MEAN, RGB_MIXED_LIC_STD

from src.utils.initializers import model_from_config

pil_test_dataset = KodakDataset(None, None, None, None)
swin_test_dataset = KodakDataset(RGBCompression(noresize=True, normalize=True, mean=RGB_MIXED_LIC_MEAN, std=RGB_MIXED_LIC_STD)
                                 ,None, None, None)
class Codec():
    def __init__(self, qualities = None, label = None, bpp = None, psnr = None, msssim = None, type = None):
        self.qualities: Iterable | None = qualities
        self.label: str | None = label
        self.bpp = [] if bpp is None else bpp
        self.psnr = [] if psnr is None else psnr
        self.msssim = [] if msssim is None else msssim
        self.type = type

    def encode(self, ref_img: PIL.Image.Image, quality: int | float, **kwargs) -> tuple[ImageFile, float]:
        raise NotImplementedError

    def eval_codec(self, dataset: BaseDataset, out_dir: str = None):
        for quality in self.qualities:
            avg_bpp = 0
            avg_psnr = 0
            avg_msssim = 0
            for i, (ref_img, _) in enumerate(dataset):
                comp_img, bpp = self.encode(ref_img, quality=quality)
                ref_data = np.array(ref_img, dtype=np.float64)
                comp_data = np.array(comp_img, dtype=np.float64)
                psnr = sk_psnr(ref_data, comp_data, data_range=255)
                torch_comp_img = to_image(comp_img)
                torch_comp_img = to_dtype(torch_comp_img, dtype=torch.float32, scale=True).unsqueeze(0)
                torch_ref_img = to_image(ref_img)
                torch_ref_img = to_dtype(torch_ref_img, dtype=torch.float32, scale=True).unsqueeze(0)
                msssim = torch_msssim(torch_comp_img, torch_ref_img, data_range=1.0).to("cpu").numpy().item()
                full_out_dir = out_dir / self.label / str(quality)
                full_out_dir.mkdir(parents=True, exist_ok=True)
                ref_img.save(out_dir / self.label / str(quality) / f'ref_{i + 1}.png')
                comp_img.save(out_dir / self.label / str(quality) / f'comp_{i + 1}.png')
                avg_bpp += bpp
                avg_psnr += psnr
                avg_msssim += msssim
            avg_bpp /= len(dataset)
            avg_psnr /= len(dataset)
            avg_msssim /= len(dataset)
            self.bpp.append(round(avg_bpp, 3))
            self.psnr.append(round(avg_psnr, 3))
            self.msssim.append(round(avg_msssim, 3))
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Evaluated quality {quality} on codec {self.label}, "
                  f"bpp: {avg_bpp}, psnr: {avg_psnr}, msssim: {avg_msssim}")

class SwinLicCodec(Codec):
    def __init__(self):
        super().__init__()
        self.qualities = [
                "SWIN-LIC_1.0.1",
                "SWIN-LIC_1.0.2",
                "SWIN-LIC_1.0.3",
                "SWIN-LIC_1.0.4",
                "SWIN-LIC_1.0.5",
                "SWIN-LIC_1.0.6",
                "SWIN-LIC_1.0.7",
                "SWIN-LIC_1.0.8",
                "SWIN-LIC_1.0.9"
            ]
        self.label = "SwinLIC"

    def encode(self, ref_img: PIL.Image.Image, quality: int | float, **kwargs) -> tuple[float, float, float]:
        raise NotImplementedError

    def eval_codec(self, dataset: BaseDataset, out_dir: str = None):
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        dataloader = torch.utils.data.DataLoader(swin_test_dataset, batch_size=1, shuffle=False)

        for quality in self.qualities:
            avg_bpp = torch.tensor(0, dtype=torch.float32).to(device)
            avg_psnr = torch.tensor(0, dtype=torch.float32).to(device)
            avg_msssim = torch.tensor(0, dtype=torch.float32).to(device)
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
                        "encoder_sd_factor": 0.05,
                        "encoder_mlp_ratio": 4,
                        "encoder_dropout": 0,
                        "encoder_attention_dropout": 0,
                        "decoder_depths": [24, 6, 2],
                        "decoder_dims": [512, 512, 256, 128],
                        "decoder_num_heads": [16, 8, 4],
                        "decoder_window_size": [[8, 8], [8, 8], [8, 8]],
                        "decoder_mlp_ratio": [4, 4, 4],
                        "decoder_sd_factor": 0.03,
                        "decoder_dropout": 0,
                        "decoder_attention_dropout": 0,
                        "bottleneck_dim": 384,
                        "no_compress": False,
                        "checkpointing": False,
                    }
                })
            model.update()
            model.eval()
            model.to(device)
            with torch.no_grad():
                for i, (x_test_in, x_test) in enumerate(dataloader):
                    x_test_in, x_test = x_test_in.to(device).detach(), x_test.to(device).detach()
                    compress_output = model.compress(x_test_in)
                    b_repr, shape = compress_output['strings'], compress_output['shape']
                    x_hat_test = model.decompress(b_repr, shape)['x_hat']

                    psnr = torch_psnr(x_test, x_hat_test, data_range=(0.0, 1.0), dim=(2, 3))
                    avg_psnr += psnr

                    msssim = torch_msssim(x_test, x_hat_test, data_range=1.0, reduction="elementwise_mean").mean()
                    avg_msssim += msssim

                    batch_size = len(b_repr[0])
                    bits = sum(len(stream) * 8 for group in b_repr for stream in group)
                    _, _, H, W = x_hat_test.shape
                    bpp = bits / (batch_size * H * W)
                    avg_bpp += bpp
                    ref_img: PIL.Image.Image = to_pil_image(x_test.detach().cpu().squeeze())
                    comp_img: PIL.Image.Image = to_pil_image(x_hat_test.detach().cpu().squeeze())
                    full_out_dir = out_dir / self.label / str(quality)
                    full_out_dir.mkdir(parents=True, exist_ok=True)
                    ref_img.save(out_dir / self.label / str(quality) / f'ref_{i + 1}.png')
                    comp_img.save(out_dir / self.label / str(quality) / f'comp_{i + 1}.png')

            avg_bpp /= len(dataloader)
            avg_psnr /= len(dataloader)
            avg_msssim /= len(dataloader)
            self.bpp.append(round(avg_bpp.cpu().numpy().item(), 3))
            self.psnr.append(round(avg_psnr.cpu().numpy().item(), 3))
            self.msssim.append(round(avg_msssim.cpu().numpy().item(), 3))
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Evaluated quality {quality} on codec {self.label}, "
                  f"bpp: {avg_bpp}, psnr: {avg_psnr}, msssim: {avg_msssim}")

class BPGCodec(Codec):
    def __init__(self):
        super().__init__()
        self.label = "BPG"
        self.qualities = list(range(51, 23, -2))

    def encode(self, ref_img: PIL.Image.Image, quality: int, **kwargs) -> tuple[ImageFile, float]:
        input_path = os.path.join("/dev/shm", f"input_{os.getpid()}.png")
        output_path = os.path.join("/dev/shm", f"output_{os.getpid()}.bpg")

        ref_img.save(input_path, format="PNG")
        try:
            subprocess.run(
                ["bpgenc", "-f", "444", "-c", "ycbcr", "-q", str(quality), "-o", output_path, input_path],
                capture_output=True,
                check=True
            )
            with open(output_path, "rb") as f:
                bpg_bytes = f.read()
            bpp = len(bpg_bytes) * 8 / (ref_img.size[0] * ref_img.size[1])
            decode_res = subprocess.run(
                ["bpgdec", "-o", "/dev/stdout", output_path],
                capture_output=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print("libbpg error:")
            print(e.stderr.decode())
            raise
        comp_img = Image.open(BytesIO(decode_res.stdout)).convert('RGB')
        return comp_img, bpp

class PILCodec(Codec, ABC):
    def __init__(self):
        super().__init__()
        self.qualities = list(range(0, 100, 5))

    def encode(self, ref_img: PIL.Image.Image, quality: int,  out_dir: str = None, **kwargs) -> tuple[ImageFile, float]:
        with BytesIO() as img_bytes:
            ref_img.save(fp=img_bytes, quality=quality, **kwargs)
            raw_bytes = img_bytes.getvalue()
            bpp = (len(raw_bytes) * 8) / (ref_img.size[0] * ref_img.size[1])
            comp_img = Image.open(BytesIO(raw_bytes))
        return comp_img, bpp



class JPEGCodec(PILCodec):
    def __init__(self):
        super().__init__()
        self.qualities = list(range(0, 70, 5))
        self.label = 'JPEG'

    def encode(self, ref_img: PIL.Image.Image, quality: int, **kwargs) -> tuple[ImageFile, float]:
        return super().encode(ref_img=ref_img, quality=quality, subsampling=2, format='JPEG', **kwargs)

class JPEG2000Codec(PILCodec):
    def __init__(self):
        super().__init__()
        self.qualities = list([q / 100 for q in range(5, 150, 5)])
        self.label = 'JPEG2000'

    def encode(self, ref_img: PIL.Image.Image, quality: float, **kwargs) -> tuple[ImageFile, float]:
        with BytesIO() as img_bytes:
            ref_img.save(fp=img_bytes, quality_layers=[24/quality], format="JPEG2000", quality_mode="rates",
                         irreversible=True, mct=1, **kwargs)
            raw_bytes = img_bytes.getvalue()
            bpp = (len(raw_bytes) * 8) / (ref_img.size[0] * ref_img.size[1])
            comp_img = Image.open(BytesIO(raw_bytes))
        return comp_img, bpp

class WebPCodec(PILCodec):
    def __init__(self):
        super().__init__()
        self.qualities = list(range(0, 90, 5))
        self.label = 'WebP'

    def encode(self, ref_img: PIL.Image.Image, quality: int, **kwargs) -> tuple[ImageFile, float]:
        return super().encode(ref_img=ref_img, quality=quality, format='WebP', alpha_quality=0, **kwargs)

class AVIFCodec(PILCodec):
    def __init__(self):
        super().__init__()
        self.qualities = list(range(0, 70, 5))
        self.label = 'AVIF'

    def encode(self, ref_img: PIL.Image.Image, quality: int, **kwargs) -> tuple[ImageFile, float]:
        return super().encode(ref_img=ref_img, quality=quality, speed=0, format='AVIF', **kwargs)

