import json
from datetime import timezone, datetime

import matplotlib.pyplot as plt

from src.data.kodak_dataset import KodakDataset
from src.data.transforms import RGBCompression, RGB_MIXED_LIC_MEAN, RGB_MIXED_LIC_STD
from src.eval.codec import JPEGCodec, JPEG2000Codec, WebPCodec, AVIFCodec, SwinLicCodec, BPGCodec, PILCodec
from src.utils.const import ARTIFACTS_PATH



jpeg_codec = JPEGCodec()
jpeg2000_codec = JPEG2000Codec()
webp_codec = WebPCodec()
avif_codec = AVIFCodec()
bpg_codec = BPGCodec()
swin_lic_codec = SwinLicCodec()

codecs = [
    # jpeg_codec,
    # jpeg2000_codec,
    # webp_codec,
    # avif_codec,
    # bpg_codec,
    swin_lic_codec,
]

pil_test_dataset = KodakDataset(None, None, None, None)
swin_test_dataset = KodakDataset(RGBCompression(noresize=True, normalize=False, mean=RGB_MIXED_LIC_MEAN, std=RGB_MIXED_LIC_STD)
                                 ,None, None, None)

artifacts_dir = ARTIFACTS_PATH / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M") + "-14")
artifacts_dir.mkdir(parents=True, exist_ok=True)

px = 1/plt.rcParams['figure.dpi']  # pixel in inches
plt.figure(figsize=(900*px, 600*px))
for codec in codecs:
    # if isinstance(codec, (PILCodec, BPGCodec)):
    #     codec.eval_codec(pil_test_dataset, out_dir=artifacts_dir)
    if isinstance(codec, SwinLicCodec):
        codec.eval_codec(swin_test_dataset, out_dir=artifacts_dir)
    plt.plot(codec.bpp, codec.psnr, label=codec.label, marker='o')

plt.grid(True)
plt.legend()
plt.xlim(0, 1.2)
plt.ylim(20, 40)
plt.yticks(range(20, 40, 2))
plt.xlabel('Szybkość bitowa [bpp]')
plt.ylabel('PSNR [dB]')
plt.margins(0, 0)
plt.tight_layout()
plt.savefig(artifacts_dir / "rd_psnr.png")

plt.figure(figsize=(900*px, 600*px))
for codec in codecs:
    plt.plot(codec.bpp, codec.msssim, label=codec.label, marker='o')

plt.grid(True)
plt.legend()
plt.xlim(0, 1.2)
plt.xlabel('Szybkość bitowa [bpp]')
plt.ylabel('MS-SSIM')
plt.margins(0, 0)
plt.tight_layout()
plt.savefig(artifacts_dir / "rd_ms_ssim.png")

with open(artifacts_dir / "data.json", "w", encoding="utf-8") as f:
    json.dump(codecs, f, indent=4, default=lambda o: o.__dict__)
