import matplotlib.pyplot as plt
import numpy as np
import torch
from io import BytesIO

from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from torchmetrics.functional.image import peak_signal_noise_ratio as torch_psnr

from src.data.kodak_dataset import KodakDataset
from src.data.transforms import RGBCompression
from src.models.swin_compression import SwinTransformerCompressionAutoencoder
from src.utils.initializers import model_from_config


def jpeg_dict():
    return {
        "JPEG": {
            'qualities': range(0, 100, 5),
            # 'bpp': [0.1731050279405382, 0.22121853298611113, 0.3267220391167535, 0.4233779907226562, 0.5086839463975694, 0.5881610446506077, 0.6602283053927951, 0.7292234632703992, 0.7862777709960939, 0.8500467936197916, 0.9063084920247396, 0.9648200141059027, 1.0377833048502605, 1.1280398898654516, 1.2398461235894096, 1.3698467678493926, 1.5734007093641498, 1.8588163587782123, 2.3501993815104165, 3.4013400607638893],
            'bpp': [],
            # 'psnr': [21.41314338821495, 23.889897940392633, 26.649062585638074, 28.08155691302055, 29.076905609682356, 29.81270646809153, 30.392321688247733, 30.902004238550436, 31.30378651648329, 31.709959216054795, 32.077323875833606, 32.40685953697653, 32.79638714385636, 33.25303632974883, 33.792594287688694, 34.415528724741584, 35.24484449408251, 36.32886458292768, 37.91212469752411, 40.55665708628455],
            'psnr': []
        }
    }
def swin_lic_dict():
    return {
        "SwinLIC": {
            "qualities": [
                "SWIN-B-IC_0.16.0",
                "SWIN-B-IC_0.16.0.1",
                "SWIN-B-IC_0.16.0.2",
                "SWIN-B-IC_0.16.0.3",
                "SWIN-B-IC_0.16.0.4",
                "SWIN-B-IC_0.16.0.5",
                "SWIN-B-IC_0.16.0.6",
                "SWIN-B-IC_0.16.0.7"
            ],
            "bpp": [],
            "psnr": [],
        }
    }

def eval_jpeg(dataset, quality):
    avg_bpp = 0
    avg_psnr = 0
    for ref_img, _ in dataset:
        with BytesIO() as img_bytes:
            ref_img.save(fp=img_bytes, format='JPEG', quality=quality, subsampling=2)
            raw_bytes = img_bytes.getvalue()
            avg_bpp += (len(raw_bytes) * 8) / (ref_img.size[0] * ref_img.size[1])
            comp_img = Image.open(BytesIO(raw_bytes))
            ref_data = np.array(ref_img, dtype=np.float64)
            comp_data = np.array(comp_img, dtype=np.float64)
            avg_psnr += sk_psnr(ref_data, comp_data, data_range=255)
    avg_bpp /= len(dataset)
    avg_psnr /= len(dataset)
    return avg_bpp, avg_psnr

def eval_swin_lic(dataloader, quality):
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
        for i, (x_test_in, x_test) in enumerate(dataloader):
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
    return avg_bpp.to("cpu").numpy(), avg_psnr.to("cpu").numpy()

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

codecs = jpeg_dict() | swin_lic_dict()

pil_test_dataset = KodakDataset(None, None, None, None)
swin_test_dataset = KodakDataset(RGBCompression(noresize=True, normalize=True),None, None, None)

swin_test_loader = torch.utils.data.DataLoader(swin_test_dataset, batch_size=1, shuffle=False)

for codec, codec_dict in codecs.items():
    for i, qual in enumerate(codec_dict["qualities"]):
        if codec == "JPEG":
            avg_bpp, avg_psnr = eval_jpeg(pil_test_dataset, qual)
        elif codec == "SwinLIC":
            avg_bpp, avg_psnr = eval_swin_lic(swin_test_loader, qual)
        codec_dict["bpp"].append(avg_bpp)
        codec_dict["psnr"].append(avg_psnr)

pass

for codec, codec_dict in codecs.items():
    plt.plot(codec_dict["bpp"], codec_dict["psnr"], label=codec, marker='o')

plt.legend()
plt.xlim(0, 1.2)
plt.ylim(20, 36)
plt.show()

