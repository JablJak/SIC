import os
import typing

import PIL.Image
import torch
from compressai.latent_codecs import EntropyBottleneckLatentCodec
from skimage.metrics import peak_signal_noise_ratio
from skimage.io import imread
from torchvision.transforms.v2.functional import to_pil_image

from src.data.coco_dataset import CocoDataset
from src.data.imagenet_dataset import ImageNetDataset
from src.data.transforms import YCbCrCompression, YCbCrDecompression, RGBDecompression, RGBCompression, denormalize, \
    RGB_IMAGENET_MEAN, RGB_IMAGENET_STD
from src.models.swin_compression import SwinTransformerCompressionAutoencoder
from src.utils.const import ARTIFACTS_PATH
from src.utils.initializers import model_from_config

if __name__ == '__main__':
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Device:", device)

    models = [
        "SWIN-S-IC_0.3.1_70",
        "SWIN-S-IC_0.3.1_100"
        # "SWIN-T-IC_0.12.0-150of400",
        # "SWIN-T-IC_0.9.4-210of400"
    ]

    transform = RGBCompression(crop_size=512, resize_size=512)
    # transform = RGBCompression(noresize=True)

    dataset = CocoDataset(transform=transform)


    pic_num = 20

    for m in models:
        psnr_sum = 0
        bpp_sum = 0
        dataloader_iter = iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8))
        model: SwinTransformerCompressionAutoencoder = typing.cast(SwinTransformerCompressionAutoencoder, model_from_config(
            {
                "module": "src.models.swin_compression.SwinTransformerCompressionAutoencoder",
                "weights": m,
                "args": {
                    "pretrained_encoder": False,
                    "encoder_type": "swin_v2_s",
                    "decoder_depths": [2, 18, 2, 2]
                }
            }))
        model.to(device)
        params = model.a_s_parameters()
        model.update()

        model.eval()
        for iteration in range(pic_num):
            x_batch, _ = next(dataloader_iter)
            x_batch = x_batch.to(device)

            with torch.no_grad():
                compress_output = model.compress(x_batch)
                b_repr, shape = compress_output['strings'], compress_output['shape']
                x_recon = model.decompress(b_repr, shape)['x_hat']

            output_transform = RGBDecompression(denorm=True).to(x_batch.device)

            in_img: PIL.Image.Image = output_transform(x_batch)[0] # TODO: Examine
            out_img: PIL.Image.Image = to_pil_image(x_recon[0])
            out_img = out_img.crop((0, 0, in_img.size[0], in_img.size[1]))

            os.makedirs(os.path.join(ARTIFACTS_PATH, m), exist_ok=True)
            im_img_path = os.path.join(ARTIFACTS_PATH, m, f"{iteration}_original.png")
            out_img_path = os.path.join(ARTIFACTS_PATH, m, f"{iteration}_compressed.png")
            in_img.save(im_img_path)
            out_img.save(out_img_path)

            x_orig_np = denormalize(x_batch, RGB_IMAGENET_MEAN, RGB_IMAGENET_STD)[0].cpu().numpy()
            x_recon_np = x_recon[0].cpu().numpy()

            psnr_value = peak_signal_noise_ratio(x_orig_np, x_recon_np)
            print(f"PSNR: {psnr_value}")
            psnr_sum += psnr_value
            bits = len(b_repr[0][0]) * 8
            bpp = bits / (in_img.size[0] * in_img.size[1])
            print(f"bpp: {bpp}")
            bpp_sum += bpp

        print(f"Avg PSNR: {(psnr_sum / pic_num):.4f}")
        print(f"Avg bpp: {(bpp_sum / pic_num):.4f}")
