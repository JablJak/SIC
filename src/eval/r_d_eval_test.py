import os
import typing

import PIL.Image
import compressai
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread
from torchvision.transforms.v2.functional import to_pil_image

from src.data.kodak_dataset import KodakDataset
from src.data.transforms import RGBDecompression, RGBCompression, denormalize, \
    RGB_IMAGENET_MEAN, RGB_IMAGENET_STD, RGB_MIXED_LIC_MEAN, RGB_MIXED_LIC_STD
from src.models.swin_compression import SwinTransformerCompressionAutoencoder
from src.utils.const import ARTIFACTS_PATH
from src.utils.initializers import model_from_config

if __name__ == '__main__':
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Device:", device)
    models = [
        ("SWIN-LIC_1.1.0", "gdn_swin_v2_b"),
    ]

    transform = RGBCompression(crop_size=[512, 512], normalize=False, mean=RGB_MIXED_LIC_MEAN, std=RGB_MIXED_LIC_STD)
    target_transform = RGBCompression(crop_size=[512, 512], noresize=True, mean=RGB_MIXED_LIC_MEAN, std=RGB_MIXED_LIC_STD)

    dataset = KodakDataset(transform=transform, target_transform=target_transform)
    pic_num = 24

    for m in models:
        psnr_sum = 0
        bpp_sum = 0
        ssim_sum = 0
        dataloader_iter = iter(torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=False))
        model: SwinTransformerCompressionAutoencoder = typing.cast(SwinTransformerCompressionAutoencoder, model_from_config(
            {
                "module": "src.models.swin_compression.SwinTransformerCompressionAutoencoder",
                # "weights": "SWIN-LIC_1.0.1",
                "args": {
                    "pretrained_encoder": False,
                    "encoder_type": "gdn_swin_v2_b",
                    "encoder_embed_dim": 128,
                    "encoder_dims": [128, 256, 512],
                    "encoder_depths": [2, 6, 18],
                    "encoder_num_heads": [4, 8, 16],
                    "encoder_window_size": [8, 8],
                    "encoder_sd_factor": 0.05,
                    "encoder_mlp_ratio": 4,
                    "encoder_dropout": 0,
                    "encoder_attention_dropout": 0,
                    "decoder_depths": [18, 6, 2],
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
            }))
        state = torch.load("/run/media/jakub/Dane/Studia/INZ/checkpoint/last_checkpoint.pth", map_location='cpu')
        model.load_state_dict(state['model'])
        # print(state['model'].keys())
        model.eval()
        model.update(force=True, update_quantiles=True)
        torch.save(model.state_dict(), f"../../models/{m[0]}.pth")
        print(model)
        model.to(device)

        print(sum(param.numel() for param in model.parameters() if param.requires_grad))
        for iteration in range(pic_num):
            x_batch, _ = next(dataloader_iter)
            x_batch = x_batch.to(device)

            with torch.no_grad():
                compress_output = model.compress(x_batch)
                b_repr, shape = compress_output['strings'], compress_output['shape']
                x_recon = model.decompress(b_repr, shape)['x_hat']

            output_transform = RGBDecompression(denorm=False, mean=RGB_MIXED_LIC_MEAN, std=RGB_MIXED_LIC_STD).to(x_batch.device)

            in_img: PIL.Image.Image = output_transform(x_batch)[0] # TODO: Examine
            out_img: PIL.Image.Image = to_pil_image(x_recon[0])
            out_img = out_img.crop((0, 0, in_img.size[0], in_img.size[1]))

            os.makedirs(os.path.join(ARTIFACTS_PATH, m[0]), exist_ok=True)
            im_img_path = os.path.join(ARTIFACTS_PATH, m[0], f"{iteration}_original.png")
            out_img_path = os.path.join(ARTIFACTS_PATH, m[0], f"{iteration}_compressed.png")
            in_img.save(im_img_path)
            out_img.save(out_img_path)

            x_orig_np = denormalize(x_batch, RGB_IMAGENET_MEAN, RGB_IMAGENET_STD)[0].cpu().numpy()
            x_recon_np = x_recon[0].cpu().numpy()

            image1 = imread(im_img_path)
            image2 = imread(out_img_path)

            psnr_value = peak_signal_noise_ratio(image1, image2, data_range=255.0)
            print(f"PSNR: {psnr_value}")
            psnr_sum += psnr_value

            # ssim_value = structural_similarity(image1, image2, min_size=7)
            # print(f"ssim: {ssim_value}")
            # ssim_sum += ssim_value
            bits = sum([sum([len(b) for b in b_repr_item]) * 8 / len(b_repr_item) for b_repr_item in b_repr])
            bpp = bits / (in_img.size[0] * in_img.size[1])
            print(f"bpp: {bpp}")
            bpp_sum += bpp
            torch.cuda.empty_cache()

        print(f"Avg PSNR: {(psnr_sum / pic_num):.4f}")
        print(f"Avg SSIM: {(ssim_sum / pic_num):.4f}")
        print(f"Avg bpp: {(bpp_sum / pic_num):.4f}")
