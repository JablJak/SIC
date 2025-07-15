import os
import typing

import PIL.Image
import torch
from compressai.latent_codecs import EntropyBottleneckLatentCodec
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imread
from torch.nn.parameter import Parameter
from torchvision.transforms.v2.functional import to_pil_image

from src.data.coco_dataset import CocoDataset
from src.data.imagenet_dataset import ImageNetDataset
from src.data.kodak_dataset import KodakDataset
from src.data.transforms import YCbCrCompression, YCbCrDecompression, RGBDecompression, RGBCompression, denormalize, \
    RGB_IMAGENET_MEAN, RGB_IMAGENET_STD
from src.models.swin_compression import SwinTransformerCompressionAutoencoder
from src.utils.checkpoint_helpers import load_training_state_with_clearml_from_file
from src.utils.const import ARTIFACTS_PATH
from src.utils.initializers import model_from_config

if __name__ == '__main__':
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Device:", device)

    models = [
        # ("SWIN-S-IC_0.20.3", "gdn_swin_v2_s"),
        # ("SWIN-S-IC-BASE_0.58.0", "gdn_swin_v2_s"),
        ("SWIN-S-IC_0.70.3", "gdn_swin_v2_s"),
        # ("SWIN-S-IC_0.3.1_100", "swin_v2_s")
        # "SWIN-T-IC_0.12.0-150of400",
        # "SWIN-T-IC_0.9.4-210of400"
    ]

    transform = RGBCompression(crop_size=512, resize_size=512, mean=(0.470, 0.447, 0.408), std=(0.270, 0.266, 0.281))
    target_transform = RGBCompression(crop_size=512, resize_size=512, normalize=False, mean=(0.470, 0.447, 0.408), std=(
        0.270, 0.266, 0.281))
    # transform = RGBCompression(crop_size=256, resize_size=256)

    dataset = CocoDataset(transform=transform, target_transform=target_transform, variant='val')


    pic_num = 20

    for m in models:
        psnr_sum = 0
        bpp_sum = 0
        ssim_sum = 0
        dataloader_iter = iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=False))
        model: SwinTransformerCompressionAutoencoder = typing.cast(SwinTransformerCompressionAutoencoder, model_from_config(
            {
                "module": "src.models.swin_compression.SwinTransformerCompressionAutoencoder",
                "args": {
                    "pretrained_encoder": False,
                    "encoder_type": "gdn_swin_v2_s",
                    "encoder_embed_dim": 96,
                    "encoder_dims": [96, 192, 384, 768],
                    "encoder_depths": [2, 2, 18, 2],
                    "encoder_num_heads": [3, 6, 12, 24],
                    "encoder_window_size": [8, 8],
                    "encoder_sd_factor": 0.1,
                    "encoder_mlp_ratio": 4,
                    "decoder_depths": [2, 18, 2, 2],
                    "decoder_dims": [768, 384, 192, 96, 48],
                    "decoder_num_heads": [24, 12, 6, 3],
                    "decoder_window_size": [[8, 8], [8, 8], [8, 8], [8, 8]],
                    "decoder_mlp_ratio": [4, 4, 4, 4],
                    "decoder_sd_factor": 0.1,
                    "no_compress": False
                }
            }))
        state = torch.load("/run/media/jakub/Dane/Studia/INZ/checkpoint/last_checkpoint.pth", map_location='cpu')
        model.load_state_dict(state['model'], strict=False)

        model.to(device)
        params: typing.Iterator[Parameter]  = model.g_s.reconstruction.activation.parameters()
        for param in params:
            print(param.data)
        model.update()
        torch.save(model.state_dict(), f"../../models/{m[0]}.pth")
        print(sum(param.numel() for param in model.parameters() if param.requires_grad))
        model.eval()
        for iteration in range(pic_num):
            x_batch, _ = next(dataloader_iter)
            x_batch = x_batch.to(device)

            with torch.no_grad():
                compress_output = model.compress(x_batch)
                b_repr, shape = compress_output['strings'], compress_output['shape']
                x_recon = model.decompress(b_repr, shape)['x_hat']
                # output = model(x_batch)
                # x_recon, y_likelihoods = output['x_hat'], None

            output_transform = RGBDecompression(denorm=True, mean=(0.470, 0.447, 0.408), std=(0.270, 0.266,
                                                                                              0.281)).to(x_batch.device)

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

            psnr_value = peak_signal_noise_ratio(image1, image2)
            print(f"PSNR: {psnr_value}")
            psnr_sum += psnr_value

            # ssim_value = structural_similarity(image1, image2, min_size=7)
            # print(f"ssim: {ssim_value}")
            # ssim_sum += ssim_value
            bits = sum([sum([len(b) for b in b_repr_item]) * 8 / len(b_repr_item) for b_repr_item in b_repr])
            bpp = bits / (in_img.size[0] * in_img.size[1])
            print(f"bpp: {bpp}")
            bpp_sum += bpp

        print(f"Avg PSNR: {(psnr_sum / pic_num):.4f}")
        print(f"Avg SSIM: {(ssim_sum / pic_num):.4f}")
        print(f"Avg bpp: {(bpp_sum / pic_num):.4f}")
