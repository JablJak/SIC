import os
import typing

import torch
from src.data.coco_dataset import CocoDataset
from src.data.transforms import  YCbCrCompression, YCbCrDecompression
from src.models.swin_compression import SwinTransformerCompressionAutoencoder
from src.utils.const import ARTIFACTS_PATH
from src.utils.initializers import model_from_config
if __name__ == '__main__':
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Device:", device)

    models = [
        "SWIN-T-IC_0.4.1-test"
    ]

    transform = YCbCrCompression(noresize=True)

    dataset = CocoDataset(transform=transform)

    dataloader_iter = iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8))
    for iteration in range(1):
        x_batch, _ = next(dataloader_iter)
        x_batch = x_batch.to(device)
        for m in models:
            model: SwinTransformerCompressionAutoencoder = typing.cast(SwinTransformerCompressionAutoencoder, model_from_config(
                {
                    "module": "src.models.swin_compression.SwinTransformerCompressionAutoencoder",
                    "weights": m,
                    "args": {
                        "pretrained_encoder": False
                    }
                }))
            model.to(device)
            model.eval()

            with torch.no_grad():
                compress_output = model.compress(x_batch)
                b_repr, shape = compress_output['strings'], compress_output['shape']
                x_recon = model.decompress(b_repr, shape)['x_hat']

            output_transform = YCbCrDecompression().to(x_batch.device)

            in_img = output_transform(x_batch)[0] # TODO: Examine
            out_img = output_transform(x_recon)[0]

            os.makedirs(os.path.join(ARTIFACTS_PATH, m), exist_ok=True)
            in_img.save(os.path.join(ARTIFACTS_PATH, m, f"{iteration}_original.png"))
            out_img.save(os.path.join(ARTIFACTS_PATH, m, f"{iteration}_compressed.png"))
