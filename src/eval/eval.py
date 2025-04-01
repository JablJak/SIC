import os
import typing

import torch

from src.data.coco_dataset import CocoDataset
from src.data.imagenet_dataset import ImageNetDataset
from src.data.transforms import YCbCrCompression, YCbCrDecompression, RGBDecompression, RGBCompression
from src.eval.interm_repr import IntermediateRepresentation
from src.models.swin_autoencoder import SwinTransformerAutoencoder
from src.models.swin_compression import SwinTransformerCompressionAutoencoder
from src.utils.const import ARTIFACTS_PATH
from src.utils.initializers import model_from_config
from src.viz.plotter import plot_reconstructions

if __name__ == '__main__':
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Device:", device)

    transform = RGBCompression(crop_size=256, resize_size=260)

    dataset = ImageNetDataset(transform=transform, variant="val.X")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)

    model: SwinTransformerCompressionAutoencoder = typing.cast(SwinTransformerCompressionAutoencoder, model_from_config(
        {
            "module": "src.models.swin_compression.SwinTransformerCompressionAutoencoder",
            "weights": "SWIN-T-IC_0.6.0-230of500",
            "args": {
                "pretrained_encoder": False
            }
        }))
    model.to(device)
    model.eval()

    intermediate = IntermediateRepresentation()
    model.encoder.register_forward_hook(intermediate.hook_fn)

    with torch.no_grad():
        x_batch, _ = next(iter(dataloader))
        x_batch = x_batch.to(device)
        model_output = model(x_batch)
        x_recon, y_likelihoods = model_output['x_hat'], model_output['likelihoods']['y']

    output_transform = RGBDecompression().to(x_batch.device)

    x_batch = output_transform(x_batch)
    x_recon = output_transform(x_recon)

    reconstructions_path = os.path.join(ARTIFACTS_PATH, f"SWIN-T-IC_0.6.0-230of500.png")
    plot_reconstructions(x_batch, x_recon, reconstructions_path, show=True)

    print(intermediate)