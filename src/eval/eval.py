import os
import typing

import torch

from src.data.coco_dataset import CocoDataset
from src.data.transforms import YCbCrCompression, YCbCrDecompression
from src.eval.interm_repr import IntermediateRepresentation
from src.models.swin_autoencoder import SwinTransformerAutoencoder
from src.utils.const import ARTIFACTS_PATH
from src.utils.initializers import model_from_config
from src.viz.plotter import plot_reconstructions

if __name__ == '__main__':
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Device:", device)

    transform = YCbCrCompression(crop_size=256, resize_size=260)

    dataset = CocoDataset(transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)

    model: SwinTransformerAutoencoder = typing.cast(SwinTransformerAutoencoder, model_from_config(
        {
            "module": "src.models.swin_autoencoder.SwinTransformerAutoencoder",
            "weights": "SWIN-T-IC_0.3.13",
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
        x_recon, y_likelihoods = model(x_batch)

    output_transform = YCbCrDecompression().to(x_batch.device)

    x_batch = output_transform(x_batch)
    x_recon = output_transform(x_recon)

    reconstructions_path = os.path.join(ARTIFACTS_PATH, f"SWIN-T-IC_0.3.4_eval.png")
    plot_reconstructions(x_batch, x_recon, reconstructions_path, show=True)

    print(intermediate)