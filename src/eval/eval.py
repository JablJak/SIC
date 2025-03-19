import argparse
import os
import typing

import torch
from torchvision.models import Swin_V2_T_Weights
from torchvision.transforms._presets import ImageClassification

from src.data.coco_dataset import CocoDataset
from src.data.transforms import YCbCrToRGB, RGBToYCbCr
from src.eval.interm_repr import IntermediateRepresentation
from src.models.swin_autoencoder import SwinTransformerAutoencoder
from src.utils.const import EXPERIMENTS_CONFIG_PATH, ARTIFACTS_PATH
from src.utils.initializers import model_from_config
from src.utils.postprocess import denormalize
from src.viz.plotter import plot_reconstructions

if __name__ == '__main__':
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Device:", device)

    transform = Swin_V2_T_Weights.DEFAULT.transforms()

    dataset = CocoDataset()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)

    model: SwinTransformerAutoencoder = typing.cast(SwinTransformerAutoencoder, model_from_config(
        {
            "module": "src.models.swin_autoencoder.SwinTransformerAutoencoder",
            "weights": "SWIN-T-IC_0.2.25",
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
        x_recon = model(x_batch)

    transform = YCbCrToRGB("0_1").to(x_batch.device)
    # TODO: This can't be here I guess

    x_batch = transform(denormalize(x_batch, ImageClassification(crop_size=0).mean, ImageClassification(crop_size=0).std)).cpu()
    x_recon = transform(denormalize(x_recon, ImageClassification(crop_size=0).mean, ImageClassification(crop_size=0).std)).cpu()

    reconstructions_path = os.path.join(ARTIFACTS_PATH, f"SWIN-T-IC_0.2.25_eval.png")
    plot_reconstructions(x_batch, x_recon, reconstructions_path, show=True)

    print(intermediate)