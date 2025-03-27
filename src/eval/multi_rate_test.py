import os
import typing

import PIL
import torch
from PIL.Image import fromarray
from torchvision import transforms
from torchvision.transforms import Compose

from src.data.coco_dataset import CocoDataset
from src.data.transforms import YCbCrToRGB, YCbCrCompression
from src.eval.interm_repr import IntermediateRepresentation
from src.models.swin_autoencoder import SwinTransformerAutoencoder
from src.utils.const import ARTIFACTS_PATH
from src.utils.initializers import model_from_config
from src.utils.postprocess import denormalize
from src.viz.plotter import plot_reconstructions

if __name__ == '__main__':
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Device:", device)

    models = [
        "SWIN-T-IC_0.3.5",
        "SWIN-T-IC_0.3.6",
        "SWIN-T-IC_0.3.7",
        "SWIN-T-IC_0.3.14",
        "SWIN-T-IC_0.3.8",
        "SWIN-T-IC_0.3.9",
        "SWIN-T-IC_0.3.10",
        "SWIN-T-IC_0.3.11",
        "SWIN-T-IC_0.3.13"
    ]

    transform = YCbCrCompression(noresize=True)

    dataset = CocoDataset(transform=transform)

    dataloader_iter = iter(torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8))
    for iteration in range(10):
        x_batch, _ = next(dataloader_iter)
        for m in models:
            model: SwinTransformerAutoencoder = typing.cast(SwinTransformerAutoencoder, model_from_config(
                {
                    "module": "src.models.swin_autoencoder.SwinTransformerAutoencoder",
                    "weights": m,
                    "args": {
                        "pretrained_encoder": False
                    }
                }))
            model.to(device)
            model.eval()

            intermediate = IntermediateRepresentation()
            model.encoder.register_forward_hook(intermediate.hook_fn)

            with torch.no_grad():
                x_batch = x_batch.to(device)
                x_recon, y_likelihoods = model(x_batch)

            input_transform = transform
            output_transform = transforms.Compose([
                YCbCrToRGB("0_1").to(device),
                transforms.ToPILImage()
            ])
            # TODO: This can't be here I guess

            in_img = output_transform(denormalize(x_batch, input_transform.mean, input_transform.std).squeeze())
            out_img = output_transform(denormalize(x_recon, input_transform.mean, input_transform.std).squeeze())

            # for i, in_img in enumerate(x_batch):
            os.makedirs(os.path.join(ARTIFACTS_PATH, m), exist_ok=True)
            in_img.save(os.path.join(ARTIFACTS_PATH, m, f"{iteration}_original.png"))
            # for i, in_img in enumerate(x_recon):
            out_img.save(os.path.join(ARTIFACTS_PATH, m, f"{iteration}_compressed.png"))



    # plot_reconstructions(x_batch, x_recon, reconstructions_path, show=True)

    # print(intermediate)