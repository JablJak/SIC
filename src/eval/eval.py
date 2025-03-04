import argparse
import typing

import torch
from torchvision.models import Swin_V2_T_Weights
from torchvision.transforms._presets import ImageClassification

from src.data.coco_dataset import CocoDataset
from src.eval.interm_repr import IntermediateRepresentation
from src.models.swin_autoencoder import SwinTransformerAutoencoder
from src.utils.const import EXPERIMENTS_CONFIG_PATH
from src.utils.initializers import model_from_config
from src.utils.postprocess import denormalize
from src.viz.plotter import plot_reconstructions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model using ClearML.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=f"{EXPERIMENTS_CONFIG_PATH}/experiment_v0.2.0.yaml",
        help="Path to the experiment configuration YAML file.",
    )

    args = parser.parse_args()
    print("CMD line args:", args)
    #
    # experiment_config = read_config(args.config)
    # experiment = Experiment(experiment_config)
    #
    # task = Task.init(project_name=experiment.project, task_name=experiment.task)
    # task.connect(experiment_config)
    # task.set_comment(experiment.comment)
    # logger = task.get_logger()

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Device:", device)

    transform = Swin_V2_T_Weights.DEFAULT.transforms()

    dataset = CocoDataset(transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)

    model: SwinTransformerAutoencoder = typing.cast(SwinTransformerAutoencoder, model_from_config(
        {
            "module": "src.models.swin_autoencoder.SwinTransformerAutoencoder",
            "weights": "SWIN-T-IC_0.2.1",
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

    x_batch = denormalize(x_batch, ImageClassification(crop_size=0).mean, ImageClassification(crop_size=0).std).cpu()
    x_recon = denormalize(x_recon, ImageClassification(crop_size=0).mean, ImageClassification(crop_size=0).std).cpu()

    plot_reconstructions(x_batch, x_recon)

    print(intermediate)