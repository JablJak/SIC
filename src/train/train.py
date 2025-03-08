import argparse
import os

import torch
from torch.utils.data import random_split
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.models import Swin_V2_T_Weights
from torchvision.transforms._presets import ImageClassification

from src.train.experiment import Experiment
from src.utils import clearml_helpers
from src.utils.clearml_helpers import save_model, start_experiment
from src.utils.const import MODEL_CHECKPOINT_PATH, MODEL_CHECKPOINT_FILE, EXPERIMENTS_CONFIG_PATH, \
    MODEL_OUTPUT_PATH
from src.utils.initializers import read_config, dataloader_from_config
from src.utils.postprocess import denormalize
from src.viz.plotter import plot_reconstructions


def _train(model, train_dataloader, val_dataloader,
           criterion, optimizer, num_epochs, device, scheduler, logger=None):
    # Train

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0

        for x, _ in train_dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            output = model(x)

            loss = criterion(output, x)

            psnr_metric = PeakSignalNoiseRatio()
            psnr_metric.to(device)
            psnr_metric.update(output, x)
            psnr = psnr_metric.compute()
            epoch_psnr += psnr

            ssim_metric = StructuralSimilarityIndexMeasure()
            ssim_metric.to(device)
            ssim_metric.update(output, x)
            ssim = ssim_metric.compute()
            epoch_ssim += ssim

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dataloader)
        avg_psnr = epoch_psnr / len(train_dataloader)
        avg_ssim = epoch_ssim / len(train_dataloader)

        print(f"[TRAIN] Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

        if logger is not None:
            logger.report_scalar(title="Loss", series="train", value=avg_loss, iteration=epoch)
            logger.report_scalar(title="PSNR", series="train", value=avg_psnr, iteration=epoch)
            logger.report_scalar(title="SSIM", series="train", value=avg_ssim, iteration=epoch)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{MODEL_CHECKPOINT_PATH}/{MODEL_CHECKPOINT_FILE}")

        # Eval
        model.eval()
        eval_loss = 0
        eval_psnr = 0
        eval_ssim = 0

        with torch.no_grad():
            for x_val, _ in val_dataloader:
                x_val = x_val.to(device)
                output_val = model(x_val)

                loss_val = criterion(output_val, x_val)
                eval_loss += loss_val.item()

                psnr_metric_val = PeakSignalNoiseRatio().to(device)
                psnr_metric_val.update(output_val, x_val)
                eval_psnr += psnr_metric_val.compute()

                ssim_metric_val = StructuralSimilarityIndexMeasure().to(device)
                ssim_metric_val.update(output_val, x_val)
                eval_ssim += ssim_metric_val.compute()

        avg_eval_loss = eval_loss / len(val_dataloader)
        avg_eval_psnr = eval_psnr / len(val_dataloader)
        avg_eval_ssim = eval_ssim / len(val_dataloader)

        print(f"[VAL] Epoch {epoch + 1}/{num_epochs}, "
              f"Loss: {avg_eval_loss:.4f}, PSNR: {avg_eval_psnr:.4f}, SSIM: {avg_eval_ssim:.4f}")

        if logger is not None:
            logger.report_scalar(title="Loss", series="eval", value=avg_eval_loss, iteration=epoch)
            logger.report_scalar(title="PSNR", series="eval", value=avg_eval_psnr, iteration=epoch)
            logger.report_scalar(title="SSIM", series="eval", value=avg_eval_ssim, iteration=epoch)

        model.train()
        if scheduler is not None:
            scheduler.step()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model using ClearML.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=f"{EXPERIMENTS_CONFIG_PATH}/experiment_v0.2.0.yaml",
        help="Path to the experiment configuration YAML file.",
    )
    parser.add_argument(
        "--model_output_path",
        type=str,
        default=MODEL_OUTPUT_PATH,
        help="Path for the output model"
    )
    parser.add_argument(
        "--model_checkpoint_path",
        type=str,
        default=MODEL_CHECKPOINT_PATH,
        help="Path for checkpoints",
    )

    parser.add_argument(
        "--offline",
        action='store_true',
        help="Whether or not to upload experiment data to ClearML",
    )

    args = parser.parse_args()
    print("CMD line args:", args)

    experiment_config = read_config(args.config)
    experiment = Experiment(experiment_config)

    if not args.offline:
        task, logger = start_experiment(experiment)
    else:
        task, logger = None, None

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Device:", device)

    transform = Swin_V2_T_Weights.DEFAULT.transforms()

    dataset = experiment.dataset
    train_dataset, val_dataset = random_split(
        dataset, [experiment.dataset_split_ratio, 1 - experiment.dataset_split_ratio]
    )

    train_dataloader = dataloader_from_config(train_dataset, experiment_config['dataloader'])
    val_dataloader = dataloader_from_config(val_dataset, experiment_config['dataloader'])

    model = experiment.model
    model.to(device)

    loss = experiment.loss
    loss.to(device)

    optimizer = experiment.optimizer
    scheduler = experiment.scheduler

    trained_model = _train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=loss,
        optimizer=optimizer,
        num_epochs=experiment.epochs,
        device=device,
        scheduler=scheduler,
        logger=logger
    )

    # TODO: TRAIN TIME

    trained_model.eval()
    with torch.no_grad():
        x_batch, _ = next(iter(val_dataloader))
        x_batch = x_batch.to(device)
        x_recon = trained_model(x_batch)

    x_batch = denormalize(x_batch, ImageClassification(crop_size=0).mean, ImageClassification(crop_size=0).std).cpu()
    x_recon = denormalize(x_recon, ImageClassification(crop_size=0).mean, ImageClassification(crop_size=0).std).cpu()

    plot_reconstructions(x_batch, x_recon)

    output_model_name = experiment.output_model_name()
    output_model_file_path = os.path.join(args.model_output_path, f"{output_model_name}.pth")

    torch.save(model.state_dict(), output_model_file_path)

    if not args.offline:
        clearml_helpers.save_model(task, experiment, output_model_file_path)