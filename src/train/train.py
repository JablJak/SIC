import argparse
import datetime
import os
import random

import numpy as np
import torch
from torch.optim import AdamW
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from contextlib import redirect_stdout

from src.data.transforms import YCBCR_IMAGENET_MEAN, YCBCR_IMAGENET_STD, \
    RGB_IMAGENET_MEAN, RGB_IMAGENET_STD, ycbcr_to_rgb, YCbCrDecompression
from src.losses.rdloss import RDLoss
from src.train.experiment import Experiment
from src.utils import clearml_helpers
from src.utils.clearml_helpers import  start_experiment
from src.utils.const import MODEL_CHECKPOINT_PATH, MODEL_CHECKPOINT_FILE, EXPERIMENTS_CONFIG_PATH, \
    MODEL_OUTPUT_PATH, ARTIFACTS_PATH
from src.utils.initializers import read_config, dataloader_from_config
from src.utils.postprocess import denormalize
from src.viz.plotter import plot_reconstructions


def _train(model, train_dataloader, val_dataloader,
           criterion, optimizer, aux_optimizer, num_epochs, device, scheduler, log_file, logger=None, ycbcr=True):
    # Train

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        epoch_bpp = 0
        epoch_lr = optimizer.param_groups[0]['lr']

        for x, _ in train_dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            output = model(x)
            x_hat, y_likelihoods = output['x_hat'], output['likelihoods']['y']
            if ycbcr:
                x = denormalize(x, YCBCR_IMAGENET_MEAN, YCBCR_IMAGENET_STD)
                x_hat = denormalize(x_hat, YCBCR_IMAGENET_MEAN, YCBCR_IMAGENET_STD)
                with redirect_stdout(log_file):
                    x, x_hat = ycbcr_to_rgb(x), ycbcr_to_rgb(x_hat, recon=True)
            else:
                x = denormalize(x, RGB_IMAGENET_MEAN, RGB_IMAGENET_STD)
                x_hat = denormalize(x_hat, RGB_IMAGENET_MEAN, RGB_IMAGENET_STD)

            if isinstance(criterion, RDLoss):
                loss, bpp = criterion(x_hat, x, y_likelihoods)
                epoch_bpp += bpp
            else:
                loss = criterion(x_hat, x)

            psnr_metric = PeakSignalNoiseRatio()
            psnr_metric.to(device)
            psnr_metric.update(x_hat, x)
            psnr = psnr_metric.compute()
            epoch_psnr += psnr

            ssim_metric = StructuralSimilarityIndexMeasure()
            ssim_metric.to(device)
            ssim_metric.update(x_hat, x)
            ssim = ssim_metric.compute()
            epoch_ssim += ssim

            loss.backward()
            optimizer.step()

            aux_loss = model.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_dataloader)
        avg_psnr = epoch_psnr / len(train_dataloader)
        avg_ssim = epoch_ssim / len(train_dataloader)
        avg_bpp = epoch_bpp / len(train_dataloader)

        print(f"[TRAIN] Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.5f}, PSNR: {avg_psnr:.4f},"
              f" SSIM: {avg_ssim:.4f}, bpp: {avg_bpp:.4f} lr: {epoch_lr}")

        def log_train():
            try:
                if logger is not None:
                    logger.report_scalar(title="Loss", series="train", value=avg_loss, iteration=epoch)
                    logger.report_scalar(title="PSNR", series="train", value=avg_psnr, iteration=epoch)
                    logger.report_scalar(title="SSIM", series="train", value=avg_ssim, iteration=epoch)
                    logger.report_scalar(title="LR", series="train", value=epoch_lr, iteration=epoch)
                    if avg_bpp != 0:
                        logger.report_scalar(title="bpp", series="train", value=avg_bpp, iteration=epoch)
            except Exception as e:
                print(f"[Warning] Logging to ClearML failed: {e}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"{MODEL_CHECKPOINT_PATH}/{MODEL_CHECKPOINT_FILE}")

        # Eval
        model.eval()
        eval_loss = 0
        eval_psnr = 0
        eval_ssim = 0
        eval_bpp = 0

        with torch.no_grad():
            for x_val, _ in val_dataloader:
                x_val = x_val.to(device)
                output = model(x_val)
                x_hat_val, y_likelihoods_val = output['x_hat'], output['likelihoods']['y']
                if ycbcr:
                    x_val = denormalize(x_val, YCBCR_IMAGENET_MEAN, YCBCR_IMAGENET_STD)
                    x_hat_val = denormalize(x_hat_val, YCBCR_IMAGENET_MEAN, YCBCR_IMAGENET_STD)
                    with redirect_stdout(log_file):
                        x_val, x_hat_val = ycbcr_to_rgb(x_val), ycbcr_to_rgb(x_hat_val, recon=True)
                else:
                    x_val = denormalize(x_val, RGB_IMAGENET_MEAN, RGB_IMAGENET_STD)
                    x_hat_val = denormalize(x_hat_val, RGB_IMAGENET_MEAN, RGB_IMAGENET_STD)
                
                if isinstance(criterion, RDLoss):
                    loss_val, bpp_val = criterion(x_hat_val, x_val, y_likelihoods_val)
                    eval_bpp += bpp_val
                else:
                    loss_val = criterion(x_hat_val, x_val)
                eval_loss += loss_val.item()

                psnr_metric_val = PeakSignalNoiseRatio().to(device)
                psnr_metric_val.update(x_hat_val, x_val)
                eval_psnr += psnr_metric_val.compute()

                ssim_metric_val = StructuralSimilarityIndexMeasure().to(device)
                ssim_metric_val.update(x_hat_val, x_val)
                eval_ssim += ssim_metric_val.compute()

        avg_eval_loss = eval_loss / len(val_dataloader)
        avg_eval_psnr = eval_psnr / len(val_dataloader)
        avg_eval_ssim = eval_ssim / len(val_dataloader)
        avg_eval_bpp = eval_bpp / len(val_dataloader)

        print(f"[VAL] Epoch {epoch + 1}/{num_epochs}, "
              f"Loss: {avg_eval_loss:.4f}, PSNR: {avg_eval_psnr:.4f}, SSIM: {avg_eval_ssim:.4f}, "
              f"bpp: {avg_eval_bpp:.4f} lr: {epoch_lr}")

        def log_val():
            try:
                if logger is not None:
                    logger.report_scalar(title="PSNR", series="eval", value=avg_eval_psnr, iteration=epoch)
                    logger.report_scalar(title="SSIM", series="eval", value=avg_eval_ssim, iteration=epoch)
                    logger.report_scalar(title="Loss", series="eval", value=avg_eval_loss, iteration=epoch)
                    if avg_eval_bpp != 0:
                        logger.report_scalar(title="bpp", series="eval", value=avg_eval_bpp, iteration=epoch)
            except Exception as e:
                print(f"[Warning] Logging to ClearML failed: {e}")

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

    # ===== Disable randomness =====
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Device:", device)
    # ==============================

    train_dataset, val_dataset = experiment.train_dataset, experiment.val_dataset

    train_dataloader = dataloader_from_config(train_dataset, experiment_config['dataloader'])
    val_dataloader = dataloader_from_config(val_dataset, experiment_config['dataloader'])

    model = experiment.model
    model.to(device)

    loss = experiment.loss
    loss.to(device)

    optimizer = experiment.optimizer
    scheduler = experiment.scheduler

    with open(f"logs/{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.log", "a") as log_file:
        trained_model = _train(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            criterion=loss,
            optimizer=optimizer,
            aux_optimizer=AdamW(model.latent_codec.parameters(), lr=0.0001),
            num_epochs=experiment.epochs,
            device=device,
            scheduler=scheduler,
            logger=logger,
            log_file=log_file
        )

    # TODO: TRAIN TIME
    model.update()

    trained_model.eval()

    output_model_name = experiment.output_model_name()
    output_model_file_path = os.path.join(args.model_output_path, f"{output_model_name}.pth")

    reconstructions_path = os.path.join(ARTIFACTS_PATH, f"{output_model_name}_post_train.png")
    if not args.offline:
        task.upload_artifact(name=f"{output_model_name} post train reconstruction", artifact_object=reconstructions_path)

    torch.save(model.state_dict(), output_model_file_path)

    if not args.offline:
        clearml_helpers.save_model(task, experiment, output_model_file_path)

    with torch.no_grad():
        x_batch, _ = next(iter(val_dataloader))
        x_batch = x_batch.to(device)
        model_output = trained_model(x_batch)
        x_recon, y_likelihoods = model_output['x_hat'], model_output['likelihoods']['y']

    output_transform = YCbCrDecompression().to(x_batch.device)

    x_batch = output_transform(x_batch)
    x_recon = output_transform(x_recon)

    plot_reconstructions(x_batch, x_recon, reconstructions_path, show=False)

