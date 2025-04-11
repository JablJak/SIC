import argparse
import datetime
import os
import random

import numpy as np
import torch
from torch import autocast, GradScaler
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from torchvision.transforms.v2.functional import to_pil_image

from src.data.transforms import YCBCR_IMAGENET_MEAN, YCBCR_IMAGENET_STD, \
    RGB_IMAGENET_MEAN, RGB_IMAGENET_STD, YCbCrDecompression, RGBDecompression
from src.losses.rdloss import RDLoss
from src.train.experiment import Experiment
from src.utils import clearml_helpers
from src.utils.checkpoint_helpers import save_training_state_with_clearml, load_training_state_with_clearml_from_file
from src.utils.clearml_helpers import  start_experiment
from src.utils.const import MODEL_CHECKPOINT_PATH, MODEL_CHECKPOINT_FILE, EXPERIMENTS_CONFIG_PATH, \
    MODEL_OUTPUT_PATH, ARTIFACTS_PATH
from src.utils.initializers import read_config, dataloader_from_config
from src.utils.postprocess import denormalize
from src.viz.plotter import plot_reconstructions


def _train(model, train_dataloader, val_dataloader, scaler, aux_optimizer_delay,
           criterion, optimizer, aux_optimizer, aux_scheduler, num_epochs, device, scheduler, logger=None, ycbcr=False,
           task=None, start_epoch=1):
    # Train

    optimize_bpp = False

    model.train()
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        epoch_bpp = 0
        epoch_lr = optimizer.param_groups[0]['lr']
        psnr_metric = PeakSignalNoiseRatio().to(device)
        ssim_metric = StructuralSimilarityIndexMeasure().to(device)

        if epoch > aux_optimizer_delay:
            optimize_bpp = True

        for x, _ in train_dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            aux_optimizer.zero_grad()
            with autocast(device_type="cuda"):
                output = model(x)
                x_hat, y_likelihoods = output['x_hat'], output['likelihoods']['y']
                if ycbcr:
                    x = denormalize(x.detach(), YCBCR_IMAGENET_MEAN, YCBCR_IMAGENET_STD)
                else:
                    x = denormalize(x.detach(), RGB_IMAGENET_MEAN, RGB_IMAGENET_STD)
                if isinstance(criterion, RDLoss):
                    loss, bpp = criterion(x_hat, x, y_likelihoods, optimize_bpp)
                    epoch_bpp += bpp.item()
                else:
                    loss = criterion(x_hat, x)

                aux_loss = model.aux_loss()

            scaler.scale(loss).backward()
            max_norm_value = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm_value)
            scaler.step(optimizer)

            if epoch > aux_optimizer_delay:
                aux_optimizer.zero_grad()
                scaler.scale(aux_loss).backward()
                scaler.step(aux_optimizer)

            scaler.update()

            epoch_loss += loss.item()

            psnr_metric.update(x_hat, x)
            psnr = psnr_metric.compute()
            epoch_psnr += psnr.item()

            ssim_metric.update(x_hat, x)
            ssim = ssim_metric.compute()
            epoch_ssim += ssim.item()


        avg_loss = epoch_loss / len(train_dataloader)
        avg_psnr = epoch_psnr / len(train_dataloader)
        avg_ssim = epoch_ssim / len(train_dataloader)
        avg_bpp = epoch_bpp / len(train_dataloader)

        print(f"[TRAIN] Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.5f}, PSNR: {avg_psnr:.4f},"
              f" SSIM: {avg_ssim:.4f}, bpp: {avg_bpp:.4f} lr: {epoch_lr}")

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

        if epoch % 10 == 0 and epoch > 0:
            save_training_state_with_clearml(
                task=task,
                model=model,
                optimizer=optimizer,
                aux_optimizer=aux_optimizer,
                scheduler=scheduler,
                aux_scheduler=aux_scheduler,
                scaler=scaler,
                current_epoch=epoch,
                save_path=f"{MODEL_CHECKPOINT_PATH}/checkpoint_{epoch}.pth"
            )

        # Eval
        model.eval()
        eval_loss = 0
        eval_psnr = 0
        eval_ssim = 0
        eval_bpp = 0

        with torch.no_grad():
            for x_val, _ in val_dataloader:
                x_val = x_val.to(device).detach()
                with autocast(device_type="cuda"):
                    output = model(x_val)
                    x_hat_val, y_likelihoods_val = output['x_hat'], output['likelihoods']['y']
                    x_hat_val = x_hat_val.detach()
                    if ycbcr:
                        x_val = denormalize(x_val, YCBCR_IMAGENET_MEAN, YCBCR_IMAGENET_STD)
                    else:
                        x_val = denormalize(x_val, RGB_IMAGENET_MEAN, RGB_IMAGENET_STD)
                    if isinstance(criterion, RDLoss):
                        loss_val, bpp_val = criterion(x_hat_val, x_val, y_likelihoods_val)
                        eval_bpp += bpp_val.item()
                    else:
                        loss_val = criterion(x_hat_val, x_val)
                eval_loss += loss_val.item()

                psnr_metric_val = PeakSignalNoiseRatio().to(device)
                psnr_metric_val.update(x_hat_val, x_val)
                eval_psnr += psnr_metric_val.compute().item()

                ssim_metric_val = StructuralSimilarityIndexMeasure().to(device)
                ssim_metric_val.update(x_hat_val, x_val)
                eval_ssim += ssim_metric_val.compute().item()

        avg_eval_loss = eval_loss / len(val_dataloader)
        avg_eval_psnr = eval_psnr / len(val_dataloader)
        avg_eval_ssim = eval_ssim / len(val_dataloader)
        avg_eval_bpp = eval_bpp / len(val_dataloader)

        print(f"[VAL] Epoch {epoch + 1}/{num_epochs}, "
              f"Loss: {avg_eval_loss:.4f}, PSNR: {avg_eval_psnr:.4f}, SSIM: {avg_eval_ssim:.4f}, "
              f"bpp: {avg_eval_bpp:.4f} lr: {epoch_lr}")

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
        if aux_scheduler is not None and epoch > aux_optimizer_delay:
            aux_scheduler.step()

        if epoch > aux_optimizer_delay and epoch % 10 == 0:
            model.update()

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

    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to the training checkpoint to resume from (includes ClearML task id)"
    )

    args = parser.parse_args()
    print("CMD line args:", args)

    experiment_config = read_config(args.config)
    experiment = Experiment(experiment_config)

    if not args.offline and not args.resume_checkpoint:
        task = start_experiment(experiment)
    else:
        task = None

    # ===== Disable randomness =====
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ==============================

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Device:", device)

    train_dataset, val_dataset = experiment.train_dataset, experiment.val_dataset

    train_dataloader = dataloader_from_config(train_dataset, experiment_config['dataloader'])
    val_dataloader = dataloader_from_config(val_dataset, experiment_config['dataloader'])

    model = experiment.model
    model.to(device)

    ycbcr = experiment.config["dataset"]["transform"][0]["module"] == "src.data.transforms.YCbCrCompression"

    if not ycbcr:
        for param in model.g_a.parameters():
            param.requires_grad = False
        for param_name, param_weights in model.g_a.named_parameters("6."):
             param_weights.requires_grad = True
        for param_name, param_weights in model.g_a.named_parameters("7."):
             param_weights.requires_grad = True

    loss = experiment.loss
    loss.to(device)

    optimizer = experiment.optimizer
    scheduler = experiment.scheduler
    ycbcr = experiment.config["dataset"]["transform"][0]["module"] == "src.data.transforms.YCbCrCompression"

    aux_optimizer = experiment.aux_optimizer
    aux_scheduler = experiment.aux_scheduler

    scaler = GradScaler()

    start_epoch = 1
    if args.resume_checkpoint:
        print(f"[INFO] Resuming training from checkpoint: {args.resume_checkpoint}")
        task, start_epoch = load_training_state_with_clearml_from_file(
            args.resume_checkpoint,
            model,
            optimizer,
            aux_optimizer,
            scheduler,
            aux_scheduler,
            scaler,
            device
        )

    logger = task.get_logger() if task is not None else None

    with open(f"logs/{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.log", "a") as log_file:
        trained_model = _train(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            criterion=loss,
            optimizer=optimizer,
            aux_optimizer=aux_optimizer,
            aux_scheduler=aux_scheduler,
            num_epochs=experiment.epochs,
            device=device,
            scheduler=scheduler,
            logger=logger,
            task=task,
            scaler=scaler,
            aux_optimizer_delay=experiment.aux_optimizer_delay,
            start_epoch=start_epoch,
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

    output_transform = (YCbCrDecompression(denorm=True) if ycbcr else RGBDecompression(True)).to(x_batch.device)

    x_batch = output_transform(x_batch)
    x_recon = to_pil_image(x_recon)

    plot_reconstructions(x_batch, x_recon, reconstructions_path, show=False)
