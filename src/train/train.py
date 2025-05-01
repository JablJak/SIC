import argparse
import datetime
import os
import random
import sys

import numpy as np
import torch
from torch import autocast, GradScaler
from torchmetrics.functional.image import peak_signal_noise_ratio
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from torchvision.transforms.v2.functional import to_pil_image

from src.data.transforms import YCBCR_IMAGENET_MEAN, YCBCR_IMAGENET_STD, \
    RGB_IMAGENET_MEAN, RGB_IMAGENET_STD, YCbCrDecompression, RGBDecompression
from src.losses.rdloss import RDLoss
from src.models.gdn_swin_transformer import LinearScheduler, GradualIntroductionLayer
from src.train.experiment import Experiment
from src.utils import clearml_helpers, initializers
from src.utils.checkpoint_helpers import save_training_state_with_clearml, load_training_state_with_clearml_from_file
from src.utils.clearml_helpers import  start_experiment
from src.utils.const import MODEL_CHECKPOINT_PATH, EXPERIMENTS_CONFIG_PATH, \
    MODEL_OUTPUT_PATH, ARTIFACTS_PATH
from src.utils.initializers import read_config, dataloader_from_config
from src.utils.postprocess import denormalize
from src.viz.plotter import plot_reconstructions


def _scaled_lambda(current_iter, start_iter, num_iters, start_lambda, end_lambda,
                   current_lambda, mode='log'):
    if mode == 'log':
        if current_iter < start_iter:
            return current_lambda
        if current_iter > start_iter + num_iters:
            return end_lambda
        else:
            lambda_exp = np.exp(np.log(end_lambda / start_lambda) / num_iters)
            return start_lambda * lambda_exp ** (current_iter - start_iter)


def _train(model, train_dataloader, val_dataloader, test_dataloader, scaler, aux_optimizer_delay, target_lambda,
           criterion, optimizer, aux_optimizer, aux_scheduler, num_epochs, device, scheduler, logger=None, ycbcr=False,
           task=None, start_epoch=1, log_file=None):
    # Train
    alpha_scheduler = LinearScheduler(total_steps=100, initial_value=1.0, final_value=1.0)
    optimize_bpp = False
    start_lambda = 0.1
    lambda_scale_iters = 30
    max_norm_value = 2
    scale_start_epoch = aux_optimizer_delay

    model.train()
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        epoch_bpp = 0
        epoch_lr = optimizer.param_groups[0]['lr']
        psnr_metric = PeakSignalNoiseRatio().to(device)
        ssim_metric = StructuralSimilarityIndexMeasure().to(device)
        alpha_set = { module.alpha for module in model.modules()
            if isinstance(module, GradualIntroductionLayer) }
        assert not len(alpha_set) == 0, "No GradualIntroductionLayer found in model"
        assert len(alpha_set) == 1, "Alpha is not equal for all GradualIntroductionLayer"

        # criterion.l = _scaled_lambda(epoch, start_iter=scale_start_epoch, num_iters=lambda_scale_iters,
        #                              start_lambda=start_lambda, end_lambda=target_lambda, current_lambda=criterion.l)

        if epoch > aux_optimizer_delay and aux_optimizer is not None:
            optimize_bpp = True

        for x_in, x in train_dataloader:
            x_in, x = x_in.to(device), x.to(device)
            optimizer.zero_grad()
            if aux_optimizer is not None:
                aux_optimizer.zero_grad()
            with autocast(device_type="cuda"):
                output = model(x_in)
                try:
                    x_hat, likelihoods = output['x_hat'], output['likelihoods']
                except TypeError:
                    x_hat, likelihoods = output['x_hat'], None
                if isinstance(criterion, RDLoss):
                    loss, bpp = criterion(x_hat, x, likelihoods, optimize_bpp)
                    epoch_bpp += bpp.item()
                else:
                    loss = criterion(x_hat, x)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            original_stdout = sys.stdout
            sys.stdout = log_file
            print("--- Normy Gradientów (L2 Norm) ---")
            total_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    print(f"Warstwa: {name}, Norma gradientu: {param_norm.item():.4f}")
                else:
                    print(f"Warstwa: {name}, Brak gradientu")
            total_norm = total_norm ** 0.5
            print(f"--- Całkowita norma gradientów: {total_norm:.4f} ---")
            sys.stdout = original_stdout

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm_value)
            scaler.step(optimizer)

            if epoch > aux_optimizer_delay and aux_optimizer is not None:
                with autocast(device_type="cuda"):
                    aux_loss = model.aux_loss()
                scaler.scale(aux_loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(aux=True), max_norm=max_norm_value)
                scaler.step(aux_optimizer)

            scaler.update()

            epoch_loss += loss.item()

            psnr_metric.update(x_hat.detach(), x.detach())
            psnr = psnr_metric.compute()
            epoch_psnr += psnr.item()

            ssim_metric.update(x_hat.detach(), x.detach())
            ssim = ssim_metric.compute()
            epoch_ssim += ssim.item()


        avg_loss = epoch_loss / len(train_dataloader)
        avg_psnr = epoch_psnr / len(train_dataloader)
        avg_ssim = epoch_ssim / len(train_dataloader)
        avg_bpp = epoch_bpp / len(train_dataloader)

        message = f"{datetime.datetime.now().strftime("%H:%M:%S")} [TRAIN] Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.5f}, PSNR: {avg_psnr:.4f}," \
              f" SSIM: {avg_ssim:.4f}, bpp: {avg_bpp:.4f}, lr: {epoch_lr}, alpha: {alpha_set.pop() if len(alpha_set) == 1 else 'N/A'}"
        print(message)

        try:
            if logger is not None:
                logger.report_text(message)
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
                aux_optimizer=None,
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
            for x_val_in, x_val in val_dataloader:
                x_val_in, x_val = x_val_in.to(device).detach(), x_val.to(device).detach()
                with autocast(device_type="cuda"):
                    output = model(x_val_in)
                    try:
                        x_hat_val, likelihoods_val = output['x_hat'], output['likelihoods']
                    except TypeError:
                        x_hat_val, likelihoods_val = output['x_hat'], None
                    x_hat_val = x_hat_val.detach()
                    if isinstance(criterion, RDLoss):
                        loss_val, bpp_val = criterion(x_hat_val, x_val, likelihoods_val)
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

        message = f"{datetime.datetime.now().strftime("%H:%M:%S")} [VAL] Epoch {epoch}/{num_epochs}, " \
              f"Loss: {avg_eval_loss:.4f}, PSNR: {avg_eval_psnr:.4f}, SSIM: {avg_eval_ssim:.4f}, " \
              f"bpp: {avg_eval_bpp:.4f} lr: {epoch_lr:.5f}"
        print(message)

        try:
            if logger is not None:
                logger.report_text(message)
                logger.report_scalar(title="PSNR", series="eval", value=avg_eval_psnr, iteration=epoch)
                logger.report_scalar(title="SSIM", series="eval", value=avg_eval_ssim, iteration=epoch)
                logger.report_scalar(title="Loss", series="eval", value=avg_eval_loss, iteration=epoch)
                if avg_eval_bpp != 0:
                    logger.report_scalar(title="bpp", series="eval", value=avg_eval_bpp, iteration=epoch)
        except Exception as e:
            print(f"[Warning] Logging to ClearML failed: {e}")

        if scheduler is not None:
            scheduler.step()

        if aux_scheduler is not None and epoch > aux_optimizer_delay and aux_optimizer is not None:
            aux_scheduler.step()

        alpha_scheduler.step()
        for module in model.modules():
            if isinstance(module, GradualIntroductionLayer):
                module.set_alpha(alpha_scheduler.get_value())

        if epoch > aux_optimizer_delay and aux_optimizer is not None:
            model.update()
            avg_test_psnr = 0
            avg_test_bpp = 0
            with torch.no_grad():
                for x_test_in, x_test in test_dataloader:
                    x_test_in, x_test = x_test_in.to(device).detach(), x_test.to(device).detach()
                    with autocast(device_type="cuda", enabled=False):
                        compress_output = model.compress(x_test_in)
                        b_repr, shape = compress_output['strings'], compress_output['shape']
                        x_hat_test = model.decompress(b_repr, shape)['x_hat']

                        psnr = peak_signal_noise_ratio(x_test, x_hat_test)
                        avg_test_psnr += psnr

                        batch_size = len(b_repr[0])
                        bits = sum(len(stream) * 8 for group in b_repr for stream in group)
                        _, _, H, W = x_hat_test.shape
                        bpp = bits / (batch_size * H * W)
                        avg_test_bpp += bpp

            avg_test_bpp /= len(test_dataloader)
            avg_test_psnr /= len(test_dataloader)

            message = f"{datetime.datetime.now().strftime("%H:%M:%S")} [TEST] Epoch {epoch}/{num_epochs}, PSNR: {avg_test_psnr:.4f}, bpp: {avg_test_bpp:.4f}"
            print(message)
            if logger is not None:
                logger.report_text(message)
                logger.report_scalar(title="PSNR", series="test", value=avg_test_psnr, iteration=epoch)
                logger.report_scalar(title="bpp", series="test", value=avg_test_bpp, iteration=epoch)
        model.train()
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

    train_dataset, val_dataset, test_dataset = experiment.train_dataset, experiment.val_dataset, experiment.test_dataset

    train_dataloader = dataloader_from_config(train_dataset, experiment_config['dataloader'])
    val_dataloader = dataloader_from_config(val_dataset, experiment_config['dataloader'])
    test_dataloader = dataloader_from_config(test_dataset, experiment_config['dataloader'])

    model = experiment.model
    model.to(device)

    ycbcr = experiment.config["train_dataset"]["transform"][0]["module"] == "src.data.transforms.YCbCrCompression"

    # for module in model.modules():
    # for name, param in model.named_parameters():
    #     if name.startswith("encoder"):
    #         param.requires_grad = False

    loss = experiment.loss
    loss.to(device)

    optimizer = experiment.optimizer
    scheduler = experiment.scheduler

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
            None,
            None,
            aux_scheduler,
            scaler,
            device
        )

    logger = task.get_logger() if task is not None else None

    # for i, param_group in enumerate(optimizer.param_groups):
    #     param_group['initial_lr'] = 2.5e-5
    # for i, param_group in enumerate(optimizer.param_groups):
    #     param_group['lr'] = 2.5e-5
    #
    # scheduler = initializers.scheduler_from_config(optimizer, experiment_config['scheduler'])

    with open(f"logs/{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}.log", "a") as log_file:
        trained_model = _train(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
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
            target_lambda=loss.l if isinstance(loss, RDLoss) else 0,
            log_file=log_file
        )

    # TODO: TRAIN TIME
    if not model.no_compress:
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
