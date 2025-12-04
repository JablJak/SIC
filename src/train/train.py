import argparse
import datetime
import gc
import os
import random
import re
import sys

import lion_pytorch
import numpy as np
import torch
from torch import autocast, GradScaler
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from torchvision.transforms.v2.functional import to_pil_image

from src.data.transforms import YCBCR_IMAGENET_MEAN, YCBCR_IMAGENET_STD, \
    RGB_IMAGENET_MEAN, RGB_IMAGENET_STD, YCbCrDecompression, RGBDecompression, RGB_COCO_MEAN, RGB_COCO_STD
from src.losses.l1_ssim import L1SSIM
from src.losses.mse_ssim import MSESSIM
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
           task=None, start_epoch=1, global_step=0, log_file=None, log_frequency=100, accumulation_steps=4):
    # Train
    alpha_scheduler = LinearScheduler(total_steps=100, initial_value=1.0, final_value=1.0)
    optimize_bpp = False
    max_norm_value = 2
    temp_checkpoint_frequency = 1000
    persist_checkpoint_frequency = 5000
    eval_frequency = 500
    test_frequency = 1000
    start_lambda = 0.1
    lambda_scale_iters = 30
    scale_start_epoch = aux_optimizer_delay
    torch.cuda.empty_cache()
    interval_loss = 0
    interval_bpp = 0
    interval_aux_loss = 0
    optimizer_stepped = False
    aux_optimizer_stepped = False
    for epoch in range(start_epoch, num_epochs):
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        accumulated_loss = 0
        accumulated_bpp = 0
        psnr_metric = PeakSignalNoiseRatio(data_range=(0.0, 1.0), reduction='elementwise_mean', dim=(2, 3)).to(device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0), reduction='elementwise_mean').to(device)
        alpha_set = { module.alpha for module in model.modules()
            if isinstance(module, GradualIntroductionLayer) }
        # assert not len(alpha_set) == 0, "No GradualIntroductionLayer found in model"
        # assert len(alpha_set) == 1, "Alpha is not equal for all GradualIntroductionLayer"

        # criterion.l = _scaled_lambda(epoch, start_iter=scale_start_epoch, num_iters=lambda_scale_iters,
        #                              start_lambda=start_lambda, end_lambda=target_lambda, current_lambda=criterion.l)
        for i, (x_in, x) in enumerate(train_dataloader):
            model.train()
            if global_step > aux_optimizer_delay and aux_optimizer is not None:
                optimize_bpp = True
            global_step += 1
            x_in, x = x_in.to(device), x.to(device)
            output = model(x_in)
            try:
                x_hat, likelihoods = output['x_hat'], output['likelihoods']
            except TypeError:
                x_hat, likelihoods = output['x_hat'], None
            if isinstance(criterion, RDLoss):
                loss, bpp = criterion(x_hat, x, likelihoods, optimize_bpp)
            else:
                loss = criterion(x_hat, x)
                bpp = torch.tensor(0.0)

            loss_scaled = loss / accumulation_steps
            loss_scaled.backward()
            
            if global_step > aux_optimizer_delay and aux_optimizer is not None:
                aux_loss = model.aux_loss()
                scaled_aux_loss = aux_loss / accumulation_steps
                scaled_aux_loss.backward()
                interval_aux_loss += aux_loss.item()

            # scaler.scale(total_loss).backward()
            interval_loss += loss.item()
            interval_bpp += bpp.item()

            psnr_metric.update(x_hat.detach(), x.detach())
            ssim_metric.update(x_hat.detach(), x.detach())

            if (i + 1) % accumulation_steps == 0:
                # scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm_value)
                # scaler.step(optimizer)
                optimizer.step()
                optimizer_stepped = True
                if global_step > aux_optimizer_delay and aux_optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(aux=True), max_norm=max_norm_value/2)
                    # scaler.step(aux_optimizer)
                    aux_optimizer.step()
                    aux_optimizer_stepped = True
                # scaler.update()
                optimizer.zero_grad()
                if aux_optimizer is not None:
                    aux_optimizer.zero_grad()
            if global_step % log_frequency == 0:
                avg_interval_loss = interval_loss / log_frequency
                avg_interval_aux_loss = interval_aux_loss / log_frequency
                avg_interval_bpp = interval_bpp / log_frequency
                avg_interval_psnr = psnr_metric.compute()
                avg_interval_ssim = ssim_metric.compute()
                lrs = str.join(", ", [f'lr{i}: {pg['lr']:4g}' for i, pg in enumerate(optimizer.param_groups)])
                message = f"{datetime.datetime.now().strftime('%H:%M:%S')} [TRAIN] Epoch {epoch}/{num_epochs}, Step: {global_step}, Loss: {avg_interval_loss:.5f}, PSNR: {avg_interval_psnr:.4f}, " \
                    f"SSIM: {avg_interval_ssim:.4f}, bpp: {avg_interval_bpp:.4f}, Aux loss: {avg_interval_aux_loss:.5f}, {lrs}, aux_lr: {aux_optimizer.param_groups[0]['lr']:4g}, " \
                    f"alpha: {alpha_set.pop() if len(alpha_set) == 1 else 'N/A'}"
                if logger is not None:
                    try:
                        logger.report_text(message)
                        logger.report_scalar(title="Loss", series="train", value=avg_interval_loss, iteration=global_step)
                        logger.report_scalar(title="Aux loss", series="train", value=avg_interval_aux_loss, iteration=global_step)
                        logger.report_scalar(title="PSNR", series="train", value=avg_interval_psnr, iteration=global_step)
                        logger.report_scalar(title="SSIM", series="train", value=avg_interval_ssim, iteration=global_step)
                        logger.report_scalar(title="LR", series="train", value=optimizer.param_groups[3]['lr'], iteration=global_step)
                        logger.report_scalar(title="Aux LR", series="train", value=aux_optimizer.param_groups[0]['lr'], iteration=global_step)
                        if avg_interval_bpp != 0:
                            logger.report_scalar(title="bpp", series="train", value=avg_interval_bpp, iteration=global_step)
                    except Exception as e:
                        print(f"[Warning] Logging to ClearML failed: {e}")
                else:
                    print(message)
                interval_bpp = 0
                interval_loss = 0
                interval_aux_loss = 0
                psnr_metric.reset()
                ssim_metric.reset()

            if global_step % temp_checkpoint_frequency == 0:
                save_training_state_with_clearml(
                    task=task,
                    model=model,
                    optimizer=optimizer,
                    aux_optimizer=aux_optimizer,
                    scheduler=scheduler,
                    aux_scheduler=aux_scheduler,
                    scaler=scaler,
                    current_epoch=epoch,
                    current_step=global_step,
                    save_path=f"{MODEL_CHECKPOINT_PATH}/last_checkpoint.pth"
                )

            if global_step % persist_checkpoint_frequency == 0:
                save_training_state_with_clearml(
                    task=task,
                    model=model,
                    optimizer=optimizer,
                    aux_optimizer=aux_optimizer,
                    scheduler=scheduler,
                    aux_scheduler=aux_scheduler,
                    scaler=scaler,
                    current_epoch=epoch,
                    current_step=global_step,
                    save_path=f"{MODEL_CHECKPOINT_PATH}/checkpoint_{global_step}.pth"
                )

            if scheduler is not None and optimizer_stepped:
                scheduler.step()

            if aux_scheduler is not None and epoch > aux_optimizer_delay and aux_optimizer is not None and aux_optimizer_stepped:
                aux_scheduler.step()

            if global_step % eval_frequency == 0:
                # Eval
                model.eval()
                psnr_metric_val = PeakSignalNoiseRatio(data_range=(0.0, 1.0), reduction='elementwise_mean', dim=(2, 3)).to(device)
                ssim_metric_val = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0), reduction='elementwise_mean').to(device)
                eval_loss = 0
                eval_psnr = 0
                eval_ssim = 0
                eval_bpp = 0
                eval_aux_loss = 0

                with torch.no_grad():
                    for x_val_in, x_val in val_dataloader:
                        x_val_in, x_val = x_val_in.to(device).detach(), x_val.to(device).detach()
                        # with autocast(device_type="cuda"):
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

                        if global_step > aux_optimizer_delay and aux_optimizer is not None:
                            aux_loss_val = model.aux_loss()
                            eval_aux_loss += aux_loss_val.item()

                        psnr_metric_val.update(x_hat_val, x_val)
                        # eval_psnr += psnr_metric_val.compute().item()

                        ssim_metric_val.update(x_hat_val, x_val)
                        # eval_ssim += ssim_metric_val.compute().item()

                avg_eval_psnr = psnr_metric_val.compute()
                avg_eval_ssim = ssim_metric_val.compute()

                avg_eval_loss = eval_loss / len(val_dataloader)
                avg_eval_bpp = eval_bpp / len(val_dataloader)
                avg_eval_aux_loss = eval_aux_loss / len(val_dataloader)

                del loss_val, output, x_hat_val, likelihoods_val, x_val_in, x_val
                gc.collect()
                torch.cuda.empty_cache()

                message = f"{datetime.datetime.now().strftime('%H:%M:%S')} [VAL] Epoch {epoch}/{num_epochs}, Step: {global_step}, " \
                      f"Loss: {avg_eval_loss:.4f}, PSNR: {avg_eval_psnr:.4f}, SSIM: {avg_eval_ssim:.4f}, " \
                      f"bpp: {avg_eval_bpp:.4f}, Aux loss: {avg_eval_aux_loss:.4f}"
                if logger is not None:
                    try:
                        logger.report_text(message)
                        logger.report_scalar(title="PSNR", series="eval", value=avg_eval_psnr, iteration=global_step)
                        logger.report_scalar(title="SSIM", series="eval", value=avg_eval_ssim, iteration=global_step)
                        logger.report_scalar(title="Loss", series="eval", value=avg_eval_loss, iteration=global_step)
                        logger.report_scalar(title="Aux loss", series="eval", value=avg_eval_aux_loss, iteration=global_step)
                        if avg_eval_bpp != 0:
                            logger.report_scalar(title="bpp", series="eval", value=avg_eval_bpp, iteration=global_step)
                    except Exception as e:
                        print(f"[Warning] Logging to ClearML failed: {e}")
                else:
                    print(message)
                gc.collect()
                torch.cuda.empty_cache()

            alpha_scheduler.step()
            for module in model.modules():
                if isinstance(module, GradualIntroductionLayer):
                    module.set_alpha(alpha_scheduler.get_value())

            if global_step > aux_optimizer_delay and aux_optimizer is not None and global_step % test_frequency == 0:
                avg_test_psnr = 0
                avg_test_ssim = 0
                avg_test_bpp = 0
                update_on_cpu(model)
                with torch.no_grad():
                    for x_test_in, x_test in test_dataloader:
                        x_test_in, x_test = x_test_in.to(device).detach(), x_test.to(device).detach()
                        compress_output = model.compress(x_test_in)
                        b_repr, shape = compress_output['strings'], compress_output['shape']
                        x_hat_test = model.decompress(b_repr, shape)['x_hat']

                        psnr = peak_signal_noise_ratio(x_test, x_hat_test, data_range=(0.0, 1.0), reduction='elementwise_mean', dim=(2,3))
                        ssim = structural_similarity_index_measure(x_test, x_hat_test, data_range=(0.0, 1.0), reduction='elementwise_mean')
                        avg_test_psnr += psnr
                        avg_test_ssim += ssim

                        batch_size = len(b_repr[0])
                        bits = sum(len(stream) * 8 for group in b_repr for stream in group)
                        _, _, H, W = x_hat_test.shape
                        bpp = bits / (batch_size * H * W)
                        avg_test_bpp += bpp

                avg_test_bpp /= len(test_dataloader)
                avg_test_psnr /= len(test_dataloader)
                avg_test_ssim /= len(test_dataloader)

                del compress_output, x_hat_test, x_test_in, x_test
                gc.collect()
                torch.cuda.empty_cache()

                message = (f"{datetime.datetime.now().strftime('%H:%M:%S')} [TEST] Epoch {epoch}/{num_epochs}, Step: "
                           f"{global_step}, Step: {global_step}, PSNR: {avg_test_psnr:.4f}, SSIM: {avg_test_ssim:.4f}, "
                           f"bpp: {avg_test_bpp:.4f}")
                if logger is not None:
                    logger.report_text(message)
                    logger.report_scalar(title="PSNR", series="test", value=avg_test_psnr, iteration=global_step)
                    logger.report_scalar(title="SSIM", series="test", value=avg_test_ssim, iteration=global_step)
                    logger.report_scalar(title="bpp", series="test", value=avg_test_bpp, iteration=global_step)
                else:
                    print(message)

    return model

def update_on_cpu(model):
    model.to("cpu")
    model.update(force=True)
    torch.cuda.empty_cache()
    model.to("cuda")


def overwrite_lrs(optimizer, aux_optimizer, experiment):
    for group, config_group in zip(optimizer.param_groups, experiment.param_groups):
        group['lr'] = config_group['lr']
    aux_optimizer.param_groups[0]['lr'] = experiment.aux_lr

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
    seed = 81
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ==============================
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Device:", device)
    # assert torch.cuda.is_available(), "CUDA is not available."

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

    # scaler = GradScaler()
    scaler = None

    start_epoch = 1
    start_step = 0
    if args.resume_checkpoint:
        print(f"[INFO] Resuming training from checkpoint: {args.resume_checkpoint}")
        task, start_epoch, start_step = load_training_state_with_clearml_from_file(
            args.resume_checkpoint,
            model,
            optimizer,
            aux_optimizer,
            None,
            aux_scheduler,
            scaler,
            device
        )

    logger = task.get_logger() if task is not None else None
    overwrite_lrs(optimizer, aux_optimizer, experiment)
    # for i, param_group in enumerate(optimizer.param_groups):
    #     param_group['initial_lr'] = 1e-4
    # for i, param_group in enumerate(optimizer.param_groups):
    #     param_group['lr'] = 1e-4
    # overwrite_lrs(optimizer, aux_optimizer, experiment)
    # scheduler = initializers.scheduler_from_config(optimizer, experiment_config['scheduler'])
    # state_dict = torch.load("/run/media/jakub/Dane/Studia/INZ/models/SWIN-S-IC_0.73.2.pth", map_location=device)
    #
    #
    # def safe_load_state_dict(model, loaded_state_dict):
    #     model_state = model.state_dict()
    #     clean_state_dict = {}
    #
    #     for k in loaded_state_dict:
    #         if k in model_state:
    #             if loaded_state_dict[k].shape == model_state[k].shape:
    #                 clean_state_dict[k] = loaded_state_dict[k]
    #             else:
    #                 print(f"Skipping incompatible shape for key '{k}': "
    #                       f"{loaded_state_dict[k].shape} vs {model_state[k].shape}")
    #         else:
    #             print(f"Skipping unknown key: {k}")
    #
    #     # Użyj "strict=False", żeby uniknąć błędów przy pominiętych parametrach
    #     model.load_state_dict(clean_state_dict, strict=False)
    #
    #
    # safe_load_state_dict(model, state_dict)
    # target_key = 'latent_codec.hyper.entropy_bottleneck._quantized_cdf'
    # expected_shape = (768, 117)
    #
    # if target_key in state_dict:
    #     tensor = state_dict[target_key]
    #     if tensor.shape[0] != expected_shape[0]:
    #         raise ValueError(f"Unexpected shape[0] for {target_key}: {tensor.shape}")
    #
    #     if tensor.shape[1] > expected_shape[1]:
    #         print(f"Truncating {target_key} from shape {tensor.shape} to {expected_shape}")
    #         state_dict[target_key] = tensor[:, :expected_shape[1]]
    #     elif tensor.shape[1] < expected_shape[1]:
    #         raise ValueError(f"{target_key} has too few elements: {tensor.shape}, expected {expected_shape}")
    #
    # filtered_state_dict = {
    #     k: v for k, v in state_dict.items() if 'latent_codec' not in k
    # }
    #
    # # Opcjonalnie pokaż usunięte klucze
    # removed_keys = set(state_dict.keys()) - set(filtered_state_dict.keys())
    # print("Removed quantiles keys:")
    # for key in removed_keys:
    #     print(f" - {key}")
    #
    # # Załaduj do modelu
    # model.load_state_dict(filtered_state_dict, strict=False)
    # start_step = 325000
    # loss.l = 1.35e-3
    # for i, param_group in enumerate(optimizer.param_groups):
    #     param_group['lr'] = 7.5e-6
    #     param_group['betas'] = (0.9, 0.95)
    #     param_group['weight_decay'] = 1e-6
    # for i, param_group in enumerate(aux_optimizer.param_groups):
    #     param_group['lr'] = 2.5e-5
    # optimizer.param_groups[0]['lr'] = 1e-5
    # optimizer.param_groups[1]['lr'] = 1e-5
    # optimizer.param_groups[2]['lr'] = 1e-5
    # optimizer.param_groups[3]['lr'] = 1e-5
    # # optimizer.param_groups[4]['lr'] = 1e-7
    # aux_optimizer.param_groups[0]['lr'] = 1e-5
    # experiment.accumulation_steps = 1
    # loss.distortion_loss.alpha = 0
    # loss.l = 1e-6
    # for name, param in model.parameters(named=True):
    #     if re.match(r"g_a\.(stages|residual_norms|downsamplers)\.[01]", name):
    #         param.requires_grad = False
    # model.g_a.stages[0].to("cpu")
    # model.g_a.stages[1].to("cpu")
    # model.g_a.residual_norms[0].to("cpu")
    # model.g_a.residual_norms[1].to("cpu")
    # model.g_a.downsamplers[0].to("cpu")
    # model.g_a.downsamplers[1].to("cpu")
    # loss.distortion_loss = MSESSIM(alpha=0.2)
    # with open(f"logs/{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.log", "a") as log_file:

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
        global_step=start_step,
        target_lambda=loss.l if isinstance(loss, RDLoss) else 0,
        log_frequency=experiment.log_frequency,
        accumulation_steps=experiment.accumulation_steps
    )

    # TODO: TRAIN TIME
    if not model.no_compress:
        update_on_cpu(model)

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
