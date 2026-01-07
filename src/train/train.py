import argparse
import datetime
import gc
import os
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from src.losses.rdloss import RDLoss
from src.models.gdn_swin_transformer import LinearScheduler, GradualIntroductionLayer
from src.train.experiment import Experiment
from src.utils import clearml_helpers
from src.utils.checkpoint_helpers import save_training_state_with_clearml, load_training_state_with_clearml_from_file
from src.utils.clearml_helpers import  start_experiment
from src.utils.const import MODEL_CHECKPOINT_PATH, EXPERIMENTS_CONFIG_PATH, MODEL_OUTPUT_PATH, ARTIFACTS_PATH
from src.utils.initializers import read_config, dataloader_from_config

def _train(model, train_dataloader, val_dataloader, test_dataloader, criterion, optimizer, aux_optimizer,
           aux_scheduler, num_epochs, device, scheduler, logger=None, task=None, start_epoch=1, global_step=0,
           log_frequency=100, accumulation_steps=4):
    # Train
    optimize_bpp = False
    max_norm_value = 2
    temp_checkpoint_frequency = 1000
    persist_checkpoint_frequency = 5000
    eval_frequency = 500
    test_frequency = 1000
    torch.cuda.empty_cache()
    interval_loss = 0
    interval_bpp = 0
    interval_aux_loss = 0
    optimizer_stepped = False
    aux_optimizer_stepped = False
    for epoch in range(start_epoch, num_epochs + 1):
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        psnr_metric = PeakSignalNoiseRatio(data_range=(0.0, 1.0), reduction='elementwise_mean', dim=(2, 3)).to(device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0), reduction='elementwise_mean').to(device)
        for i, (x_in, x) in enumerate(train_dataloader):
            model.train()
            if aux_optimizer is not None:
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

            if aux_optimizer is not None:
                aux_loss = model.aux_loss()
                scaled_aux_loss = aux_loss / accumulation_steps
                scaled_aux_loss.backward()
                interval_aux_loss += aux_loss.item()

            interval_loss += loss.item()
            interval_bpp += bpp.item()

            psnr_metric.update(x_hat.detach(), x.detach())
            ssim_metric.update(x_hat.detach(), x.detach())

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm_value)
                optimizer.step()
                optimizer_stepped = True
                if aux_optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(aux=True), max_norm=max_norm_value/2)
                    aux_optimizer.step()
                    aux_optimizer_stepped = True
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
                message = f"{datetime.datetime.now().strftime('%H:%M:%S')} [TRAIN] Epoch {epoch}/{num_epochs}, " \
                          f"Step: {global_step}, Loss: {avg_interval_loss:.5f}, PSNR: {avg_interval_psnr:.4f}, " \
                          f"SSIM: {avg_interval_ssim:.4f}, bpp: {avg_interval_bpp:.4f}, Aux loss: " \
                          f"{avg_interval_aux_loss:.5f}, {lrs}, aux_lr: {aux_optimizer.param_groups[0]['lr']:4g}"
                if logger is not None:
                    try:
                        logger.report_text(message)
                        logger.report_scalar(title="Loss", series="train", value=avg_interval_loss, iteration=global_step)
                        logger.report_scalar(title="Aux loss", series="train", value=avg_interval_aux_loss, iteration=global_step)
                        logger.report_scalar(title="PSNR", series="train", value=avg_interval_psnr, iteration=global_step)
                        logger.report_scalar(title="SSIM", series="train", value=avg_interval_ssim, iteration=global_step)
                        logger.report_scalar(title="LR", series="train", value=optimizer.param_groups[0]['lr'], iteration=global_step)
                        logger.report_scalar(title="Conv LR", series="train", value=optimizer.param_groups[1]['lr'], iteration=global_step)
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
                    current_epoch=epoch,
                    current_step=global_step,
                    save_path=f"{MODEL_CHECKPOINT_PATH}/checkpoint_{global_step}.pth"
                )

            if scheduler is not None and optimizer_stepped:
                scheduler.step()

            if aux_scheduler is not None and epoch and aux_optimizer is not None and aux_optimizer_stepped:
                aux_scheduler.step()

            if global_step % eval_frequency == 0:
                # Eval
                model.eval()
                psnr_metric_val = PeakSignalNoiseRatio(data_range=(0.0, 1.0), reduction='elementwise_mean', dim=(2, 3)).to(device)
                ssim_metric_val = StructuralSimilarityIndexMeasure(data_range=(0.0, 1.0), reduction='elementwise_mean').to(device)
                eval_loss = 0
                eval_bpp = 0
                eval_aux_loss = 0

                with torch.no_grad():
                    for x_val_in, x_val in val_dataloader:
                        x_val_in, x_val = x_val_in.to(device).detach(), x_val.to(device).detach()
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

                        if aux_optimizer is not None:
                            aux_loss_val = model.aux_loss()
                            eval_aux_loss += aux_loss_val.item()

                        psnr_metric_val.update(x_hat_val, x_val)
                        ssim_metric_val.update(x_hat_val, x_val)

                avg_eval_psnr = psnr_metric_val.compute()
                avg_eval_ssim = ssim_metric_val.compute()

                avg_eval_loss = eval_loss / len(val_dataloader)
                avg_eval_bpp = eval_bpp / len(val_dataloader)
                avg_eval_aux_loss = eval_aux_loss / len(val_dataloader)

                del loss_val, output, x_hat_val, likelihoods_val, x_val_in, x_val
                gc.collect()
                torch.cuda.empty_cache()

                message = f"{datetime.datetime.now().strftime('%H:%M:%S')} [VAL] Epoch {epoch}/{num_epochs}, " \
                          f"Step: {global_step}, Loss: {avg_eval_loss:.4f}, PSNR: {avg_eval_psnr:.4f}, SSIM: " \
                          f"{avg_eval_ssim:.4f}, bpp: {avg_eval_bpp:.4f}, Aux loss: {avg_eval_aux_loss:.4f}"
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

            if aux_optimizer is not None and global_step % test_frequency == 0:
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

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Device:", device)

    train_dataset, val_dataset, test_dataset = experiment.train_dataset, experiment.val_dataset, experiment.test_dataset

    train_dataloader = dataloader_from_config(train_dataset, experiment_config['dataloader'])
    val_dataloader = dataloader_from_config(val_dataset, experiment_config['dataloader'])
    test_dataloader = dataloader_from_config(test_dataset, experiment_config['dataloader'])

    model = experiment.model
    model.to(device)

    loss = experiment.loss
    loss.to(device)

    optimizer = experiment.optimizer
    scheduler = experiment.scheduler

    aux_optimizer = experiment.aux_optimizer
    aux_scheduler = experiment.aux_scheduler

    start_epoch = 1
    start_step = 0
    if args.resume_checkpoint:
        print(f"[INFO] Resuming training from checkpoint: {args.resume_checkpoint}")
        task, start_epoch, start_step = load_training_state_with_clearml_from_file(
            args.resume_checkpoint,
            model,
            optimizer,
            aux_optimizer,
            scheduler,
            aux_scheduler,
            device
        )

    logger = task.get_logger() if task is not None else None
    overwrite_lrs(optimizer, aux_optimizer, experiment)
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
        start_epoch=start_epoch,
        global_step=start_step,
        log_frequency=experiment.log_frequency,
        accumulation_steps=experiment.accumulation_steps
    )

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
