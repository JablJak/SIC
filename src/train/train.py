import torch
from clearml import Task, OutputModel
from torch.utils.data import DataLoader, random_split
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.models import Swin_V2_T_Weights
from torchvision.transforms._presets import ImageClassification

from src.train.experiment import Experiment
from src.utils.const import PROJECT_ROOT, MODEL_CHECKPOINT_PATH, MODEL_CHECKPOINT_FILE, EXPERIMENTS_CONFIG_PATH, \
    MODEL_OUTPUT_PATH
from src.utils.initializers import read_config, dataloader_from_config
from src.utils.postprocess import denormalize
from src.viz.plotter import plot_reconstructions


def _train(model, dataloader, criterion, optimizer, num_epochs, logger):
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0

        for x, _ in dataloader:
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

        avg_loss = epoch_loss / len(dataloader)
        avg_psnr = epoch_psnr / len(dataloader)
        avg_ssim = epoch_ssim / len(dataloader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

        logger.report_scalar(title="Loss", series="train", value=avg_loss, iteration=epoch)
        logger.report_scalar(title="PSNR", series="train", value=avg_psnr, iteration=epoch)
        logger.report_scalar(title="SSIM", series="train", value=avg_ssim, iteration=epoch)

        torch.save(model.state_dict(), f"{MODEL_CHECKPOINT_PATH}/{MODEL_CHECKPOINT_FILE}")
    return model

if __name__ == '__main__':
    experiment_config = read_config(f"{EXPERIMENTS_CONFIG_PATH}/experiment_v0.2.0.yaml")
    experiment = Experiment(experiment_config)

    task = Task.init(project_name=experiment.project, task_name=experiment.task)
    task.connect(experiment_config)
    task.set_comment(experiment.comment)
    logger = task.get_logger()

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

    _train(
        model=model,
        dataloader=train_dataloader,
        criterion=loss,
        optimizer=optimizer,
        num_epochs=experiment.epochs,
        logger=logger
    )

    # TODO: TRAIN TIME

    model.eval()
    with torch.no_grad():
        x_batch, _ = next(iter(val_dataloader))
        x_batch = x_batch.to(device)
        
        x_recon = model(x_batch)

    x_batch = denormalize(x_batch, ImageClassification(crop_size=0).mean, ImageClassification(crop_size=0).std).cpu()
    x_recon = denormalize(x_recon, ImageClassification(crop_size=0).mean, ImageClassification(crop_size=0).std).cpu()

    plot_reconstructions(x_batch, x_recon)

    output_model = OutputModel(task=task, name=experiment.model_name())
    output_model.set_upload_destination(f"{MODEL_OUTPUT_PATH}")
    output_model.update_weights(f"{MODEL_OUTPUT_PATH}/{experiment.model_name()}.pth")

    model_id = output_model.id
    print(f"Saved ClearML model with ID: {model_id}")