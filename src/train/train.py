import torch
from clearml import Task, OutputModel
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.models import Swin_V2_T_Weights
from torchvision.transforms._presets import ImageClassification

from src.models.swin_autoencoder import SwinTransformerAutoencoder
from src.data.imagenet_dataset import ImageNetDataset
from src.losses.mse_ssim import MSESSIM
from src.utils.const import PROJECT_ROOT
from src.utils.postprocess import denormalize
from src.viz.plotter import plot_reconstructions


def train(model, dataloader, criterion, optimizer, num_epochs, logger):
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

        torch.save(model.state_dict(), f"{PROJECT_ROOT}/checkpoint/model_v0.2.0.pth")
    return model

if __name__ == '__main__':
    task = Task.init(project_name="INZ", task_name="Init")

    logger = task.get_logger()

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print("Device:", device)

    transform = Swin_V2_T_Weights.DEFAULT.transforms()

    imagenet_dataset = ImageNetDataset(transform=transform)
    imagenet_subset = imagenet_dataset.subset(num_classes=2, num_samples=50)
    train_dataset, val_dataset = random_split(imagenet_subset, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=8)

    model = SwinTransformerAutoencoder()
    model.to(device)

    loss = MSESSIM()
    loss.to(device)

    train(
        model=model,
        dataloader=train_dataloader,
        criterion=loss,
        optimizer=AdamW(model.parameters(), lr=8e-5, weight_decay=1e-2),
        num_epochs=5,
        logger=logger
    )

    model.eval()
    with torch.no_grad():
        x_batch, _ = next(iter(val_dataloader))
        x_batch = x_batch.to(device)
        
        x_recon = model(x_batch)

    x_batch = denormalize(x_batch, ImageClassification(crop_size=0).mean, ImageClassification(crop_size=0).std).cpu()
    x_recon = denormalize(x_recon, ImageClassification(crop_size=0).mean, ImageClassification(crop_size=0).std).cpu()

    output_model = OutputModel(task=task, name="init_v0.2.0")
    output_model.update_weights("checkpoint/model_v0.2.0.pth")
    # output_model.comment("Initial test pretrained model")

    model_id = output_model.id
    print(f"Saved ClearML model with ID: {model_id}")


    plot_reconstructions(x_batch, x_recon)