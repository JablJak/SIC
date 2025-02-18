import mlflow
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.models import Swin_V2_T_Weights
from torchvision.transforms._presets import ImageClassification

from src.models.swin_autoencoder import SwinTransformerAutoencoder
from src.data.imagenet_dataset import ImageNetDataset
from src.losses.mse_ssim import MSE_SSIM
from src.utils.postprocess import denormalize
from src.viz.plotter import plot_reconstructions

import matplotlib.pyplot as plt

def train(model, dataloader, criterion, optimizer, num_epochs):
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
        mlflow.log_metric("epoch_loss", epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
        torch.save(model.state_dict(), f"checkpoint/model_v1.pth")
    return model

if __name__ == '__main__':
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("/mlflow-pytorch-quickstart")

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
    model.load_state_dict(torch.load(f"checkpoint/model_v0.1.0.pth", map_location=device, weights_only=True))

    loss = MSE_SSIM()
    loss.to(device)

    train(
        model=model,
        dataloader=train_dataloader,
        criterion=loss,
        optimizer=AdamW(model.parameters(), lr=8e-5, weight_decay=1e-2),
        num_epochs=5
    )

    model.eval()
    with torch.no_grad():
        x_batch, _ = next(iter(val_dataloader))
        x_batch = x_batch.to(device)
        
        x_recon = model(x_batch)

    x_batch = denormalize(x_batch, ImageClassification(crop_size=0).mean, ImageClassification(crop_size=0).std).cpu()
    x_recon = denormalize(x_recon, ImageClassification(crop_size=0).mean, ImageClassification(crop_size=0).std).cpu()

    mlflow.pytorch.log_model(model, "model_v0.1.1")

    plot_reconstructions(x_batch, x_recon)