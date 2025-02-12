import copy
import torch
from torch.nn import MSELoss
from torch.optim import AdamW
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms._presets import ImageClassification

from swin_autoencoder import SwinTransformerAutoencoder

def crop_dataset(dataset: ImageFolder, num_classes=5, num_samples=200):
    selected_samples = []
    class_counts = {}
    for sample_path, label in dataset.samples:
        if label not in class_counts:
            class_counts[label] = 0
        if class_counts[label] < num_samples:
            selected_samples.append((sample_path, label))
            class_counts[label] += 1
        if len(selected_samples) == num_classes * num_samples:
            break

    result = copy.deepcopy(dataset)
    result.samples = selected_samples
    result.targets = [lbl for (_, lbl) in selected_samples]
    return result


def train(model, dataloader, criterion, optimizer, num_epochs):
    torch.save(model.state_dict(), f"checkpoint/model.pth")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    return model

def denormalize(x, mean, std):
    tensor = x
    if x.ndim == 3:
        tensor = tensor.unsqueeze(0)
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


if __name__ == '__main__':
    model = SwinTransformerAutoencoder()
    
    transform = model.encoder_weights.transforms()
    
    imagenet_train_dataset = ImageFolder("data/imagenet", transform=transform)
    cropped_train_dataset = crop_dataset(imagenet_train_dataset)
    dataloader = DataLoader(cropped_train_dataset, batch_size=16, shuffle=True, num_workers=8)
    
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model.to(device)
    print("Device:", device)
    
    train(model, dataloader, criterion=MSELoss(), optimizer=AdamW(model.parameters(), lr=1e-4), num_epochs=50)
    
    model.eval()
    with torch.no_grad():
        x_batch, _ = next(iter(dataloader))
        x_batch = x_batch.to(device)
        
        x_recon = model(x_batch)

    x_batch = denormalize(x_batch, ImageClassification(crop_size=0).mean, ImageClassification(crop_size=0).std).cpu()
    x_recon = denormalize(x_recon, ImageClassification(crop_size=0).mean, ImageClassification(crop_size=0).std).cpu()

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(4):
        
        axes[0, i].imshow(x_batch[i].permute(1, 2, 0).detach().numpy())
        axes[0, i].set_title("Oryginał")
        axes[0, i].axis("off")

        
        axes[1, i].imshow(x_recon[i].permute(1, 2, 0).detach().numpy())
        axes[1, i].set_title("Rekonstrukcja")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()