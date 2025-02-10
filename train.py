import copy

from torchvision.datasets import ImageNet, DatasetFolder
from torch.utils.data import DataLoader
from torchvision import transforms

from swin_autoencoder import SwinTransformerAutoencoder


def crop_dataset(dataset: DatasetFolder, num_classes=5, num_samples=200):
    selected_samples = []
    class_counts = {}
    for label, sample in dataset.samples:
        if label not in class_counts:
            class_counts[label] = 0
        if class_counts[label] < num_samples:
            selected_samples.append(sample)
            class_counts[label] += 1
        if len(selected_samples) == num_classes * num_samples:
            break
            
    result = copy.deepcopy(dataset)
    result.samples = selected_samples
    result.targets = class_counts.keys()
    return result


if __name__ == '__main__':
    imagenet_dataset = ImageNet("./data/imagenet")
    model = SwinTransformerAutoencoder()
    transform = model.encoder_weights.transforms
    imagenet_dataset.transform = transform
    cropped_dataset = crop_dataset(imagenet_dataset)
    dataloader = DataLoader(cropped_dataset, batch_size=32, shuffle=True)

    # Example iteration over the DataLoader
    for batch in dataloader:
        print(batch)
