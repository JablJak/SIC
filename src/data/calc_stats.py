import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.mixed_lic_dataset import MixedLICDataset

BATCH_SIZE = 1
NUM_WORKERS = 4


def calculate_stats():
    dataset = MixedLICDataset(variant="train", transform=transforms.Compose([
           transforms.PILToTensor(),
           transforms.ConvertImageDtype(torch.float),
       ]))

    print(f"Znaleziono {len(dataset)} obrazów.")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    mean = 0.0
    std = 0.0
    total_images_count = 0

    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    print("\n" + "=" * 40)
    print(f"Obliczone wartości dla: {type(dataset)}")
    print("=" * 40)
    print(f"Mean: {mean.tolist()}")
    print(f"Std:  {std.tolist()}")
    print("=" * 40)


if __name__ == "__main__":
    calculate_stats()