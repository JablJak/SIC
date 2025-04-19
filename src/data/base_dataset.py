import copy
from typing import Any

from torch import nn
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder


class BaseDataset(ImageFolder):
    def __init__(self, root_dir, transform=None, target_transform=None, csv_file=None):
        """
        Args:
        root_dir (string): Directory with the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        csv_file (string, optional): Path to the csv file with annotations.
        """
        super().__init__(root_dir, transform)
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.csv_file = csv_file

    def subset(self, num_classes=50, num_samples=500):
        selected_samples = []
        class_counts = {}
        for sample_path, label in self.samples:
            if label not in class_counts:
                class_counts[label] = 0
            if class_counts[label] < num_samples:
                selected_samples.append((sample_path, label))
                class_counts[label] += 1
            if len(selected_samples) == num_classes * num_samples:
                break

        result = copy.deepcopy(self)
        result.samples = selected_samples
        result.targets = [lbl for (_, lbl) in selected_samples]
        return result

    def has_val(self):
        return False

    def train_test_split(self, ratio: float = 0.8):
        return random_split(self, [ratio, 1 - ratio])

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        orig_sample = self.loader(path)
        sample = self.transform(orig_sample)
        target = self.target_transform(orig_sample)

        return sample, target


    @classmethod
    def validation_set(cls, transform: nn.Module | None) -> "BaseDataset":
        raise AttributeError(f"{cls.__name__} does not implement validate_set")