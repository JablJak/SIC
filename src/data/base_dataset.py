import copy
from pathlib import Path
from typing import Union

from torchvision.datasets import ImageFolder


class BaseDataset(ImageFolder):
    def __init__(self, root_dir, transform=None, csv_file=None):
        """
        Args:
        root_dir (string): Directory with the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        csv_file (string, optional): Path to the csv file with annotations.
        """
        super().__init__(root_dir, transform)
        self.root_dir = root_dir
        self.transform = transform
        self.csv_file = csv_file
        self.dataset = ImageFolder(root=root_dir, transform=transform)

    def subset(self, num_classes=50, num_samples=500):
        selected_samples = []
        class_counts = {}
        for sample_path, label in self.dataset.samples:
            if label not in class_counts:
                class_counts[label] = 0
            if class_counts[label] < num_samples:
                selected_samples.append((sample_path, label))
                class_counts[label] += 1
            if len(selected_samples) == num_classes * num_samples:
                break

        result = copy.deepcopy(self.dataset)
        result.samples = selected_samples
        result.targets = [lbl for (_, lbl) in selected_samples]
        return result