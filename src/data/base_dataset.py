import copy
import math

import torch
from torch import nn
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as f


def split_image(img, patch_size):
    H, W = img.shape[-2:]
    ph, pw = patch_size

    n_h = math.ceil(H / ph)
    n_w = math.ceil(W / pw)

    stride_h = (H - ph) / (n_h - 1) if n_h > 1 else 0
    stride_w = (W - pw) / (n_w - 1) if n_w > 1 else 0

    patches = []
    for i in range(n_h):
        for j in range(n_w):
            y0 = int(round(i * stride_h))
            x0 = int(round(j * stride_w))
            patch = img[..., y0:y0 + ph, x0:x0 + pw]
            patches.append(patch)
    return patches

class BaseDataset(ImageFolder):
    def __init__(self, root_dir, transform=None, target_transform=None, csv_file=None, patch_sizes=((256, 384), (384, 256)), train=False):
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
        self.patch_sizes = patch_sizes
        self.train = train

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
    #
    # def __getitem__(self, index: int) -> tuple[Any, Any]:
    #     """
    #     Args:
    #         index (int): Index
    #
    #     Returns:
    #         tuple: (sample, target) where target is class_index of the target class.
    #     """
    #     path, _ = self.samples[index]
    #     orig_sample = self.loader(path)
    #     sample = self.transform(orig_sample)
    #     target = self.target_transform(orig_sample)
    #
    #     return sample, target

    def __getitem__(self, index: int):
        if self.train:
            path, _ = self.samples[index]
            orig_sample = self.loader(path)

            img_tensor = f.to_tensor(orig_sample)

            sample_patch_groups = []
            target_patch_groups = []

            for ps in self.patch_sizes:
                patches = split_image(img_tensor, ps)

                sample_patches = patches
                if self.transform is not None:
                    sample_patches = [self.transform(p) for p in patches]

                target_patches = patches
                if self.target_transform is not None:
                    target_patches = [self.target_transform(p) for p in patches]

                sample_patch_groups.append(torch.stack(sample_patches))
                target_patch_groups.append(torch.stack(target_patches))

            return sample_patch_groups, target_patch_groups
        else:
            path, _ = self.samples[index]
            orig_sample = self.loader(path)
            sample = self.transform(orig_sample)
            target = self.target_transform(orig_sample)

            return sample, target

    @classmethod
    def validation_set(cls, transform: nn.Module | None) -> "BaseDataset":
        raise AttributeError(f"{cls.__name__} does not implement validate_set")