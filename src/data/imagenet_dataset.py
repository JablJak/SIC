import os

from src.data.base_dataset import BaseDataset


class ImageNetDataset(BaseDataset):
    def __init__(self, transform=None):
        super().__init__(root_dir="data/imagenet", transform=transform)