import os
import shutil

import kagglehub
from torch import nn

from src.data.base_dataset import BaseDataset
from src.utils.const import PROJECT_ROOT

COCO_ROOT = f"{PROJECT_ROOT}/data/datasets/awsaf49/coco-2017-dataset/versions/2/coco2017"

class CocoDataset(BaseDataset):
    def __init__(self, variant: str | None, transform=None, target_transform=None):
        if not os.path.exists(COCO_ROOT) or len(os.listdir(COCO_ROOT)) == 0:
            path = kagglehub.dataset_download("awsaf49/coco-2017-dataset")
            print("Path to dataset files:", path)
        if not os.path.exists(os.path.join(COCO_ROOT, "train")):
            os.mkdir(os.path.join(COCO_ROOT, "train"))
            shutil.move(os.path.join(COCO_ROOT, "train2017"), os.path.join(COCO_ROOT, "train"))
        if not os.path.exists(os.path.join(COCO_ROOT, "val")):
            os.mkdir(os.path.join(COCO_ROOT, "val"))
            shutil.move(os.path.join(COCO_ROOT, "val2017"), os.path.join(COCO_ROOT, "val"))
        if not os.path.exists(os.path.join(COCO_ROOT, "test")):
            os.mkdir(os.path.join(COCO_ROOT, "test"))
            shutil.move(os.path.join(COCO_ROOT, "test2017"), os.path.join(COCO_ROOT, "test"))

        self.variant = self._choose_variant(variant)
        variant_path = f"{variant}" if variant is not None else ""
        super().__init__(root_dir=os.path.join(COCO_ROOT, variant_path), transform=transform, target_transform=target_transform)

    def _choose_variant(self, variant: str | None) -> str | None:
        if variant is None:
            return None
        allowed_variants = [
            "train",
            "test",
            "val"
        ]
        if variant not in allowed_variants:
            raise ValueError(f"Variant {variant} is not allowed")
        else:
            return variant

    def has_val(self):
        return True

    @classmethod
    def validation_set(cls, transform: nn.Module | None) -> "CocoDataset":
        return cls(variant="val2017", transform=transform)