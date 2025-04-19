import torch
from torch import nn
from torchvision.models import Swin_V2_T_Weights
from torchvision.transforms import transforms

from src.data.base_dataset import BaseDataset
from src.utils.const import PROJECT_ROOT


class ImageNetDataset(BaseDataset):
    def __init__(self, variant: str | None, transform: nn.Module = transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            Swin_V2_T_Weights.DEFAULT.transforms()
        ]
    ), target_transform=None
        ):
        self.variant = variant
        variant_path = f"/{variant}" if variant is not None else ""
        super().__init__(root_dir=f"{PROJECT_ROOT}/data/imagenet{variant_path}", transform=transform, target_transform=target_transform)

    def _choose_variant(self, variant: str | None) -> str | None:
        if variant is None:
            return None
        allowed_variants = [
            "train.X1",
            "train.X2",
            "train.X3",
            "train.X4",
            "val.X"
        ]
        if variant not in allowed_variants:
            raise ValueError(f"Variant {variant} is not allowed")
        else:
            return variant

    def has_val(self):
        return True

    @classmethod
    def validation_set(cls, transform: nn.Module | None) -> "ImageNetDataset":
        return cls(variant="val.X", transform=transform)