import torch
from torchvision.models import Swin_V2_T_Weights
from torchvision.transforms import transforms

from src.data.base_dataset import BaseDataset
from src.utils.const import PROJECT_ROOT


class KodakDataset(BaseDataset):
    # TODO: Docstring
    def __init__(self, transform=transforms.Compose(
        [
            # TODO: This can't be done in transform I guess
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            Swin_V2_T_Weights.DEFAULT.transforms()
        ]), target_transform=None):
        super().__init__(root_dir=f"{PROJECT_ROOT}/data/kodak", transform=transform, target_transform=target_transform)