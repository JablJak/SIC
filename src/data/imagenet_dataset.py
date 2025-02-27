from torchvision.models import Swin_V2_T_Weights

from src.data.base_dataset import BaseDataset
from src.utils.const import PROJECT_ROOT


class ImageNetDataset(BaseDataset):
    def __init__(self, transform=Swin_V2_T_Weights.DEFAULT.transforms()):
        super().__init__(root_dir=f"{PROJECT_ROOT}/data/imagenet", transform=transform)