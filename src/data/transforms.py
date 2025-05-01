from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import InterpolationMode, functional

RGB_IMAGENET_MEAN = (0.485, 0.456, 0.406)
RGB_IMAGENET_STD = (0.229, 0.224, 0.225)
RGB_COCO_MEAN = (0.470, 0.447, 0.408)
RGB_COCO_STD = (0.270, 0.266, 0.281)
YCBCR_IMAGENET_MEAN = (0.459, -0.030, 0.019)
YCBCR_IMAGENET_STD = (0.150, 0.140, 0.149)
RGB_TO_YCBCR_MAT = [
    [0.299, 0.587, 0.114],
    [-0.168736, -0.331264, 0.5],
    [0.5, -0.418688, -0.081312]
]
YCBCR_TO_RGB_MAT = [
    [1.0, 0.0, 1.402],
    [1.0, -0.344136, -0.714136],
    [1.0, 1.772, 0.0]
]

class RGBCompression(nn.Module):
    def __init__(
            self,
            mean: Tuple[float, ...] = RGB_IMAGENET_MEAN,
            std: Tuple[float, ...] = RGB_IMAGENET_STD,
            interpolation: InterpolationMode = InterpolationMode.BICUBIC,
            crop_size: int = 256,
            resize_size: int = 256,
            antialias: Optional[bool] = True,
            noresize = False,
            normalize = True
    ) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.interpolation = interpolation
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        self.antialias = antialias
        self.noresize = noresize
        self.normalize = normalize

    def forward(self, img: Tensor) -> Tensor:
        if not self.noresize:
            img = functional.resize(img, self.resize_size, interpolation=self.interpolation, antialias=self.antialias)
            img = functional.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = functional.pil_to_tensor(img)
        img = functional.convert_image_dtype(img, torch.float)
        if self.normalize:
            img = functional.normalize(img, mean=list(self.mean), std=list(self.std))
        return img


class RGBDecompression(nn.Module):
    def __init__(
            self,
            denorm: bool,
            mean: Tuple[float, ...] = RGB_IMAGENET_MEAN,
            std: Tuple[float, ...] = RGB_IMAGENET_STD,
    ) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.denorm = denorm

    def forward(self, img: Tensor) -> Tensor:
        if self.denorm:
            img = denormalize(img, self.mean, self.std)
        single_img = img.dim() == 3

        pil_images = []
        for i in range(img.shape[0]):
            single_img_tensor = img[i]
            pil_img = functional.to_pil_image(single_img_tensor)
            pil_images.append(pil_img)

        if single_img:
            return pil_images[0]
        else:
            return pil_images


class YCbCrCompression(nn.Module):
    def __init__(
            self,
            mean: Tuple[float, ...] = YCBCR_IMAGENET_MEAN,
            std: Tuple[float, ...] = YCBCR_IMAGENET_STD,
            interpolation: InterpolationMode = InterpolationMode.BICUBIC,
            crop_size: int = 256,
            resize_size: int = 260,
            antialias: Optional[bool] = True,
            noresize=False
    ) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.interpolation = interpolation
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        self.antialias = antialias
        self.noresize = noresize


    def forward(self, img: Tensor) -> Tensor:
        if not self.noresize:
            img = functional.resize(img, self.resize_size, interpolation=self.interpolation, antialias=self.antialias)
            img = functional.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = functional.pil_to_tensor(img)
        img = functional.convert_image_dtype(img, torch.float)
        img = rgb_to_ycbcr(img)
        img = functional.normalize(img, mean=list(self.mean), std=list(self.std))
        return img


class YCbCrDecompression(nn.Module):
    def __init__(
            self,
            denorm: bool,
            mean: Tuple[float, ...] = YCBCR_IMAGENET_MEAN,
            std: Tuple[float, ...] = YCBCR_IMAGENET_STD,
    ) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.denorm = denorm

    def forward(self, img: Tensor) -> list:
        if self.denorm:
            img = denormalize(img, self.mean, self.std)
        img = ycbcr_to_rgb(img)
        single_img = img.dim() == 3

        pil_images = []
        for i in range(img.shape[0]):
            single_img_tensor = img[i]
            pil_img = functional.to_pil_image(single_img_tensor)
            pil_images.append(pil_img)

        if single_img:
            return pil_images[0]
        else:
            return pil_images


def denormalize(img: Tensor, mean: Tuple[float, ...], std: Tuple[float, ...]) -> Tensor:
    if img.ndim == 3:
        img = img.unsqueeze(0)

    mean = torch.tensor(mean, device=img.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=img.device).view(1, -1, 1, 1)

    return img * std + mean

def rgb_to_ycbcr(img: Tensor) -> Tensor:
    single_img = False
    if img.dim() == 3:
        single_img = True
        img = img.unsqueeze(0)
    transform_mat = torch.tensor(RGB_TO_YCBCR_MAT).float().to(img.device)

    b, c, h, w = img.shape
    x_reshaped = img.view(b, 3, -1)

    ycbcr = torch.matmul(transform_mat, x_reshaped)
    ycbcr = ycbcr.view(b, 3, h, w)

    y_channel = ycbcr[:, 0, :, :]

    if torch.any(y_channel > 1):
        print(f"[WARNING] Y channel max > 1 after transformation: {y_channel.max().item():.4f}")

    if torch.any(y_channel < 0):
        print(f"[WARNING] Y channel min < 0 after transformation: {y_channel.min().item():.4f}")

    ycbcr[:, 0, :, :] = torch.clamp(ycbcr[:, 0, :, :], 0.0, 1.0)

    if single_img:
        ycbcr = ycbcr.squeeze()
    return ycbcr


def ycbcr_to_rgb(img: Tensor, recon = False) -> Tensor:
    single_img = False
    if img.dim() == 3:
        single_img = True
        img = img.unsqueeze(0)
    transform_mat = torch.tensor(YCBCR_TO_RGB_MAT).float().to(img.device)

    b, c, h, w = img.shape
    x_reshaped = img.view(b, 3, -1)

    rgb = torch.matmul(transform_mat, x_reshaped)
    rgb = rgb.view(b, 3, h, w)

    if rgb.max() > 1.00001:
        print(f"[WARNING] RGB max > 1 after transformation: {rgb.max().item():.4f}, recon: {recon}")

    if rgb.min() < -0.00001:
        print(f"[WARNING] RGB min < 0 after transformation: {rgb.min().item():.4f}, recon: {recon}")

    rgb = torch.clamp(rgb, 0.0, 1.0)

    if single_img:
        rgb = rgb.squeeze()

    return rgb