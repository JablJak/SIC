import random
from math import ceil, floor
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torchvision.transforms.v2
from torch import Tensor
from torchvision.transforms import InterpolationMode, functional
from torchvision.transforms.v2 import RandomResizedCrop, Transform
from torchvision.transforms.v2.functional import pad

RGB_IMAGENET_MEAN = (0.485, 0.456, 0.406)
RGB_IMAGENET_STD = (0.229, 0.224, 0.225)
RGB_COCO_MEAN = (0.470, 0.447, 0.408)
RGB_COCO_STD = (0.270, 0.266, 0.281)
RGB_MIXED_LIC_MEAN = (0.448, 0.429, 0.392)
RGB_MIXED_LIC_STD = (0.240, 0.230, 0.242)
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
            mean: Tuple[float, ...] = RGB_COCO_MEAN,
            std: Tuple[float, ...] = RGB_COCO_STD,
            interpolation: InterpolationMode = InterpolationMode.BICUBIC,
            crop_size: list[int] = [256, 256],
            resize_size: list[int] = [256, 256],
            antialias: Optional[bool] = True,
            noresize = False,
            normalize = True
    ) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.interpolation = interpolation
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.antialias = antialias
        self.noresize = noresize
        self.normalize = normalize
        self.crop = RandomResizedCropInScales(size=self.crop_size)

    def forward(self, img: Tensor) -> Tuple[Tensor, Tensor]:
        if not isinstance(img, Tensor):
            img = functional.pil_to_tensor(img)
        if not self.noresize:
            img = self.crop(img)
        img = functional.convert_image_dtype(img, torch.float)
        norm_img = functional.normalize(img, mean=list(self.mean), std=list(self.std))
        return norm_img, img


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

class RandomResizedCropInScales(nn.Module):
    def __init__(self,
                 size: int | tuple[int, int],
                 scales: tuple[float, ...] = (1.0, 2.0, 4.0),
                 interpolation: InterpolationMode | str = "area"
                 ):
        super().__init__()
        self.size = size
        self.scales = scales
        self.interpolation = interpolation

        if isinstance(self.size, (tuple, list)):
            self.target_H, self.target_W = self.size
        else:
            self.target_H = self.target_W = self.size

    def forward(self, img: Tensor) -> Tensor:
        H, W = img.shape[-2:]
        scales = self._eligible_scales(img)
        if len(scales) == 0:
            w_pad = max(0, self.target_W - W)
            h_pad = max(0, self.target_H - H)
            l_pad, r_pad = floor(w_pad / 2), ceil(w_pad / 2)
            t_pad, b_pad = floor(h_pad / 2), ceil(h_pad / 2)
            img = functional.pad(img, [l_pad, t_pad, r_pad, b_pad], padding_mode="reflect")
            H, W = img.shape[-2:]
            selected_scale = 1.0
        else:
            selected_scale = random.choice(scales)
        downscaled_img = functional.resize(img, [round(H / selected_scale), round(W / selected_scale)],
                                           interpolation=InterpolationMode.BOX, antialias=False)

        ds_H, ds_W = downscaled_img.shape[-2:]

        top = random.randint(0, ds_H - self.target_H)
        left = random.randint(0, ds_W - self.target_W)

        crop = functional.crop(img=downscaled_img, top=top, left=left, height=self.target_H, width=self.target_W)

        return crop

    def _eligible_scales(self, img: Tensor) -> tuple[float, ...]:
        H, W = img.shape[-2:]
        eligible_scales = []
        for scale in self.scales:
            if H >= self.target_H * scale and W >= self.target_W * scale:
                eligible_scales.append(scale)
        return tuple(eligible_scales)



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