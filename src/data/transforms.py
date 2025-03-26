from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import InterpolationMode, functional


class YCbCrCompression(nn.Module):
    def __init__(
            self,
            crop_size: int = 256,
            resize_size: int = 260,
            mean: Tuple[float, ...] = (0.4560, 0.5926, 1.0980),
            std: Tuple[float, ...] = (0.2266, 0.1483, 0.2506),
            interpolation: InterpolationMode = InterpolationMode.BICUBIC,
            antialias: Optional[bool] = True,
            noresize = False
    ) -> None:
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation
        self.antialias = antialias
        self.ycbcr = RGBToYCbCr()
        self.noresize = noresize

    def forward(self, img: Tensor) -> Tensor:
        if not self.noresize:
            img = functional.resize(img, self.resize_size, interpolation=self.interpolation, antialias=self.antialias)
            img = functional.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = functional.pil_to_tensor(img)
        img = functional.convert_image_dtype(img, torch.float)
        img = self.ycbcr(img)
        img = functional.normalize(img, mean=self.mean, std=self.std)
        return img

class RGBToYCbCr(nn.Module):
    """
    Warstwa PyTorch do konwersji tensora obrazu z przestrzeni RGB do YCbCr.
    Obsługuje tensory w kształcie (3, height, width) lub (batch_size, 3, height, width).

    Parametry:
        in_range (str): Zakres wartości wejściowych - '0_1' dla [0, 1] lub '0_255' dla [0, 255]
        out_range (str): Zakres wartości wyjściowych - '0_1' dla [0, 1] lub '0_255' dla [0, 255]
    """

    def __init__(self, in_range='0_1', out_range='0_1'):
        super(RGBToYCbCr, self).__init__()

        self.in_range = in_range
        self.out_range = out_range

        # Definiujemy współczynniki transformacji zgodnie ze standardem ITU-R BT.601
        self.register_buffer('matrix', torch.tensor([
            [0.299, 0.587, 0.114],  # Y
            [-0.168736, -0.331264, 0.5],  # Cb
            [0.5, -0.418688, -0.081312]  # Cr
        ]).float())

        # Offset dla Cb i Cr
        self.register_buffer('offset', torch.tensor([0, 128, 128]).view(1, 3, 1, 1).float())

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor RGB w kształcie (3, height, width) lub (batch_size, 3, height, width)

        Returns:
            Tensor YCbCr w tym samym kształcie
        """
        # Sprawdź czy wejście to pojedynczy obraz czy batch
        is_single_image = False
        if x.dim() == 3:  # Pojedynczy obraz (3, H, W)
            if x.size(0) != 3:
                raise ValueError("Pojedynczy obraz musi mieć kształt (3, height, width)")
            is_single_image = True
            x = x.unsqueeze(0)  # Dodaj wymiar batch -> (1, 3, H, W)
        elif x.dim() != 4 or x.size(1) != 3:
            raise ValueError("Wejściowy tensor musi mieć kształt (3, height, width) lub (batch_size, 3, height, width)")

        # Normalizacja wejścia do zakresu [0, 1]
        if self.in_range == '0_255':
            x = x / 255.0

        # Przekształcenie kształtu dla mnożenia macierzowego
        b, c, h, w = x.shape
        x_reshaped = x.view(b, 3, -1)  # (batch, 3, h*w)

        # Transformacja macierzowa
        ycbcr = torch.matmul(self.matrix, x_reshaped)  # (batch, 3, h*w)
        ycbcr = ycbcr.view(b, 3, h, w)

        # # Dodanie offsetu 128 dla kanałów Cb i Cr
        if self.out_range == '0_1':
            offset = self.offset / 255.0
            ycbcr = ycbcr + offset
        else:
            ycbcr = ycbcr + self.offset

        # Skalowanie wyjścia jeśli potrzebne
        if self.out_range == '0_255' and self.in_range == '0_1':
            ycbcr = ycbcr * 255.0

        # Jeśli wejście było pojedynczym obrazem, usuń wymiar batch
        if is_single_image:
            ycbcr = ycbcr.squeeze(0)

        return ycbcr


class YCbCrToRGB(nn.Module):
    """
    Warstwa PyTorch do konwersji tensora obrazu z przestrzeni YCbCr do RGB.
    Obsługuje tensory w kształcie (3, height, width) lub (batch_size, 3, height, width).

    Parametry:
        in_range (str): Zakres wartości wejściowych - '0_1' dla [0, 1] lub '0_255' dla [0, 255]
        out_range (str): Zakres wartości wyjściowych - '0_1' dla [0, 1] lub '0_255' dla [0, 255]
    """

    def __init__(self, in_range='0_1', out_range='0_1'):
        super(YCbCrToRGB, self).__init__()

        self.in_range = in_range
        self.out_range = out_range

        # Definiujemy macierz odwrotną do transformacji zgodnie ze standardem ITU-R BT.601
        self.register_buffer('matrix', torch.tensor([
            [1.0, 0.0, 1.402],  # R
            [1.0, -0.344136, -0.714136],  # G
            [1.0, 1.772, 0.0]  # B
        ]).float())

        # Offset dla Cb i Cr
        self.register_buffer('offset', torch.tensor([0, 128, 128]).view(1, 3, 1, 1).float())

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Tensor YCbCr w kształcie (3, height, width) lub (batch_size, 3, height, width)

        Returns:
            Tensor RGB w tym samym kształcie
        """
        # Sprawdź czy wejście to pojedynczy obraz czy batch
        is_single_image = False
        if x.dim() == 3:  # Pojedynczy obraz (3, H, W)
            if x.size(0) != 3:
                raise ValueError("Pojedynczy obraz musi mieć kształt (3, height, width)")
            is_single_image = True
            x = x.unsqueeze(0)  # Dodaj wymiar batch -> (1, 3, H, W)
        elif x.dim() != 4 or x.size(1) != 3:
            raise ValueError("Wejściowy tensor musi mieć kształt (3, height, width) lub (batch_size, 3, height, width)")

        # Normalizacja wejścia do zakresu [0, 1] jeśli potrzebne
        ycbcr = x.clone()
        if self.in_range == '0_255':
            ycbcr = ycbcr / 255.0

        # Odejmowanie offsetu od kanałów Cb i Cr
        if self.in_range == '0_1':
            offset = self.offset / 255.0
            ycbcr = ycbcr - offset
        else:
            ycbcr = ycbcr - self.offset

        # Przekształcenie kształtu dla mnożenia macierzowego
        b, c, h, w = ycbcr.shape
        ycbcr_reshaped = ycbcr.view(b, 3, -1)  # (batch, 3, h*w)

        # Transformacja macierzowa
        rgb = torch.matmul(self.matrix, ycbcr_reshaped)  # (batch, 3, h*w)
        rgb = rgb.view(b, 3, h, w)

        # Skalowanie wyjścia jeśli potrzebne
        if self.out_range == '0_255' and self.in_range == '0_1':
            rgb = rgb * 255.0

        # Upewnij się, że wartości są w odpowiednim zakresie
        if self.out_range == '0_1':
            rgb = torch.clamp(rgb, 0.0, 1.0)
        else:
            rgb = torch.clamp(rgb, 0.0, 255.0)

        # Jeśli wejście było pojedynczym obrazem, usuń wymiar batch
        if is_single_image:
            rgb = rgb.squeeze(0)

        return rgb