import math
from typing import Mapping, Any

import torch
from torch import nn, Tensor


class RDLoss(nn.Module):
    def __init__(self, distortion_loss: nn.Module, l: float):
        super(RDLoss, self).__init__()
        self.l = l
        self.distortion_loss = distortion_loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, likelihoods: dict[str, Tensor], optimize_bpp = True) -> torch.Tensor:
        distortion = self.distortion_loss(input, target)
        N, _, H, W = input.size()
        num_pixels = N * H * W

        if optimize_bpp:
            total_log_likelihood = 0
            for k, lh_tensor in likelihoods.items():
                lh_tensor_clamped = lh_tensor.clamp(min=1e-9)
                total_log_likelihood += torch.log(lh_tensor_clamped).sum()
            total_bits = total_log_likelihood / (-math.log(2))
            bpp_loss = total_bits / num_pixels
        else:
            bpp_loss = torch.tensor(0,device=input.device, dtype=input.dtype)
        loss = distortion + self.l * (bpp_loss if optimize_bpp else 0)

        return loss, bpp_loss
