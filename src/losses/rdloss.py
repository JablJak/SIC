import math

import torch
from torch import nn


class RDLoss(nn.Module):
    def __init__(self, distortion_loss: nn.Module, l: float):
        super(RDLoss, self).__init__()
        self.l = l
        self.distortion_loss = distortion_loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, y_likelihoods: torch.Tensor, optimize_bpp = True) -> torch.Tensor:
        distortion = self.distortion_loss(input, target)
        N, _, H, W = input.size()
        num_pixels = N * H * W
        if optimize_bpp:
            bpp_loss = torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels)
        else:
            bpp_loss = torch.tensor(0,device=input.device, dtype=input.dtype)
        loss = distortion +  self.l * (bpp_loss if optimize_bpp else 0)

        return loss, bpp_loss
