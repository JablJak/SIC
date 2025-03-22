import math

import torch
from torch import nn


class RDLoss(nn.Module):
    def __init__(self, distortion_loss: nn.Module, l: float):
        super(RDLoss, self).__init__()
        self.l = l
        self.distortion_loss = distortion_loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, y_likelihoods: torch.Tensor):
        distortion = self.distortion_loss(input, target)
        N, _, H, W = input.size()
        num_pixels = N * H * W
        bpp_loss = torch.log(y_likelihoods).sum() / (-math.log(2) * num_pixels)

        loss = self.l * distortion + bpp_loss

        return loss, bpp_loss
