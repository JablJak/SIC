import torch
from torch import nn, Tensor


class LearnableTempSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.randn(1))

    def forward(self, input: Tensor) -> Tensor:
        return torch.sigmoid(input/self.temperature)