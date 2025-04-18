import torch
from torch import nn, Tensor


class LearnableTempSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, input: Tensor) -> Tensor:
        temp = torch.clamp(self.temperature, min=1e-6)
        return torch.sigmoid(input/temp)