import torch
import torch.nn as nn


class Bias(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, x):
        return x + self.bias


class Swish(nn.Module):
    """Ordinary (non memory-efficient) Swish used for export compatibility."""

    def forward(self, x):
        return x * torch.sigmoid(x)
