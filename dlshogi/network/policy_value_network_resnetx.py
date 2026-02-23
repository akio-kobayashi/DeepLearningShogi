import torch
import torch.nn as nn

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM, MAX_MOVE_LABEL_NUM
from dlshogi.network.common import Bias, Swish


class ResNetXBlock(nn.Module):
    def __init__(self, channels: int, activation: nn.Module):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + residual)


class PolicyValueNetwork(nn.Module):
    """Modernized, configurable ResNet implementation for parallel introduction.

    This implementation does not replace legacy model definitions and is selected
    explicitly via network names such as:
      - resnetx10
      - resnetx20x256_fcl384_swish
    """

    def __init__(
        self,
        blocks: int,
        channels: int,
        activation: nn.Module | None = None,
        fcl: int = 256,
    ):
        super().__init__()
        if activation is None:
            activation = nn.ReLU()

        self.stem_conv3 = nn.Conv2d(
            in_channels=FEATURES1_NUM,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.stem_conv1 = nn.Conv2d(
            in_channels=FEATURES1_NUM,
            out_channels=channels,
            kernel_size=1,
            padding=0,
            bias=False,
        )
        self.stem_hand = nn.Conv2d(
            in_channels=FEATURES2_NUM,
            out_channels=channels,
            kernel_size=1,
            bias=False,
        )
        self.stem_bn = nn.BatchNorm2d(channels)
        self.act = activation

        self.blocks = nn.Sequential(
            *[ResNetXBlock(channels, activation) for _ in range(blocks)]
        )

        self.policy_head = nn.Conv2d(
            in_channels=channels,
            out_channels=MAX_MOVE_LABEL_NUM,
            kernel_size=1,
            bias=False,
        )
        self.policy_bias = Bias(9 * 9 * MAX_MOVE_LABEL_NUM)

        self.value_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=MAX_MOVE_LABEL_NUM,
            kernel_size=1,
            bias=False,
        )
        self.value_bn = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.value_fc1 = nn.Linear(9 * 9 * MAX_MOVE_LABEL_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x = self.stem_conv3(x1) + self.stem_conv1(x1) + self.stem_hand(x2)
        x = self.act(self.stem_bn(x))
        x = self.blocks(x)

        policy = self.policy_head(x)
        policy = self.policy_bias(torch.flatten(policy, 1))

        value = self.act(self.value_bn(self.value_conv(x)))
        value = self.act(self.value_fc1(torch.flatten(value, 1)))
        value = self.value_fc2(value)
        return policy, value

    def set_swish(self, memory_efficient: bool = True) -> None:
        activation = nn.SiLU() if memory_efficient else Swish()
        for _, m in self.named_modules():
            if isinstance(m, (PolicyValueNetwork, ResNetXBlock)):
                m.act = activation

