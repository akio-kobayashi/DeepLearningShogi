import torch
import torch.nn as nn
import torch.nn.functional as F

from dlshogi.common import *
from dlshogi.network.common import Bias

k = 192
fcl = 256 # fully connected layers
class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=k, kernel_size=1, padding=0, bias=False)
        self.l1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=k, kernel_size=1, bias=False) # pieces_in_hand
        self.l2 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l3 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l4 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l5 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l6 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l7 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l8 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l9 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l10 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l11 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l12 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l13 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l14 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l15 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l16 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l17 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l18 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l19 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l20 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        self.l21 = nn.Conv2d(in_channels=k, out_channels=k, kernel_size=3, padding=1, bias=False)
        # policy network
        self.l22 = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.l22_2 = Bias(9*9*MAX_MOVE_LABEL_NUM)
        # value network
        self.l22_v = nn.Conv2d(in_channels=k, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1)
        self.l23_v = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.l24_v = nn.Linear(fcl, 1)
        self.norm1 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm2 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm3 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm4 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm5 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm6 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm7 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm8 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm9 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm10 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm11 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm12 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm13 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm14 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm15 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm16 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm17 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm18 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm19 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm20 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm21 = nn.BatchNorm2d(k, eps=2e-05)
        self.norm22_v = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM, eps=2e-05)

    def forward(self, x1, x2):
        u1_1_1 = self.l1_1_1(x1)
        u1_1_2 = self.l1_1_2(x1)
        u1_2 = self.l1_2(x2)
        u = u1_1_1 + u1_1_2 + u1_2

        residual_layers = (
            (self.norm1, self.norm2, self.l2, self.l3),
            (self.norm3, self.norm4, self.l4, self.l5),
            (self.norm5, self.norm6, self.l6, self.l7),
            (self.norm7, self.norm8, self.l8, self.l9),
            (self.norm9, self.norm10, self.l10, self.l11),
            (self.norm11, self.norm12, self.l12, self.l13),
            (self.norm13, self.norm14, self.l14, self.l15),
            (self.norm15, self.norm16, self.l16, self.l17),
            (self.norm17, self.norm18, self.l18, self.l19),
            (self.norm19, self.norm20, self.l20, self.l21),
        )
        for norm_a, norm_b, conv_a, conv_b in residual_layers:
            h = F.relu(norm_a(u))
            h = F.relu(norm_b(conv_a(h)))
            u = conv_b(h) + u

        h21 = F.relu(self.norm21(u))
        # policy network
        h22 = self.l22(h21)
        h22_1 = self.l22_2(torch.flatten(h22, 1))
        # value network
        h22_v = F.relu(self.norm22_v(self.l22_v(h21)))
        h23_v = F.relu(self.l23_v(torch.flatten(h22_v, 1)))
        return h22_1, self.l24_v(h23_v)
