import torch
import torch.nn as nn


class L1Dist(nn.Module):
    def __init__(self):
        super(L1Dist, self).__init__()

    def forward(self, x1, x2):
        return torch.abs(x1 - x2)