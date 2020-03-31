__author__ = 'max'

import torch
import torch.nn as nn


class AdaIN2d(nn.Module):
    def __init__(self, in_channels, in_features):
        super(AdaIN2d, self).__init__()
        self.norm = nn.InstanceNorm2d(in_channels, affine=False, track_running_stats=False)
        self.net = nn.Linear(in_features, 2 * in_channels)
        self.reset_parameters()

    def forward(self, x, h):
        # [batch, num_features * 2]
        h = self.net(h)
        bs, fs = h.size()
        h.view(bs, fs, 1, 1)
        # [batch, num_features, 1, 1]
        b, s = h.chunk(2, 1)
        x = self.norm(x)
        return x * (s + 1) + b

    def reset_parameters(self):
        nn.init.constant_(self.net.weight, 0.0)
        nn.init.constant_(self.net.bias, 0.0)
