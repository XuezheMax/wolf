__author__ = 'max'

from collections import OrderedDict
from typing import Dict
import torch
import torch.nn as nn

from wolf.modules.encoders.encoder import Encoder
from wolf.nnet.resnets import ResNetBatchNorm, ResNetGroupNorm


class GlobalResNetEncoderBatchNorm(Encoder):
    """
    Global ResNet Encoder with batch normalization
    """
    def __init__(self, levels, in_planes, out_planes, hidden_planes, activation):
        super(GlobalResNetEncoderBatchNorm, self).__init__()
        layers = list()
        assert len(hidden_planes) == levels
        for level in range(levels):
            hidden_channels = hidden_planes[level]
            layers.append(('resnet{}'.format(level),
                           ResNetBatchNorm(in_planes,
                                           [hidden_channels, hidden_channels],
                                           [1, 2], activation)))
            in_planes = hidden_channels

        layers.append(('top', nn.Conv2d(in_planes, out_planes, 1, bias=True)))
        layers.append(('activate', nn.ELU(inplace=True)))
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # [batch, out_planes, h, w]
        out = self.net(x)
        # [batch, out_planes * h * w]
        return out.view(out.size(0), -1)

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            return self(x)

    @classmethod
    def from_params(cls, params: Dict) -> "GlobalResNetEncoderBatchNorm":
        return GlobalResNetEncoderBatchNorm(**params)


class GlobalResNetEncoderGroupNorm(Encoder):
    """
    Global ResNet Encoder with batch normalization
    """
    def __init__(self, levels, in_planes, out_planes, hidden_planes, activation, num_groups):
        super(GlobalResNetEncoderGroupNorm, self).__init__()
        layers = list()
        assert len(hidden_planes) == levels
        assert len(num_groups) == levels
        for level in range(levels):
            hidden_channels = hidden_planes[level]
            n_groups = num_groups[level]
            layers.append(('resnet{}'.format(level),
                           ResNetGroupNorm(in_planes,
                                           [hidden_channels, hidden_channels],
                                           [1, 2], activation, num_groups=n_groups)))
            in_planes = hidden_channels

        layers.append(('top', nn.Conv2d(in_planes, out_planes, 1, bias=True)))
        layers.append(('activate', nn.ELU(inplace=True)))
        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # [batch, out_planes, h, w]
        out = self.net(x)
        # [batch, out_planes * h * w]
        return out.view(out.size(0), -1)

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            return self(x)

    @classmethod
    def from_params(cls, params: Dict) -> "GlobalResNetEncoderGroupNorm":
        return GlobalResNetEncoderGroupNorm(**params)


GlobalResNetEncoderBatchNorm.register('global_resnet_bn')
GlobalResNetEncoderGroupNorm.register('global_resnet_gn')
