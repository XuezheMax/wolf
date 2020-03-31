__author__ = 'max'

from collections import OrderedDict
from typing import Dict
import torch
import torch.nn as nn

from wolf.modules.encoders.encoder import Encoder
from wolf.nnet.resnets import ResNetBatchNorm, DeResNetBatchNorm, ResNetGroupNorm, DeResNetGroupNorm


class LocalResNetEncoderBatchNorm(Encoder):
    """
        Local ResNet Encoder with batch normalization
        """

    def __init__(self, levels, in_planes, out_planes, hidden_planes, activation):
        super(LocalResNetEncoderBatchNorm, self).__init__()
        layers = list()
        assert len(hidden_planes) == levels
        for level in range(levels):
            hidden_channels = hidden_planes[level]
            layers.append(('resnet{}'.format(level),
                           ResNetBatchNorm(in_planes,
                                           [hidden_channels, hidden_channels],
                                           [1, 2], activation)))
            in_planes = hidden_channels

        in_planes = hidden_planes[-1]
        hidden_planes = [out_planes, ] + hidden_planes
        for level in reversed(range(levels)):
            hidden_channels = hidden_planes[level]
            layers.append(('deresnet{}'.format(level),
                           DeResNetBatchNorm(in_planes,
                                             [in_planes, hidden_channels],
                                             [1, 2], [0, 1], activation)))
            in_planes = hidden_channels

        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # [batch, out_planes, h, w]
        return self.net(x)

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            return self(x)

    @classmethod
    def from_params(cls, params: Dict) -> "LocalResNetEncoderBatchNorm":
        return LocalResNetEncoderBatchNorm(**params)


class LocalResNetEncoderGroupNorm(Encoder):
    """
        Local ResNet Encoder with batch normalization
        """

    def __init__(self, levels, in_planes, out_planes, hidden_planes, activation, num_groups):
        super(LocalResNetEncoderGroupNorm, self).__init__()
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

        in_planes = hidden_planes[-1]
        hidden_planes = [out_planes, ] + hidden_planes
        for level in reversed(range(levels)):
            hidden_channels = hidden_planes[level]
            n_groups = num_groups[level]
            layers.append(('deresnet{}'.format(level),
                           DeResNetGroupNorm(in_planes,
                                             [in_planes, hidden_channels],
                                             [1, 2], [0, 0], activation, num_groups=n_groups)))
            in_planes = hidden_channels

        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # [batch, out_planes, h, w]
        return self.net(x)

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            return self(x)

    @classmethod
    def from_params(cls, params: Dict) -> "LocalResNetEncoderGroupNorm":
        return LocalResNetEncoderGroupNorm(**params)


LocalResNetEncoderBatchNorm.register('local_resnet_bn')
LocalResNetEncoderGroupNorm.register('local_resnet_gn')
