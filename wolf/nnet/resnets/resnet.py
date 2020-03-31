__author__ = 'max'

import torch.nn as nn

from wolf.nnet.resnets.resnet_batchnorm import ResNetBlockBatchNorm, DeResNetBlockBatchNorm
from wolf.nnet.resnets.resnet_weightnorm import ResNetBlockWeightNorm, DeResNetBlockWeightNorm
from wolf.nnet.resnets.resnet_groupnorm import ResNetBlockGroupNorm, DeResNetBlockGroupNorm

__all__ = ['ResNetBatchNorm', 'ResNetGroupNorm', 'ResNetWeightNorm',
           'DeResNetBatchNorm', 'DeResNetGroupNorm', 'DeResNetWeightNorm']


class _ResNet(nn.Module):
    def __init__(self, resnet_block, inplanes, planes, strides, activation, **kwargs):
        super(_ResNet, self).__init__()
        assert len(planes) == len(strides)

        blocks = []
        for i in range(len(planes)):
            plane = planes[i]
            stride = strides[i]
            block = resnet_block(inplanes, plane, stride=stride, activation=activation, **kwargs)
            blocks.append(block)
            inplanes = plane

        self.main = nn.Sequential(*blocks)

    def init(self, x, init_scale=1.0):
        for block in self.main:
            x = block.init(x, init_scale=init_scale)
        return x

    def forward(self, x):
        return self.main(x)


class _DeResNet(nn.Module):
    def __init__(self, deresnet_block, inplanes, planes, strides, output_paddings, activation, **kwargs):
        super(_DeResNet, self).__init__()
        assert len(planes) == len(strides)
        assert len(planes) == len(output_paddings)

        blocks = []
        for i in range(len(planes)):
            plane = planes[i]
            stride = strides[i]
            output_padding = output_paddings[i]
            block = deresnet_block(inplanes, plane, stride=stride, output_padding=output_padding,
                                   activation=activation, **kwargs)
            blocks.append(block)
            inplanes = plane

        self.main = nn.Sequential(*blocks)

    def init(self, x, init_scale=1.0):
        for block in self.main:
            x = block.init(x, init_scale=init_scale)
        return x

    def forward(self, x):
        return self.main(x)


class ResNetBatchNorm(_ResNet):
    def __init__(self, inplanes, planes, strides, activation):
        super(ResNetBatchNorm, self).__init__(ResNetBlockBatchNorm, inplanes, planes, strides, activation)


class ResNetWeightNorm(_ResNet):
    def __init__(self, inplanes, planes, strides, activation):
        super(ResNetWeightNorm, self).__init__(ResNetBlockWeightNorm, inplanes, planes, strides, activation)


class ResNetGroupNorm(_ResNet):
    def __init__(self, inplanes, planes, strides, activation, num_groups):
        super(ResNetGroupNorm, self).__init__(ResNetBlockGroupNorm, inplanes, planes, strides, activation,
                                              num_groups=num_groups)


class DeResNetBatchNorm(_DeResNet):
    def __init__(self, inplanes, planes, strides, output_paddings, activation):
        super(DeResNetBatchNorm, self).__init__(DeResNetBlockBatchNorm, inplanes, planes, strides,
                                                output_paddings, activation)


class DeResNetWeightNorm(_DeResNet):
    def __init__(self, inplanes, planes, strides, output_paddings, activation):
        super(DeResNetWeightNorm, self).__init__(DeResNetBlockWeightNorm, inplanes, planes, strides,
                                                 output_paddings, activation)


class DeResNetGroupNorm(_DeResNet):
    def __init__(self, inplanes, planes, strides, output_paddings, activation, num_groups):
        super(DeResNetGroupNorm, self).__init__(DeResNetBlockGroupNorm, inplanes, planes, strides,
                                                output_paddings, activation, num_groups=num_groups)
