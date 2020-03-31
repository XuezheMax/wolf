__author__ = 'max'

import torch.nn as nn
from wolf.nnet.weight_norm import Conv2dWeightNorm, ConvTranspose2dWeightNorm


def conv3x3_weightnorm(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return Conv2dWeightNorm(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def deconv3x3_weightnorm(in_planes, out_planes, stride=1, output_padding=0):
    "3x3 deconvolution with padding"
    return ConvTranspose2dWeightNorm(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                                     output_padding=output_padding, bias=True)


class ResNetBlockWeightNorm(nn.Module):
    def __init__(self, inplanes, planes, stride=1, activation='relu'):
        super(ResNetBlockWeightNorm, self).__init__()
        assert activation in ['relu', 'elu', 'leaky_relu']
        self.conv1 = conv3x3_weightnorm(inplanes, planes, stride)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True, negative_slope=1e-1)
        self.conv2 = conv3x3_weightnorm(planes, planes)
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = Conv2dWeightNorm(inplanes, planes, kernel_size=1, stride=stride, bias=True)
        self.downsample = downsample

    def init(self, x, init_scale=1.0):
        residual = x

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out = self.conv1.init(x, init_scale=init_scale)
        out = self.activation(out)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out = self.conv2.init(out, init_scale=init_scale)

        if self.downsample is not None:
            residual = self.downsample.init(x, init_scale=init_scale)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out += residual
        out = self.activation(out)
        return out

    def forward(self, x):
        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        residual = x

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out = self.conv1(x)
        out = self.activation(out)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out += residual
        out = self.activation(out)
        return out


class DeResNetBlockWeightNorm(nn.Module):
    def __init__(self, inplanes, planes, stride=1, output_padding=0, activation='relu'):
        super(DeResNetBlockWeightNorm, self).__init__()
        assert activation in ['relu', 'elu', 'leaky_relu']
        self.deconv1 = deconv3x3_weightnorm(inplanes, planes, stride, output_padding)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True, negative_slope=1e-1)
        self.deconv2 = deconv3x3_weightnorm(planes, planes)
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = ConvTranspose2dWeightNorm(inplanes, planes, kernel_size=1, stride=stride,
                                                   output_padding=output_padding, bias=True)
        self.downsample = downsample

    def init(self, x, init_scale=1.0):
        # [batch, planes, h * stride - stride + 1 + op, w * stride - stride + 1 + op]
        residual = x

        # [batch, planes, h * stride - stride + 1 + op, w * stride - stride + 1 + op]
        out = self.deconv1.init(x, init_scale=init_scale)
        out = self.activation(out)

        # [batch, planes, h * stride - stride + 1 + op, w * stride - stride + 1 + op]
        out = self.deconv2.init(out, init_scale=init_scale)

        if self.downsample is not None:
            residual = self.downsample.init(x, init_scale=init_scale)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out += residual
        out = self.activation(out)
        return out

    def forward(self, x):
        # [batch, planes, h * stride - stride + 1 + op, w * stride - stride + 1 + op]
        residual = x

        # [batch, planes, h * stride - stride + 1 + op, w * stride - stride + 1 + op]
        out = self.deconv1(x)
        out = self.activation(out)

        # [batch, planes, h * stride - stride + 1 + op, w * stride - stride + 1 + op]
        out = self.deconv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out += residual
        out = self.activation(out)
        return out
