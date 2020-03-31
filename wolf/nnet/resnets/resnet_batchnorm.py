__author__ = 'max'

import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def deconv3x3(in_planes, out_planes, stride=1, output_padding=0):
    "3x3 deconvolution with padding"
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
                              output_padding=output_padding, bias=False)


class ResNetBlockBatchNorm(nn.Module):
    def __init__(self, inplanes, planes, stride=1, activation='relu'):
        super(ResNetBlockBatchNorm, self).__init__()
        assert activation in ['relu', 'elu', 'leaky_relu']
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True, negative_slope=1e-1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        self.downsample = downsample
        self.reset_parameters()

    def reset_parameters(self):
        # batch norm 1
        nn.init.constant_(self.bn1.weight, 1.0)
        nn.init.constant_(self.bn1.bias, 0.0)
        # batch norm 2
        nn.init.constant_(self.bn2.weight, 1.0)
        nn.init.constant_(self.bn2.bias, 0.0)
        if self.downsample is not None:
            assert isinstance(self.downsample[1], nn.BatchNorm2d)
            nn.init.constant_(self.downsample[1].weight, 1.0)
            nn.init.constant_(self.downsample[1].bias, 0.0)

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            return self(x)

    def forward(self, x):
        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        residual = x

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out += residual
        out = self.activation(out)
        return out


class DeResNetBlockBatchNorm(nn.Module):
    def __init__(self, inplanes, planes, stride=1, output_padding=0, activation='relu'):
        super(DeResNetBlockBatchNorm, self).__init__()
        assert activation in ['relu', 'elu', 'leaky_relu']
        self.deconv1 = deconv3x3(inplanes, planes, stride, output_padding)
        self.bn1 = nn.BatchNorm2d(planes)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True, negative_slope=1e-1)
        self.deconv2 = deconv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.ConvTranspose2d(inplanes, planes, kernel_size=1, stride=stride,
                                   output_padding=output_padding, bias=False),
                nn.BatchNorm2d(planes)
            )
        self.downsample = downsample
        self.reset_parameters()

    def reset_parameters(self):
        # batch norm 1
        nn.init.constant_(self.bn1.weight, 1.0)
        nn.init.constant_(self.bn1.bias, 0.0)
        # batch norm 2
        nn.init.constant_(self.bn2.weight, 1.0)
        nn.init.constant_(self.bn2.bias, 0.0)
        if self.downsample is not None:
            assert isinstance(self.downsample[1], nn.BatchNorm2d)
            nn.init.constant_(self.downsample[1].weight, 1.0)
            nn.init.constant_(self.downsample[1].bias, 0.0)

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            return self(x)

    def forward(self, x):
        # [batch, planes, h * stride - stride + 1 + op, w * stride - stride + 1 + op]
        residual = x

        # [batch, planes, h * stride - stride + 1 + op, w * stride - stride + 1 + op]
        out = self.deconv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        # [batch, planes, h * stride - stride + 1 + op, w * stride - stride + 1 + op]
        out = self.deconv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out += residual
        out = self.activation(out)
        return out
