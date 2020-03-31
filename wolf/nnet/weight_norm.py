__author__ = 'max'

from overrides import overrides
import torch
import torch.nn as nn


class LinearWeightNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearWeightNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.05)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
        self.linear = nn.utils.weight_norm(self.linear)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            # [batch, out_features]
            out = self(x).view(-1, self.linear.out_features)
            # [out_features]
            mean = out.mean(dim=0)
            std = out.std(dim=0)
            inv_stdv = init_scale / (std + 1e-6)

            self.linear.weight_g.mul_(inv_stdv.unsqueeze(1))
            if self.linear.bias is not None:
                self.linear.bias.add_(-mean).mul_(inv_stdv)
            return self(x)

    def forward(self, input):
        return self.linear(input)


class Conv2dWeightNorm(nn.Module):
    """
    Conv2d with weight normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dWeightNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.05)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        self.conv = nn.utils.weight_norm(self.conv)

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            # [batch, n_channels, H, W]
            out = self(x)
            n_channels = out.size(1)
            out = out.transpose(0, 1).contiguous().view(n_channels, -1)
            # [n_channels]
            mean = out.mean(dim=1)
            std = out.std(dim=1)
            inv_stdv = init_scale / (std + 1e-6)

            self.conv.weight_g.mul_(inv_stdv.view(n_channels, 1, 1, 1))
            if self.conv.bias is not None:
                self.conv.bias.add_(-mean).mul_(inv_stdv)
            return self(x)

    def forward(self, input):
        return self.conv(input)

    @overrides
    def extra_repr(self):
        return self.conv.extra_repr()


class ConvTranspose2dWeightNorm(nn.Module):
    """
    Convolution transpose 2d with weight normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super(ConvTranspose2dWeightNorm, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                         padding=padding, output_padding=output_padding, groups=groups,
                                         bias=bias, dilation=dilation)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.deconv.weight, mean=0.0, std=0.05)
        if self.deconv.bias is not None:
            nn.init.constant_(self.deconv.bias, 0)
        self.deconv = nn.utils.weight_norm(self.deconv, dim=1)

    def _output_padding(self, input, output_size):
        return self.deconv._output_padding(input, output_size)

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            # [batch, n_channels, H, W]
            out = self(x)
            n_channels = out.size(1)
            out = out.transpose(0, 1).contiguous().view(n_channels, -1)
            # [n_channels]
            mean = out.mean(dim=1)
            std = out.std(dim=1)
            inv_stdv = init_scale / (std + 1e-6)

            self.deconv.weight_g.mul_(inv_stdv.view(1, n_channels, 1, 1))
            if self.deconv.bias is not None:
                self.deconv.bias.add_(-mean).mul_(inv_stdv)
            return self(x)

    def forward(self, input):
        return self.deconv(input)

    @overrides
    def extra_repr(self):
        return self.deconv.extra_repr()
