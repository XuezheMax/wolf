__author__ = 'max'

import torch
import torch.nn as nn
import torch.nn.functional as F

from wolf.nnet.weight_norm import Conv2dWeightNorm, LinearWeightNorm
from wolf.nnet.shift_conv import ShiftedConv2d


class NICEMLPBlock(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, activation):
        super(NICEMLPBlock, self).__init__()
        assert activation in ['relu', 'elu', 'leaky_relu']
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.fc2 = nn.Linear(hidden_features, hidden_features, bias=True)
        self.fc3 = LinearWeightNorm(hidden_features, out_features, bias=True)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True, negative_slope=1e-1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        # fc1
        out = self.activation(self.fc1(x))
        # fc2
        out = self.activation(self.fc2(out))
        # fc3
        out = self.fc3(out)
        return out

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            # init fc1
            out = self.activation(self.fc1(x))
            # init fc2
            out = self.activation(self.fc2(out))
            # fc3
            out = self.fc3.init(out, init_scale=0.0 * init_scale)
            return out


class NICEConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, activation, normalize=None, num_groups=None):
        super(NICEConvBlock, self).__init__()
        assert activation in ['relu', 'elu', 'leaky_relu']
        assert normalize in [None, 'batch_norm', 'group_norm', 'instance_norm']
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False)
        self.conv3 = Conv2dWeightNorm(hidden_channels, out_channels, kernel_size=3, padding=1, bias=True)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True, negative_slope=1e-1)

        if normalize is None:
            self.norm1 = None
            self.norm2 = None
        elif normalize == 'batch_norm':
            self.norm1 = nn.BatchNorm2d(hidden_channels)
            self.norm2 = nn.BatchNorm2d(hidden_channels)
        elif normalize == 'instance_norm':
            self.norm1 = nn.InstanceNorm2d(hidden_channels, affine=True)
            self.norm2 = nn.InstanceNorm2d(hidden_channels, affine=True)
        else:
            self.norm1 = nn.GroupNorm(num_groups, hidden_channels, affine=True)
            self.norm2 = nn.GroupNorm(num_groups, hidden_channels, affine=True)
        self.reset_parameters()

    def reset_parameters(self):
        if self.norm1 is not None and self.norm1.affine:
            # norm 1
            nn.init.constant_(self.norm1.weight, 1.0)
            nn.init.constant_(self.norm1.bias, 0.0)
            # norm 2
            nn.init.constant_(self.norm2.weight, 1.0)
            nn.init.constant_(self.norm2.bias, 0.0)

    def forward(self, x, h=None):
        out = self.conv1(x)
        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.activation(out)
        # conv2
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)
        if h is not None:
            out = out + h
        out = self.activation(out)
        # conv3
        out = self.conv3(out)
        return out

    def init(self, x, h=None, init_scale=1.0):
        with torch.no_grad():
            out = self.conv1(x)
            if self.norm1 is not None:
                out = self.norm1(out)
            out = self.activation(out)
            # init conv2
            out = self.conv2(out)
            if self.norm2 is not None:
                out = self.norm2(out)
            if h is not None:
                out = out + h
            out = self.activation(out)
            # init conv3
            out = self.conv3.init(out, init_scale=0.0 * init_scale)
            return out


class MCFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, hidden_channels, order, activation):
        super(MCFBlock, self).__init__()
        self.shift_conv = ShiftedConv2d(in_channels, hidden_channels, kernel_size, order=order, bias=False)
        assert activation in ['relu', 'elu', 'leaky_relu']

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True, negative_slope=1e-1)
        self.conv1x1 = Conv2dWeightNorm(hidden_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x, h=None, shifted=True):
        c = self.shift_conv(x, shifted=shifted)
        if h is not None:
            c = c + h
        c = self.activation(c)
        c = self.conv1x1(c)
        return c

    def init(self, x, h=None, init_scale=1.0):
        with torch.no_grad():
            c = self.shift_conv(x)
            if h is not None:
                c = c + h
            c = self.activation(c)
            c = self.conv1x1.init(c, init_scale=0.0 * init_scale)
            return c


class LocalLinearCondNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(LocalLinearCondNet, self).__init__()
        padding = kernel_size // 2
        self.net = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, h, x=None):
        return self.net(h)


class GlobalLinearCondNet(nn.Module):
    def __init__(self, in_features, out_features):
        super(GlobalLinearCondNet, self).__init__()
        self.net = nn.Linear(in_features, out_features)

    def forward(self, h, x=None):
        out = self.net(h)
        bs, fs = out.size()
        return out.view(bs, fs, 1, 1)


class GlobalAttnCondNet(nn.Module):
    def __init__(self, q_dim, k_dim, out_dim):
        super(GlobalAttnCondNet, self).__init__()
        self.query_proj = nn.Linear(q_dim, out_dim, bias=True)
        self.key_proj = nn.Conv2d(k_dim, out_dim, kernel_size=1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        # key proj
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.constant_(self.key_proj.bias, 0)
        # query proj
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.constant_(self.query_proj.bias, 0)

    def forward(self, h, x):
        # [batch, out_dim]
        h = self.query_proj(h)
        # [batch, out_dim, height, width]
        key = self.key_proj(x)
        bs, dim, height, width = key.size()
        # [batch, height, width]
        attn_weights = torch.einsum('bd,bdhw->bhw', h, key)
        attn_weights = F.softmax(attn_weights.view(bs, -1), dim=-1).view(bs, height, width)
        # [batch, out_dim, height, width]
        out = h.view(bs, dim, 1, 1) * attn_weights.unsqueeze(1)
        return out
