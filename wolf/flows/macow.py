__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn

from wolf.flows.flow import Flow
from wolf.flows.normalization import ActNorm2dFlow
from wolf.flows.permutation import Conv1x1Flow
from wolf.flows.couplings import NICE2d, MaskedConvFlow
from wolf.flows.multiscale_architecture import MultiScaleArchitecture


class MaCowUnit(Flow):
    """
    A Unit of Flows with an MCF(A), MCF(B), an Conv1x1, followd by an ActNorm and an activation.
    """

    def __init__(self, in_channels, kernel_size, h_channels=0, inverse=False,
                 transform='affine', alpha=1.0, h_type=None, activation='relu'):
        super(MaCowUnit, self).__init__(inverse)
        self.conv1 = MaskedConvFlow(in_channels, (kernel_size[0], kernel_size[1]), order='A',
                                    h_channels=h_channels, transform=transform, alpha=alpha,
                                    h_type=h_type, activation=activation)
        self.conv2 = MaskedConvFlow(in_channels, (kernel_size[0], kernel_size[1]), order='B',
                                    h_channels=h_channels, transform=transform, alpha=alpha,
                                    h_type=h_type, activation=activation)
        self.actnorm1 = ActNorm2dFlow(in_channels, inverse=inverse)

        self.conv3 = MaskedConvFlow(in_channels, (kernel_size[1], kernel_size[0]), order='C',
                                    h_channels=h_channels, transform=transform, alpha=alpha,
                                    h_type=h_type, activation=activation)
        self.conv4 = MaskedConvFlow(in_channels, (kernel_size[1], kernel_size[0]), order='D',
                                    h_channels=h_channels, transform=transform, alpha=alpha,
                                    h_type=h_type, activation=activation)
        self.actnorm2 = ActNorm2dFlow(in_channels, inverse=inverse)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # MCF1
        out, logdet_accum = self.conv1.forward(input, h=h)
        # MCF2
        out, logdet = self.conv2.forward(out, h=h)
        logdet_accum = logdet_accum + logdet
        # ActNorm1
        out, logdet = self.actnorm1.forward(out)
        logdet_accum = logdet_accum + logdet
        # MCF3
        out, logdet = self.conv3.forward(out, h=h)
        logdet_accum = logdet_accum + logdet
        # MCF4
        out, logdet = self.conv4.forward(out, h=h)
        logdet_accum = logdet_accum + logdet
        # ActNorm2
        out, logdet = self.actnorm2.forward(out)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # ActNorm2
        out, logdet_accum = self.actnorm2.backward(input)
        # MCF4
        out, logdet = self.conv4.backward(out, h=h)
        logdet_accum = logdet_accum + logdet
        # MCF3
        out, logdet = self.conv3.backward(out, h=h)
        logdet_accum = logdet_accum + logdet
        # ActNorm1
        out, logdet = self.actnorm1.backward(out)
        logdet_accum = logdet_accum + logdet
        # MCF2
        out, logdet = self.conv2.backward(out, h=h)
        logdet_accum = logdet_accum + logdet
        # MCF1
        out, logdet = self.conv1.backward(out, h=h)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # MCF1
        out, logdet_accum = self.conv1.init(data, h=h, init_scale=init_scale)
        # MCF2
        out, logdet = self.conv2.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        # ActNorm1
        out, logdet = self.actnorm1.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        # MCF3
        out, logdet = self.conv3.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        # MCF4
        out, logdet = self.conv4.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        # ActNorm2
        out, logdet = self.actnorm2.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class MaCowStep(Flow):
    """
        A step of Macow Flows
        """

    def __init__(self, in_channels, kernel_size, hidden_channels, h_channels, inverse=False,
                 transform='affine', alpha=1.0, coupling_type='conv', h_type=None,
                 activation='relu', normalize=None, num_groups=None, **kwargs):
        super(MaCowStep, self).__init__(inverse)
        num_units = 2
        self.actnorm1 = ActNorm2dFlow(in_channels, inverse=inverse)
        self.conv1x1 = Conv1x1Flow(in_channels, inverse=inverse)
        units = [MaCowUnit(in_channels, kernel_size, h_channels=h_channels, transform=transform,
                           alpha=alpha, inverse=inverse, h_type=h_type, activation=activation)
                 for _ in range(num_units)]
        self.units1 = nn.ModuleList(units)
        self.coupling1_up = NICE2d(in_channels, hidden_channels=hidden_channels,
                                   h_channels=h_channels, transform=transform, alpha=alpha, inverse=inverse,
                                   type=coupling_type, h_type=h_type, split_type='continuous', order='up',
                                   activation=activation, normalize=normalize, num_groups=num_groups)
        self.coupling1_dn = NICE2d(in_channels, hidden_channels=hidden_channels,
                                   h_channels=h_channels, transform=transform, alpha=alpha, inverse=inverse,
                                   type=coupling_type, h_type=h_type, split_type='continuous', order='down',
                                   activation=activation, normalize=normalize, num_groups=num_groups)

        self.actnorm2 = ActNorm2dFlow(in_channels, inverse=inverse)

        units = [MaCowUnit(in_channels, kernel_size, h_channels=h_channels, transform=transform,
                           alpha=alpha, inverse=inverse, h_type=h_type, activation=activation)
                 for _ in range(num_units)]
        self.units2 = nn.ModuleList(units)
        self.coupling2_up = NICE2d(in_channels, hidden_channels=hidden_channels,
                                   h_channels=h_channels, transform=transform, alpha=alpha, inverse=inverse,
                                   type=coupling_type, h_type=h_type, split_type='skip', order='up',
                                   activation=activation, normalize=normalize, num_groups=num_groups)
        self.coupling2_dn = NICE2d(in_channels, hidden_channels=hidden_channels,
                                   h_channels=h_channels, transform=transform, alpha=alpha, inverse=inverse,
                                   type=coupling_type, h_type=h_type, split_type='skip', order='down',
                                   activation=activation, normalize=normalize, num_groups=num_groups)

    def sync(self):
        self.conv1x1.sync()

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # part1
        out, logdet_accum = self.actnorm1.forward(input)
        out, logdet = self.conv1x1.forward(out)
        logdet_accum = logdet_accum + logdet
        for unit in self.units1:
            out, logdet = unit.forward(out, h=h)
            logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling1_up.forward(out, h=h)
        logdet_accum = logdet_accum + logdet
        out, logdet = self.coupling1_dn.forward(out, h=h)
        logdet_accum = logdet_accum + logdet
        # part 2
        out, logdet = self.actnorm2.forward(out)
        logdet_accum = logdet_accum + logdet
        for unit in self.units2:
            out, logdet = unit.forward(out, h=h)
            logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling2_up.forward(out, h=h)
        logdet_accum = logdet_accum + logdet
        out, logdet = self.coupling2_dn.forward(out, h=h)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # part 2
        out, logdet_accum = self.coupling2_dn.backward(input, h=h)
        out, logdet = self.coupling2_up.backward(out, h=h)
        logdet_accum = logdet_accum + logdet

        for unit in reversed(self.units2):
            out, logdet = unit.backward(out, h=h)
            logdet_accum = logdet_accum + logdet

        out, logdet = self.actnorm2.backward(out)
        logdet_accum = logdet_accum + logdet
        # part1
        out, logdet = self.coupling1_dn.backward(out, h=h)
        logdet_accum = logdet_accum + logdet
        out, logdet = self.coupling1_up.backward(out, h=h)
        logdet_accum = logdet_accum + logdet

        for unit in reversed(self.units1):
            out, logdet = unit.backward(out, h=h)
            logdet_accum = logdet_accum + logdet

        out, logdet = self.conv1x1.backward(out)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.actnorm1.backward(out)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm1.init(data, init_scale=init_scale)
        out, logdet = self.conv1x1.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        for unit in self.units1:
            out, logdet = unit.init(out, h=h, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling1_up.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        out, logdet = self.coupling1_dn.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        # part 2
        out, logdet = self.actnorm2.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        for unit in self.units2:
            out, logdet = unit.init(out, h=h, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling2_up.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        out, logdet = self.coupling2_dn.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class MaCow(MultiScaleArchitecture):
    """
    MaCow model in paper https://arxiv.org/pdf/1902.04208.pdf
    """

    def __init__(self, levels, num_steps, in_channels, factors, hidden_channels,
                 h_channels=0, inverse=False, transform='affine', prior_transform='affine',
                 alpha=1.0, kernel_size=(2, 3), coupling_type='conv', h_type=None,
                 activation='relu', normalize=None, num_groups=None):
        assert len(kernel_size) == 2, 'kernel size should contain two numbers'
        super(MaCow, self).__init__(MaCowStep, levels, num_steps, in_channels, factors,
                                    hidden_channels, h_channels=h_channels, inverse=inverse,
                                    transform=transform, prior_transform=prior_transform, alpha=alpha,
                                    kernel_size=kernel_size, coupling_type=coupling_type, h_type=h_type,
                                    activation=activation, normalize=normalize, num_groups=num_groups)

    @classmethod
    def from_params(cls, params: Dict) -> "MaCow":
        return MaCow(**params)


MaCow.register('macow')
