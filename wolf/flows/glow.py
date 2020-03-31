__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple
import torch

from wolf.flows.flow import Flow
from wolf.flows.normalization import ActNorm2dFlow
from wolf.flows.permutation import Conv1x1Flow
from wolf.flows.couplings import NICE2d
from wolf.flows.multiscale_architecture import MultiScaleArchitecture


class GlowUnit(Flow):
    """
    A unit of Glow
    """
    def __init__(self, in_channels, hidden_channels=512, h_channels=0, inverse=False,
                 transform='affine', alpha=1.0, coupling_type='conv', h_type=None,
                 activation='relu', normalize=None, num_groups=None):
        super(GlowUnit, self).__init__(inverse)
        self.coupling1_up = NICE2d(in_channels, hidden_channels=hidden_channels,
                                   h_channels=h_channels, transform=transform, alpha=alpha, inverse=inverse,
                                   type=coupling_type, h_type=h_type, split_type='continuous', order='up',
                                   activation=activation, normalize=normalize, num_groups=num_groups)

        self.coupling1_dn = NICE2d(in_channels, hidden_channels=hidden_channels,
                                   h_channels=h_channels, transform=transform, alpha=alpha, inverse=inverse,
                                   type=coupling_type, h_type=h_type, split_type='continuous', order='down',
                                   activation=activation, normalize=normalize, num_groups=num_groups)

        self.actnorm = ActNorm2dFlow(in_channels, inverse=inverse)

        self.coupling2_up = NICE2d(in_channels, hidden_channels=hidden_channels,
                                   h_channels=h_channels, transform=transform, alpha=alpha, inverse=inverse,
                                   type=coupling_type, h_type=h_type, split_type='skip', order='up',
                                   activation=activation, normalize=normalize, num_groups=num_groups)

        self.coupling2_dn = NICE2d(in_channels, hidden_channels=hidden_channels,
                                   h_channels=h_channels, transform=transform, alpha=alpha, inverse=inverse,
                                   type=coupling_type, h_type=h_type, split_type='skip', order='down',
                                   activation=activation, normalize=normalize, num_groups=num_groups)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # block1, type=continuous
        out, logdet_accum = self.coupling1_up.forward(input, h=h)

        out, logdet = self.coupling1_dn.forward(out, h=h)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        out, logdet = self.actnorm.forward(out)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        # block2, type=skip
        out, logdet = self.coupling2_up.forward(out, h=h)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling2_dn.forward(out, h=h)
        logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # block2, type=skip
        out, logdet_accum = self.coupling2_dn.backward(input, h=h)

        out, logdet = self.coupling2_up.backward(out, h=h)
        logdet_accum = logdet_accum + logdet

        # ===============================================================================

        out, logdet = self.actnorm.backward(out)
        logdet_accum = logdet_accum + logdet

        # ===============================================================================

        # block1, type=continuous
        out, logdet = self.coupling1_dn.backward(out, h=h)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling1_up.backward(out, h=h)
        logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    @overrides
    def init(self, data: torch.Tensor, h=None, init_scale=1.0):
        # block1, type=continuous
        out, logdet_accum = self.coupling1_up.init(data, h=h, init_scale=init_scale)

        out, logdet = self.coupling1_dn.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        out, logdet = self.actnorm.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        # block2, type=skip
        out, logdet = self.coupling2_up.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling2_dn.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        return out, logdet_accum


class GlowStep(Flow):
    """
    A step of Glow
    """
    def __init__(self, in_channels, hidden_channels=512, h_channels=0, inverse=False,
                 transform='affine', alpha=1.0, coupling_type='conv', h_type=None,
                 activation='relu', normalize=None, num_groups=None, **kwargs):
        super(GlowStep, self).__init__(inverse)
        self.actnorm = ActNorm2dFlow(in_channels, inverse=inverse)
        self.conv1x1 = Conv1x1Flow(in_channels, inverse=inverse)
        self.unit = GlowUnit(in_channels, hidden_channels=hidden_channels, h_channels=h_channels,
                             inverse=inverse, transform=transform, alpha=alpha, coupling_type=coupling_type,
                             h_type=h_type, activation=activation, normalize=normalize, num_groups=num_groups)

    def sync(self):
        self.conv1x1.sync()

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm.forward(input)

        out, logdet = self.conv1x1.forward(out)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.unit.forward(out, h=h)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.unit.backward(input, h=h)

        out, logdet = self.conv1x1.backward(out)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.actnorm.backward(out)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm.init(data, init_scale=init_scale)

        out, logdet = self.conv1x1.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.unit.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class Glow(MultiScaleArchitecture):
    """
    Glow model in paper https://arxiv.org/pdf/1807.03039.pdf
    """

    def __init__(self, levels, num_steps, in_channels, factors, hidden_channels,
                 h_channels=0, inverse=False, transform='affine', prior_transform='affine', alpha=1.0,
                 coupling_type='conv', h_type=None, activation='relu', normalize=None, num_groups=None):
        super(Glow, self).__init__(GlowStep, levels, num_steps, in_channels, factors,
                                   hidden_channels, h_channels=h_channels, inverse=inverse,
                                   transform=transform, prior_transform=prior_transform,
                                   alpha=alpha, coupling_type=coupling_type, h_type=h_type,
                                   activation=activation, normalize=normalize, num_groups=num_groups)

    @classmethod
    def from_params(cls, params: Dict) -> "Glow":
        return Glow(**params)


Glow.register('glow')
