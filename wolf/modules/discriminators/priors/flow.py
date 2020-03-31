__author__ = 'max'

from overrides import overrides
import math
from typing import Dict, Tuple
import torch
import torch.nn as nn

from wolf.flows.flow import Flow
from wolf.flows.normalization import ActNorm1dFlow
from wolf.flows.permutation import InvertibleLinearFlow
from wolf.flows.couplings import NICE1d
from wolf.modules.discriminators.priors.prior import Prior


class PriorFlowUnit(Flow):
    """
    A unit of Prior Flow
    """
    def __init__(self, in_features, hidden_features=512, inverse=False,
                 transform='affine', alpha=1.0, coupling_type='mlp', activation='elu'):
        super(PriorFlowUnit, self).__init__(inverse)
        self.coupling1_up = NICE1d(in_features, hidden_features=hidden_features, transform=transform, alpha=alpha,
                                   inverse=inverse, type=coupling_type, split_type='continuous', order='up', activation=activation)

        self.coupling1_dn = NICE1d(in_features, hidden_features=hidden_features, transform=transform, alpha=alpha,
                                   inverse=inverse, type=coupling_type, split_type='continuous', order='down', activation=activation)

        self.actnorm = ActNorm1dFlow(in_features, inverse=inverse)

        self.coupling2_up = NICE1d(in_features, hidden_features=hidden_features, transform=transform, alpha=alpha,
                                   inverse=inverse, type=coupling_type, split_type='skip', order='up', activation=activation)

        self.coupling2_dn = NICE1d(in_features, hidden_features=hidden_features, transform=transform, alpha=alpha,
                                   inverse=inverse, type=coupling_type, split_type='skip', order='down', activation=activation)

    @overrides
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # block1, type=continuous
        out, logdet_accum = self.coupling1_up.forward(input)

        out, logdet = self.coupling1_dn.forward(out)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        out, logdet = self.actnorm.forward(out)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        # block2, type=skip
        out, logdet = self.coupling2_up.forward(out)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling2_dn.forward(out)
        logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # block2, type=skip
        out, logdet_accum = self.coupling2_dn.backward(input)

        out, logdet = self.coupling2_up.backward(out)
        logdet_accum = logdet_accum + logdet

        # ===============================================================================

        out, logdet = self.actnorm.backward(out)
        logdet_accum = logdet_accum + logdet

        # ===============================================================================

        # block1, type=continuous
        out, logdet = self.coupling1_dn.backward(out)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling1_up.backward(out)
        logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    @overrides
    def init(self, data: torch.Tensor, init_scale=1.0):
        # block1, type=continuous
        out, logdet_accum = self.coupling1_up.init(data, init_scale=init_scale)

        out, logdet = self.coupling1_dn.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        out, logdet = self.actnorm.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        # ================================================================================

        # block2, type=skip
        out, logdet = self.coupling2_up.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.coupling2_dn.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        return out, logdet_accum


class PriorFlowStep(Flow):
    """
    A step of Prior Flow
    """
    def __init__(self, in_features, hidden_features=512, inverse=False,
                 transform='affine', alpha=1.0, coupling_type='mlp', activation='elu', **kwargs):
        super(PriorFlowStep, self).__init__(inverse)
        self.actnorm = ActNorm1dFlow(in_features, inverse=inverse)
        self.linear = InvertibleLinearFlow(in_features, inverse=inverse)
        self.unit = PriorFlowUnit(in_features, hidden_features=hidden_features, inverse=inverse,
                                  transform=transform, alpha=alpha, coupling_type=coupling_type, activation=activation)

    def sync(self):
        self.linear.sync()

    @overrides
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm.forward(input)

        out, logdet = self.linear.forward(out)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.unit.forward(out)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.unit.backward(input)

        out, logdet = self.linear.backward(out)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.actnorm.backward(out)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.actnorm.init(data, init_scale=init_scale)

        out, logdet = self.linear.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.unit.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class PriorFlow(Flow):
    """
    Prior Flow
    """
    def __init__(self, num_steps, in_features, hidden_features, transform='affine', alpha=1.0, coupling_type='mlp', activation='elu'):
        inverse = True
        super(PriorFlow, self).__init__(inverse)
        steps = [PriorFlowStep(in_features, hidden_features, transform=transform, alpha=alpha, inverse=inverse,
                               coupling_type=coupling_type, activation=activation) for _ in range(num_steps)]
        self.steps = nn.ModuleList(steps)

    def sync(self):
        for step in self.steps:
            step.sync()

    @overrides
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = input
        # [batch]
        logdet_accum = input.new_zeros(input.size(0))
        for step in self.steps:
            out, logdet = step.forward(out)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        for step in reversed(self.steps):
            out, logdet = step.backward(out)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out = data
        # [batch]
        logdet_accum = data.new_zeros(data.size(0))
        for step in self.steps:
            out, logdet = step.init(out, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class FlowPrior(Prior):
    """
    Prior base class
    """

    def __init__(self, num_steps, in_features, hidden_features, transform='affine', alpha=1.0, coupling_type='mlp', activation='elu'):
        super(FlowPrior, self).__init__()
        self.flow = PriorFlow(num_steps, in_features, hidden_features, transform=transform, alpha=alpha,
                              coupling_type=coupling_type, activation=activation)

    @overrides
    def log_probability(self, z):
        size = z.size()
        # [batch, nsamples, dim] -> [batch * nsamples, dim]
        z = z.view(-1, size[2])
        epsilon, logdet = self.flow.bwdpass(z)
        # [batch * nsamples]
        log_probs = epsilon.mul(epsilon).sum(dim=1) + math.log(math.pi * 2.) * size[2]
        log_probs = log_probs * -0.5 + logdet
        return log_probs.view(size[0], size[1])

    @overrides
    def sample(self, nsamples, dim, device=torch.device('cpu')):
        # [nsamples, dim]
        epsilon = torch.randn(nsamples, dim, device=device)
        z, _ = self.flow.fwdpass(epsilon)
        return z

    @overrides
    def calcKL(self, z, eps, mu, logvar):
        dim = z.size(2)
        cc = math.log(math.pi * 2.)
        # calc posterior
        # [batch, nsamples, dim]
        log_posterior = logvar.unsqueeze(1) + eps.pow(2)
        # [batch, nsamples, dim] --> [batch, nsamples]
        log_posterior = log_posterior.sum(dim=2) + cc * dim
        # [batch]
        log_posterior = log_posterior.mean(dim=1) * -0.5

        # calc prior
        nsamples = z.size(1)
        # [batch, nsamples, dim]
        epsilon, logdet = self.flow.bwdpass(z)
        # [batch, numels]
        epsilon = epsilon.view(epsilon.size(0), -1)
        # [batch]
        log_prior = epsilon.mul(epsilon).sum(dim=1) + cc * epsilon.size(1)
        log_prior = (log_prior * -0.5 + logdet).div(nsamples)
        return log_posterior - log_prior

    @overrides
    def init(self, z, eps, mu, logvar, init_scale=1.0):
        dim = z.size(2)
        cc = math.log(math.pi * 2.)
        # calc posterior
        # [batch, nsamples, dim]
        log_posterior = logvar.unsqueeze(1) + eps.pow(2)
        # [batch, nsamples, dim] --> [batch, nsamples]
        log_posterior = log_posterior.sum(dim=2) + cc * dim
        # [batch]
        log_posterior = log_posterior.mean(dim=1) * -0.5

        # calc prior
        nsamples = z.size(1)
        # [batch, nsamples, dim]
        epsilon, logdet = self.flow.bwdpass(z, init=True, init_scale=init_scale)
        # [batch, numels]
        epsilon = epsilon.view(epsilon.size(0), -1)
        # [batch]
        log_prior = epsilon.mul(epsilon).sum(dim=1) + cc * epsilon.size(1)
        log_prior = (log_prior * -0.5 + logdet).div(nsamples)
        return log_posterior - log_prior

    def sync(self):
        self.flow.sync()

    @classmethod
    def from_params(cls, params: Dict) -> "FlowPrior":
        return FlowPrior(**params)


FlowPrior.register('flow')
