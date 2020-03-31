__author__ = 'max'

from typing import Dict
from overrides import overrides
import math
import torch
import torch.nn as nn


class Prior(nn.Module):
    """
    Prior base class
    """
    _registry = dict()

    def __init__(self):
        super(Prior, self).__init__()

    def log_probability(self, z):
        raise NotImplementedError

    def sample(self, nsample, dim, device=torch.device('cpu')):
        raise NotImplementedError

    def calcKL(self, z, eps, mu, logvar):
        raise NotImplementedError

    def init(self, z, eps, mu, logvar, init_scale=1.0):
        raise NotImplementedError

    def sync(self):
        pass

    @classmethod
    def register(cls, name: str):
        Prior._registry[name] = cls

    @classmethod
    def by_name(cls, name: str):
        return Prior._registry[name]

    @classmethod
    def from_params(cls, params: Dict) -> "Prior":
        raise NotImplementedError


class NormalPrior(Prior):
    """
    Prior base class
    """

    def __init__(self):
        super(NormalPrior, self).__init__()

    @overrides
    def log_probability(self, z):
        # [batch, nsamples, dim]
        dim = z.size(2)
        # [batch, nsamples]
        log_probs = z.pow(2).sum(dim=2) + math.log(math.pi * 2.) * dim
        return log_probs * -0.5

    @overrides
    def sample(self, nsamples, dim, device=torch.device('cpu')):
        epsilon = torch.randn(nsamples, dim, device=device)
        return epsilon

    @overrides
    def calcKL(self, z, eps, mu, logvar):
        return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)

    @overrides
    def init(self, z, eps, mu, logvar, init_scale=1.0):
        return self.calcKL(z, eps, mu, logvar)

    @classmethod
    def from_params(cls, params: Dict) -> "NormalPrior":
        return NormalPrior()


NormalPrior.register('normal')
