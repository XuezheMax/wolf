__author__ = 'max'

import math
from overrides import overrides
from typing import Tuple
import torch


class Transform():
    def calc_params(self, params):
        return params

    @staticmethod
    def fwd(z: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    def bwd(z: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__ + '(' + self.extra_repr() + ')'

    def extra_repr(self):
        return ''


class Additive(Transform):
    def __init__(self):
        super(Additive, self).__init__()

    @staticmethod
    @overrides
    def fwd(z: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = params
        z = z + mu
        logdet = z.new_zeros(z.size(0))
        return z, logdet

    @staticmethod
    @overrides
    def bwd(z: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = params
        z = z - mu
        logdet = z.new_zeros(z.size(0))
        return z, logdet


class Affine(Transform):
    def __init__(self, dim, alpha):
        super(Affine, self).__init__()
        self.dim = dim
        self.alpha = alpha

    @overrides
    def calc_params(self, params):
        mu, log_scale = params.chunk(2, dim=self.dim)
        scale = log_scale.mul_(0.5).tanh_().mul(self.alpha).add(1.0)
        return mu, scale

    @staticmethod
    @overrides
    def fwd(z: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, scale = params
        z = scale * z + mu
        logdet = scale.log().view(z.size(0), -1).sum(dim=1)
        return z, logdet

    @staticmethod
    @overrides
    def bwd(z: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, scale = params
        z = (z - mu).div(scale + 1e-12)
        logdet = scale.log().view(z.size(0), -1).sum(dim=1) * -1.0
        return z, logdet

    @overrides
    def extra_repr(self):
        return 'dim={}, alpha={}'.format(self.dim, self.alpha)


class ReLU(Transform):
    def __init__(self, dim):
        super(ReLU, self).__init__()
        self.dim = dim

    @overrides
    def calc_params(self, params):
        mu, log_scale = params.chunk(2, dim=self.dim)
        scale = log_scale.tanh_()
        return mu, scale

    @staticmethod
    @overrides
    def fwd(z: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, scale = params
        scale = scale * z.gt(0.0).type_as(z) + 1
        z = scale * z + mu
        logdet = scale.log().view(z.size(0), -1).sum(dim=1)
        return z, logdet

    @staticmethod
    @overrides
    def bwd(z: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, scale = params
        z = z - mu
        scale = scale * z.gt(0.0).type_as(z) + 1
        z = z.div(scale + 1e-12)
        logdet = scale.log().view(z.size(0), -1).sum(dim=1) * -1.0
        return z, logdet


def arccosh(x):
    return torch.log(x + torch.sqrt(x.pow(2) - 1))


def arcsinh(x):
    return torch.log(x + torch.sqrt(x.pow(2) + 1))


class NLSQ(Transform):
    # A = 8 * math.sqrt(3) / 9 - 0.05  # 0.05 is a small number to prevent exactly 0 slope
    logA = math.log(8 * math.sqrt(3) / 9 - 0.05)  # 0.05 is a small number to prevent exactly 0 slope

    def __init__(self, dim):
        super(NLSQ, self).__init__()
        self.dim = dim

    @overrides
    def calc_params(self, params):
        a, logb, cprime, logd, g = params.chunk(5, dim=self.dim)

        # for stability
        logb = logb.mul_(0.4)
        cprime = cprime.mul_(0.3)
        logd = logd.mul_(0.4)

        # b = logb.add_(2.0).sigmoid_()
        # d = logd.add_(2.0).sigmoid_()
        # c = (NLSQ.A * b / d).mul(cprime.tanh_())

        c = (NLSQ.logA + logb - logd).exp_().mul(cprime.tanh_())
        b = logb.exp_()
        d = logd.exp_()
        return a, b, c, d, g

    @staticmethod
    @overrides
    def fwd(z: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b, c, d, g = params

        arg = (d * z).add_(g)
        denom = arg.pow(2).add_(1)
        c = c / denom
        z = b * z + a + c
        logdet = torch.log(b - 2 * c * d * arg / denom)
        logdet = logdet.view(z.size(0), -1).sum(dim=1)
        return z, logdet

    @staticmethod
    @overrides
    def bwd(z: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        a, b, c, d, g = params

        # double needed for stability. No effect on overall speed
        a = a.double()
        b = b.double()
        c = c.double()
        d = d.double()
        g = g.double()
        z = z.double()

        aa = -b * d.pow(2)
        bb = (z - a) * d.pow(2) - 2 * b * d * g
        cc = (z - a) * 2 * d * g - b * (1 + g.pow(2))
        dd = (z - a) * (1 + g.pow(2)) - c

        p = (3 * aa * cc - bb.pow(2)) / (3 * aa.pow(2))
        q = (2 * bb.pow(3) - 9 * aa * bb * cc + 27 * aa.pow(2) * dd) / (27 * aa.pow(3))

        t = -2 * torch.abs(q) / q * torch.sqrt(torch.abs(p) / 3)
        inter_term1 = -3 * torch.abs(q) / (2 * p) * torch.sqrt(3 / torch.abs(p))
        inter_term2 = 1 / 3 * arccosh(torch.abs(inter_term1 - 1) + 1)
        t = t * torch.cosh(inter_term2)

        tpos = -2 * torch.sqrt(torch.abs(p) / 3)
        inter_term1 = 3 * q / (2 * p) * torch.sqrt(3 / torch.abs(p))
        inter_term2 = 1 / 3 * arcsinh(inter_term1)
        tpos = tpos * torch.sinh(inter_term2)

        t[p > 0] = tpos[p > 0]
        z = t - bb / (3 * aa)
        arg = d * z + g
        denom = arg.pow(2) + 1
        logdet = torch.log(b - 2 * c * d * arg / denom.pow(2))

        z = z.float()
        logdet = logdet.float().view(z.size(0), -1).sum(dim=1) * -1.0
        return z, logdet


class SymmELU(Transform):
    def __init__(self, dim):
        super(SymmELU, self).__init__()
        self.dim = dim

    @overrides
    def calc_params(self, params):
        mu, log_scale = params.chunk(2, dim=self.dim)
        scale = log_scale.mul_(0.5).tanh_()
        return mu, scale

    @staticmethod
    @overrides
    def fwd(z: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, scale = params
        sgn = torch.sign(z)
        tmp = torch.exp(-torch.abs(z))
        z = z - sgn * scale * (tmp - 1.0) + mu
        logdet = (scale * tmp + 1).log().view(z.size(0), -1).sum(dim=1)
        return z, logdet

    @staticmethod
    @overrides
    def bwd(z: torch.Tensor, params) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, scale = params
        z = -torch.sign(z) * scale * (torch.exp(-torch.abs(z)) - 1.0) + mu
        return z, None

    @overrides
    def extra_repr(self):
        return 'dim={}'.format(self.dim)
