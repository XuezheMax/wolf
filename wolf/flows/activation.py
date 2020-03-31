__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple
import math
import torch
import torch.nn.functional as F

from wolf.flows.flow import Flow
from wolf.utils import logPlusOne


class IdentityFlow(Flow):
    def __init__(self, inverse=False):
        super(IdentityFlow, self).__init__(inverse)

    @overrides
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, *]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        return input, input.new_zeros(input.size(0))

    @overrides
    def backward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, *]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        return input, input.new_zeros(input.size(0))

    @overrides
    def init(self, data, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data)

    @overrides
    def extra_repr(self):
        return 'inverse={}'.format(self.inverse)

    @classmethod
    def from_params(cls, params: Dict) -> "IdentityFlow":
        return IdentityFlow(**params)


class PowshrinkFlow(Flow):
    def __init__(self, exponent=2.0, inverse=False):
        super(PowshrinkFlow, self).__init__(inverse)
        assert exponent >= 1.0, 'exponent should be greater or equal to 1.0'
        self.exponent=exponent

    @overrides
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, *]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        sign = input.sign()
        input = input * sign
        mask = input.lt(1.0).type_as(input)
        out = input * (1.0 - mask) + input.pow(self.exponent) * mask
        out = out * sign
        # [batch]
        logdet = ((input + 1e-8).log().mul(self.exponent - 1) + math.log(self.exponent)).mul(mask).view(input.size(0), -1).sum(dim=1)
        return out, logdet

    @overrides
    def backward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, *]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        sign = input.sign()
        input = input * sign
        mask = input.lt(1.0).type_as(input)
        out = input * (1.0 - mask) + input.pow(1. / self.exponent) * mask
        out = out * sign
        # [batch]
        logdet = ((input + 1e-8).log().mul(1.0 / self.exponent - 1) - math.log(self.exponent)).mul(mask).view(out.size(0), -1).sum(dim=1)
        return out, logdet

    @overrides
    def init(self, data, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data)

    @overrides
    def extra_repr(self):
        return 'inverse={}'.format(self.inverse)

    @classmethod
    def from_params(cls, params: Dict) -> "PowshrinkFlow":
        return PowshrinkFlow(**params)


class LeakyReLUFlow(Flow):
    def __init__(self, negative_slope=0.1, inverse=False):
        super(LeakyReLUFlow, self).__init__(inverse)
        assert negative_slope > 0.0, 'negative slope should be positive'
        self.negative_slope = negative_slope

    @overrides
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, *]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        out = F.leaky_relu(input, self.negative_slope, False)
        log_slope = math.log(self.negative_slope)
        # [batch]
        logdet = input.view(input.size(0), -1).lt(0.0).type_as(input).sum(dim=1) * log_slope
        return out, logdet

    @overrides
    def backward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, *]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        negative_slope = 1.0 / self.negative_slope
        out = F.leaky_relu(input, negative_slope, False)
        log_slope = math.log(negative_slope)
        # [batch]
        logdet = input.view(input.size(0), -1).lt(0.0).type_as(input).sum(dim=1) * log_slope
        return out, logdet

    @overrides
    def init(self, data, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data)

    @overrides
    def extra_repr(self):
        return 'inverse={}, negative_slope={}'.format(self.inverse, self.negative_slope)

    @classmethod
    def from_params(cls, params: Dict) -> "LeakyReLUFlow":
        return LeakyReLUFlow(**params)


class ELUFlow(Flow):
    def __init__(self, alpha=1.0, inverse=False):
        super(ELUFlow, self).__init__(inverse)
        self.alpha = alpha

    @overrides
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, *]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        out = F.elu(input, self.alpha, False)
        # [batch, numel]
        input = input.view(input.size(0), -1)
        logdet = input + math.log(self.alpha)
        # [batch]
        logdet = (input.lt(0.0).type_as(input) * logdet).sum(dim=1)
        return out, logdet

    @overrides
    def backward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, *]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        mask = input.lt(0.0).type_as(input)
        out = input * (1.0 - mask) + mask * logPlusOne(input.div(self.alpha))
        # [batch, numel]
        out_flat = out.view(input.size(0), -1)
        logdet = out_flat + math.log(self.alpha)
        # [batch]
        logdet = (mask.view(out_flat.size()) * logdet).sum(dim=1).mul(-1.0)
        return out, logdet

    @overrides
    def init(self, data, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data)

    @overrides
    def extra_repr(self):
        return 'inverse={}, alpha={}'.format(self.inverse, self.alpha)

    @classmethod
    def from_params(cls, params: Dict) -> "ELUFlow":
        return ELUFlow(**params)


class SigmoidFlow(Flow):
    def __init__(self, inverse=False):
        super(SigmoidFlow, self).__init__(inverse)

    @overrides
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, *]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        out = input.sigmoid()
        logdet = F.softplus(input) + F.softplus(-input)
        logdet = logdet.view(logdet.size(0), -1).sum(dim=1) * -1.
        return out, logdet

    @overrides
    def backward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, *]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        eps = 1e-12
        out = torch.log(torch.reciprocal(input + eps) - 1.  + eps) * -1.
        logdet = torch.log(input + eps) + torch.log((1. - input) + eps)
        logdet = logdet.view(logdet.size(0), -1).sum(dim=1) * -1.
        return out, logdet

    @overrides
    def init(self, data, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data)

    @overrides
    def extra_repr(self):
        return 'inverse={}'.format(self.inverse)

    @classmethod
    def from_params(cls, params: Dict) -> "SigmoidFlow":
        return SigmoidFlow(**params)


PowshrinkFlow.register('power_shrink')
LeakyReLUFlow.register('leaky_relu')
ELUFlow.register('elu')
IdentityFlow.register('identity')
SigmoidFlow.register('sigmoid')
