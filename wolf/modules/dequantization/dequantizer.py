__author__ = 'max'

from typing import Dict, Tuple
from overrides import overrides
import math
import torch
import torch.nn as nn

from wolf.flows.flow import Flow
from wolf.flows.activation import SigmoidFlow
from wolf.modules.encoders.encoder import Encoder


class DeQuantizer(nn.Module):
    """
    Dequantizer base class
    """
    _registry = dict()

    def __init__(self):
        super(DeQuantizer, self).__init__()

    def dequantize(self, x, nsamples=1) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x: Tensor [batch, channels, height, width]
                tensor for input images
            nsamples: int (default: 1)
                number of samples

        Returns: Tensor1, Tensor2
            Tensor1: noise tensor for dequantization [batch, nsamples, channels, height, width]
            Tensor2: log probs [batch, nsamples]

        """
        raise NotImplementedError

    def init(self, x, init_scale=1.0):
        raise NotImplementedError

    @classmethod
    def register(cls, name: str):
        DeQuantizer._registry[name] = cls

    @classmethod
    def by_name(cls, name: str):
        return DeQuantizer._registry[name]

    @classmethod
    def from_params(cls, params: Dict):
        raise NotImplementedError


class UniformDeQuantizer(DeQuantizer):
    def __init__(self):
        super(UniformDeQuantizer, self).__init__()

    @overrides
    def dequantize(self, x, nsamples=1) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, nsamples, channels, H, W]
        return x.new_empty(x.size(0), nsamples, *x.size()[1:]).uniform_(), \
               x.new_zeros(x.size(0), nsamples)

    @overrides
    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            return self.dequantize(x)

    @classmethod
    def from_params(cls, params: Dict) -> "UniformDeQuantizer":
        return UniformDeQuantizer()


class FlowDeQuantizer(DeQuantizer):
    def __init__(self, encoder: Encoder, flow: Flow):
        super(FlowDeQuantizer, self).__init__()
        self.encoder = encoder
        self.flow = DequantFlow(flow)

    @overrides
    def dequantize(self, x, nsamples=1) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = x.size(0)
        # [batch, *]
        h = self.encoder(x)
        # [batch * nsamples, channels, H, W]
        epsilon = torch.randn(batch * nsamples, *x.size()[1:], device=x.device)
        if nsamples > 1:
            # [batch, nsamples, *]
            h = h.unsqueeze(1) + h.new_zeros(batch, nsamples, *h.size()[1:])
            # [batch * nsamples, *]
            h = h.view(-1, *h.size()[2:])
        u, logdet = self.flow.fwdpass(epsilon, h)
        # [batch * nsamples, channels * H * W]
        epsilon = epsilon.view(epsilon.size(0), -1)
        # [batch * nsamples]
        log_posteriors = epsilon.mul(epsilon).sum(dim=1) + math.log(math.pi * 2.) * epsilon.size(1)
        log_posteriors = log_posteriors.mul(-0.5) - logdet
        return u.view(batch, nsamples, *x.size()[1:]), log_posteriors.view(batch, nsamples)

    @overrides
    def init(self, x, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, *]
        h = self.encoder.init(x, init_scale=init_scale)
        # [batch, channels, H, W]
        epsilon = torch.randn(x.size(), device=x.device)
        u, logdet = self.flow.fwdpass(epsilon, h, init=True, init_scale=init_scale)
        # [batch, channels * H * W]
        epsilon = epsilon.view(epsilon.size(0), -1)
        # [batch ]
        log_posteriors = epsilon.mul(epsilon).sum(dim=1) + math.log(math.pi * 2.) * epsilon.size(1)
        log_posteriors = log_posteriors.mul(-0.5) - logdet
        return u, log_posteriors

    @classmethod
    def from_params(cls, params: Dict) -> "FlowDeQuantizer":
        flow_params = params.pop('flow')
        flow = Flow.by_name(flow_params.pop('type')).from_params(flow_params)
        encoder_params = params.pop('encoder')
        encoder = Encoder.by_name(encoder_params.pop('type')).from_params(encoder_params)
        return FlowDeQuantizer(encoder, flow)


class DequantFlow(Flow):
    def __init__(self, core: Flow):
        assert not core.inverse
        super(DequantFlow, self).__init__(False)
        assert not core.inverse, 'dequantization flow should not be in inverse mode.'
        self.core = core
        self.sigmoid = SigmoidFlow(inverse=False)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.core.forward(input, h=h)
        out, logdet = self.sigmoid.forward(out)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.sigmoid.backward(input)
        out, logdet = self.core.backward(out, h=h)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.core.init(data, h=h, init_scale=init_scale)
        out, logdet = self.sigmoid.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum


UniformDeQuantizer.register('uniform')
FlowDeQuantizer.register('flow')
