__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter

from wolf.flows.flow import Flow


class ActNorm1dFlow(Flow):
    def __init__(self, in_features, inverse=False):
        super(ActNorm1dFlow, self).__init__(inverse)
        self.in_features = in_features
        self.log_scale = Parameter(torch.Tensor(in_features))
        self.bias = Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.log_scale, mean=0, std=0.05)
        nn.init.constant_(self.bias, 0.)

    @overrides
    def forward(self, input: torch.Tensor, mask: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, N1, N2, ..., in_channels]
            mask: Tensor or None
                mask tensor [batch, N1, N2, ...,Nl]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_channels], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        dim = input.dim()
        out = input * self.log_scale.exp() + self.bias
        if mask is not None:
            out = out * mask.unsqueeze(dim - 1)

        logdet = self.log_scale.sum(dim=0, keepdim=True)
        if dim > 2:
            num = np.prod(input.size()[1:-1]).astype(float) if mask is None else mask.view(out.size(0), -1).sum(dim=1)
            logdet = logdet * num
        return out, logdet

    @overrides
    def backward(self, input: torch.Tensor, mask: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: input: Tensor
                input tensor [batch, N1, N2, ..., in_channels]
            mask: Tensor or None
                mask tensor [batch, N1, N2, ...,Nl]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_channels], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        dim = input.dim()
        out = (input - self.bias).div(self.log_scale.exp() + 1e-8)
        if mask is not None:
            out = out * mask.unsqueeze(dim - 1)

        logdet = self.log_scale.sum(dim=0, keepdim=True) * -1.0
        if input.dim() > 2:
            num = np.prod(input.size()[1:-1]).astype(float) if mask is None else mask.view(out.size(0), -1).sum(dim=1)
            logdet = logdet * num
        return out, logdet

    @overrides
    def init(self, data, mask: Union[torch.Tensor, None] = None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            data: input: Tensor
                input tensor [batch, N1, N2, ..., in_channels]
            mask: Tensor or None
                mask tensor [batch, N1, N2, ...,Nl]
            init_scale: float
                initial scale

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_channels], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        with torch.no_grad():
            # [batch * N1 * ... * Nl, in_features]
            out, _ = self.forward(data, mask=mask)
            out = out.view(-1, self.in_features)
            mean = out.mean(dim=0)
            std = out.std(dim=0)
            inv_stdv = init_scale / (std + 1e-6)

            self.log_scale.add_(inv_stdv.log())
            self.bias.add_(-mean).mul_(inv_stdv)
            return self.forward(data, mask=mask)

    @overrides
    def extra_repr(self):
        return 'inverse={}, in_features={}'.format(self.inverse, self.in_features)

    @classmethod
    def from_params(cls, params: Dict) -> "ActNorm1dFlow":
        return ActNorm1dFlow(**params)


class ActNorm2dFlow(Flow):
    def __init__(self, in_channels, inverse=False):
        super(ActNorm2dFlow, self).__init__(inverse)
        self.in_channels = in_channels
        self.log_scale = Parameter(torch.Tensor(in_channels, 1, 1))
        self.bias = Parameter(torch.Tensor(in_channels, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.log_scale, mean=0, std=0.05)
        nn.init.constant_(self.bias, 0.)

    @overrides
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        batch, channels, H, W = input.size()
        # [channels, 1, 1]
        log_scale = self.log_scale
        bias = self.bias
        out = input * log_scale.exp() + bias
        logdet = log_scale.view(1, channels).sum(dim=1).mul(H * W)
        return out, logdet

    @overrides
    def backward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        batch, channels, H, W = input.size()
        # [channels, 1, 1]
        log_scale = self.log_scale
        bias = self.bias
        out = (input - bias).div(log_scale.exp() + 1e-8)
        logdet = log_scale.view(1, channels).sum(dim=1).mul(H * -W)
        return out, logdet

    @overrides
    def init(self, data, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # [batch, n_channels, H, W]
            out, _ = self.forward(data)
            out = out.transpose(0, 1).contiguous().view(self.in_channels, -1)
            # [n_channels, 1, 1]
            mean = out.mean(dim=1).view(self.in_channels, 1, 1)
            std = out.std(dim=1).view(self.in_channels, 1, 1)
            inv_stdv = init_scale / (std + 1e-6)

            self.log_scale.add_(inv_stdv.log())
            self.bias.add_(-mean).mul_(inv_stdv)
            return self.forward(data)

    @overrides
    def extra_repr(self):
        return 'inverse={}, in_channels={}'.format(self.inverse, self.in_channels)

    @classmethod
    def from_params(cls, params: Dict) -> "ActNorm2dFlow":
        return ActNorm2dFlow(**params)


ActNorm1dFlow.register('actnorm')
ActNorm2dFlow.register('actnorm2d')
