__author__ = 'max'

import math
from typing import Dict, Tuple, Union
import torch
import torch.nn as nn

from wolf.flows.flow import Flow


class Generator(nn.Module):
    """
    class for Generator with a Flow.
    """

    def __init__(self, flow: Flow):
        super(Generator, self).__init__()
        self.flow = flow

    def sync(self):
        self.flow.sync()

    def generate(self, epsilon: torch.Tensor,
                 h: Union[None, torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            epsilon: Tensor [batch, channels, height, width]
                epslion for generation
            h: Tensor or None [batch, dim]
                conditional input

        Returns: Tensor1, Tensor2
            Tensor1: generated tensor [batch, channels, height, width]
            Tensor2: log probabilities [batch]

        """
        # [batch, channel, height, width]
        z, logdet = self.flow.fwdpass(epsilon, h)
        return z, logdet

    def encode(self, x: torch.Tensor, h: Union[None, torch.Tensor] = None) -> torch.Tensor:
        """

        Args:
            x: Tensor [batch, channels, height, width]
                The input data.
            h: Tensor or None [batch, dim]
                conditional input

        Returns: Tensor [batch, channels, height, width]
            The tensor for encoded epsilon.

        """
        return self.flow.bwdpass(x, h)[0]

    def log_probability(self, x: torch.Tensor, h: Union[None, torch.Tensor] = None) -> torch.Tensor:
        """

        Args:
            x: Tensor [batch, channel, height, width]
                The input data.
            h: Tensor or None [batch, dim]
                conditional input

        Returns: Tensor [batch,]
            The tensor of the log probabilities of x

        """
        # [batch, channel, height, width]
        epsilon, logdet = self.flow.bwdpass(x, h)
        # [batch, numels]
        epsilon = epsilon.view(epsilon.size(0), -1)
        # [batch]
        log_probs = epsilon.mul(epsilon).sum(dim=1) + math.log(math.pi * 2.) * epsilon.size(1)
        return log_probs.mul(-0.5) + logdet

    def init(self, data: torch.Tensor, h=None, init_scale=1.0):
        return self.flow.bwdpass(data, h, init=True, init_scale=init_scale)

    @classmethod
    def from_params(cls, params: Dict) -> "Generator":
        flow_params = params.pop('flow')
        flow = Flow.by_name(flow_params.pop('type')).from_params(flow_params)
        return Generator(flow)
