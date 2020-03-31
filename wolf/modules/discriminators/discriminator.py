__author__ = 'max'

from typing import Dict
import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Discriminator base class
    """
    _registry = dict()

    def __init__(self):
        super(Discriminator, self).__init__()

    def sample_from_prior(self, nsamples=1, device=torch.device('cpu')):
        """

        Args:
            nsamples: int
                Number of samples
            device: torch.device
                device to store the samples

        Returns: Tensor[nsamples, dim]
            the tensor of samples

        """
        return None

    def sample_from_posterior(self, x, y=None, nsamples=1, random=True):
        """

        Args:
            x: Tensor
                The input data
            y: Tensor or None
                The label id of the data (for conditional generation).
            nsamples: int
                Number of samples for each instance.
            random: bool
                if True, perform random sampling.

        Returns: Tensor1, Tensor2
            Tensor1: samples from the posterior [batch, nsamples, dim]
            Tensor2: log probabilities [batch, nsamples]

        """
        return None, None

    def sampling_and_KL(self, x, y=None, nsamples=1):
        """

        Args:
            x: Tensor
                The input data
            y: Tensor or None
                The label id of the data (for conditional generation).
            nsamples: int
                Number of samples for each instance.

        Returns: Tensor1, Tensor2, Tensor3, Tensor4
            Tensor1: samples from the posterior [batch, nsamples, dim]
            Tensor2: tensor for KL [batch,]
            # Tensor3: log probabilities of posterior [batch, nsamples]
            # Tensor4: log probabilities of prior [batch, nsamples]

        """
        return None, None

    def init(self, x, y=None, init_scale=1.0):
        with torch.no_grad():
            return self.sampling_and_KL(x, y=y)

    def to_device(self, device):
        pass

    def sync(self):
        pass

    @classmethod
    def register(cls, name: str):
        Discriminator._registry[name] = cls

    @classmethod
    def by_name(cls, name: str):
        return Discriminator._registry[name]

    @classmethod
    def from_params(cls, params: Dict) -> "Discriminator":
        return Discriminator()


Discriminator.register('base')
