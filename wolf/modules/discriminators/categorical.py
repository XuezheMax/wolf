__author__ = 'max'

from typing import Dict
from overrides import overrides
import math
import torch
import torch.nn as nn
from torch.distributions import Categorical

from wolf.modules.discriminators.discriminator import Discriminator


class CategoricalDiscriminator(Discriminator):
    """
    Prior with categorical distribution (using for class label conditioned generation)
    """
    def __init__(self, num_events, dim, activation='relu', probs=None, logits=None):
        super(CategoricalDiscriminator, self).__init__()
        if probs is not None and logits is not None:
            raise ValueError("Either `probs` or `logits` can be specified, but not both.")

        if probs is not None:
            assert len(probs) == num_events, 'number of probs must match number of events.'
            probs = torch.tensor(probs).float()
            self.cat_dist = Categorical(probs=probs)
        elif logits is not None:
            assert len(logits) == num_events, 'number of logits must match number of events.'
            logits = torch.tensor(logits).float()
            self.cat_dist = Categorical(logits=logits)
        else:
            probs = torch.full((num_events, ), 1.0 / num_events).float()
            self.cat_dist = Categorical(probs=probs)

        if activation == 'relu':
            Actv = nn.ReLU(inplace=True)
        elif activation == 'elu':
            Actv = nn.ELU(inplace=True)
        else:
            Actv = nn.LeakyReLU(inplace=True, negative_slope=1e-1)
        self.embed = nn.Embedding(num_events, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            Actv,
            nn.Linear(4 * dim, 4 * dim),
            Actv,
            nn.Linear(4 * dim, dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)

    @overrides
    def to_device(self, device):
        logits = self.cat_dist.logits.to(device)
        self.cat_dist = Categorical(logits=logits)

    @overrides
    def init(self, x, y=None, init_scale=1.0):
        with torch.no_grad():
            z, KL = self.sampling_and_KL(x, y=y)
            return z.squeeze(1), KL

    @overrides
    def sample_from_prior(self, nsamples=1, device=torch.device('cpu')):
        # [nsamples]
        cids = self.cat_dist.sample((nsamples, )).to(device)
        cids = torch.sort(cids)[0]
        # [nsamples, dim]
        return self.net(self.embed(cids))

    @overrides
    def sample_from_posterior(self, x, y=None, nsamples=1, random=True):
        assert y is not None
        log_probs = x.new_zeros(x.size(0), nsamples)
        # [batch, nsamples, dim]
        z = self.net(self.embed(y)).unsqueeze(1) + log_probs.unsqueeze(2)
        return z, log_probs

    @overrides
    def sampling_and_KL(self, x, y=None, nsamples=1):
        # [batch, nsamples, dim]
        z, _ = self.sample_from_posterior(x, y=y, nsamples=nsamples, random=True)
        # [batch,]
        log_probs_prior = self.cat_dist.log_prob(y)
        KL = -log_probs_prior
        return z, KL

    @classmethod
    def from_params(cls, params: Dict) -> "CategoricalDiscriminator":
        return CategoricalDiscriminator(**params)


CategoricalDiscriminator.register('categorical')
