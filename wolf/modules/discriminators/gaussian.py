__author__ = 'max'

from typing import Dict
from overrides import overrides
import math
import torch

from wolf.nnet.weight_norm import LinearWeightNorm
from wolf.modules.discriminators.discriminator import Discriminator
from wolf.modules.encoders.encoder import Encoder
from wolf.modules.discriminators.priors.prior import Prior


class GaussianDiscriminator(Discriminator):
    def __init__(self, encoder: Encoder, in_dim, dim, prior: Prior):
        super(GaussianDiscriminator, self).__init__()
        self.dim = dim
        self.encoder = encoder
        self.fc = LinearWeightNorm(in_dim, 2 * dim, bias=True)
        self.prior = prior

    def forward(self, x):
        c = self.encoder(x)
        c = self.fc(c)
        mu, logvar = c.chunk(2, dim=1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar, nsamples=1, random=True):
        # [batch, dim]
        size = mu.size()
        std = logvar.mul(0.5).exp()
        # [batch, nsamples, dim]
        if random:
            eps = torch.randn(size[0], nsamples, size[1], device=mu.device)
        else:
            eps = mu.new_zeros(size[0], nsamples, size[1])
        return eps.mul(std.unsqueeze(1)).add(mu.unsqueeze(1)), eps

    @staticmethod
    def log_probability_posterior(eps, logvar):
        size = eps.size()
        dim = size[2]
        # [batch, nsamples, dim]
        log_probs = logvar.unsqueeze(1) + eps.pow(2)
        # [batch, 1]
        cc = math.log(math.pi * 2.) * dim
        # [batch, nsamples, dim] --> [batch, nsamples]
        log_probs = log_probs.sum(dim=2) + cc
        return log_probs * -0.5

    @overrides
    def sample_from_prior(self, nsamples=1, device=torch.device('cpu')):
        return self.prior.sample(nsamples, self.dim, device)

    @overrides
    def sample_from_posterior(self, x, y=None, nsamples=1, random=True):
        # [batch, dim]
        mu, logvar = self(x)
        # [batch, nsamples, dim]
        z, eps = GaussianDiscriminator.reparameterize(mu, logvar, nsamples=nsamples, random=random)
        # [batch, nsamples]
        log_probs = GaussianDiscriminator.log_probability_posterior(eps, logvar)
        return z, log_probs

    @overrides
    def sampling_and_KL(self, x, y=None, nsamples=1):
        mu, logvar = self(x)
        # [batch, nsamples, dim]
        z, eps = GaussianDiscriminator.reparameterize(mu, logvar, nsamples=nsamples, random=True)
        # [batch,]
        KL = self.prior.calcKL(z, eps, mu, logvar)
        # [batch, nsamples]
        # log_probs_posterior = GaussianDiscriminator.log_probability_posterior(eps, logvar)
        # log_probs_prior = GaussianDiscriminator.log_probability_prior(z)
        return z, KL

    @overrides
    def init(self, x, y=None, init_scale=1.0):
        with torch.no_grad():
            c = self.encoder.init(x, init_scale=init_scale)
            c = self.fc.init(c, init_scale=0.01 * init_scale)
            mu, logvar = c.chunk(2, dim=1)
            # [batch, 1, dim]
            z, eps = GaussianDiscriminator.reparameterize(mu, logvar, nsamples=1, random=True)
            # [batch,]
            KL = self.prior.init(z, eps, mu, logvar, init_scale=init_scale)
            return z.squeeze(1), KL

    @overrides
    def sync(self):
        self.prior.sync()

    @classmethod
    def from_params(cls, params: Dict) -> "GaussianDiscriminator":
        encoder_params = params.pop('encoder')
        encoder = Encoder.by_name(encoder_params.pop('type')).from_params(encoder_params)
        prior_params = params.pop('prior')
        prior = Prior.by_name(prior_params.pop('type')).from_params(prior_params)
        return GaussianDiscriminator(encoder=encoder, prior=prior, **params)


GaussianDiscriminator.register('gaussian')
