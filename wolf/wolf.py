__author__ = 'max'

import os
import json
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.distributed as dist
from apex.parallel import DistributedDataParallel, convert_syncbn_model

from wolf.data.image import preprocess, postprocess
from wolf.modules import DeQuantizer
from wolf.modules import Discriminator
from wolf.modules import Generator


class WolfCore(nn.Module):
    """
    core module for FloWAE model
    """
    def __init__(self, generator: Generator, discriminator: Discriminator, dequantizer: DeQuantizer):
        super(WolfCore, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.dequantizer = dequantizer

    def sync(self):
        self.generator.sync()
        self.discriminator.sync()

    def init(self, x, y=None, init_scale=1.0):
        z = self.discriminator.init(x, y=y, init_scale=init_scale)[0]
        self.dequantizer.init(x, init_scale=init_scale)
        self.generator.init(x, h=z, init_scale=init_scale)

    def synthesize(self, nums, image_size, tau=1.0, n_bits=8, device=torch.device('cpu')):
        # [nsamples, imagesize]
        epsilon = torch.randn(nums, *image_size, device=device)
        epsilon *= tau
        z = self.discriminator.sample_from_prior(nums, device)
        imgs, _ = self.generator.generate(epsilon, h=z)
        imgs = postprocess(imgs, n_bits=n_bits)
        return imgs

    def encode_global(self, data, y=None, n_bits=8, nsamples=1, random=False):
        x = preprocess(data, n_bits)
        z, _ = self.discriminator.sample_from_posterior(x, y=y, nsamples=nsamples, random=random)
        return z

    def encode(self, data, y=None, n_bits=8, nsamples=1, random=False):
        size = data.size()
        x = preprocess(data, n_bits)
        # [batch, nsamples, dim]
        z, _ = self.discriminator.sample_from_posterior(x, y=y, nsamples=nsamples, random=random)
        if random:
            # [batch, nsamples, c, h, w]
            u, _ = self.dequantizer.dequantize(x, nsamples=nsamples)
            x = preprocess(data, n_bits, u)
        else:
            # [batch, nsamples, c, h, w]
            x = x.unsqueeze(1) + x.new_zeros(size[0], nsamples, *size[1:])
        # [batch*nsamples, dim]
        zz = z.view(-1, z.size(2)) if z is not None else z
        # [batch*nsamples, c, h, w]
        xx = x.view(-1, *size[1:])
        epsilon = self.generator.encode(xx, h=zz)
        return z, epsilon.view(x.size())

    def decode(self, epsilon, z=None, n_bits=8):
        imgs = self.generator.generate(epsilon, h=z)[0]
        imgs = postprocess(imgs, n_bits=n_bits)
        return imgs

    def forward(self, data, y=None, n_bits=8, nsamples=1):
        # [batch, channels, height, width]
        x = preprocess(data, n_bits)
        # [batch, nsamples, channel, height, width]
        u, log_probs_dequant = self.dequantizer.dequantize(x, nsamples=nsamples)
        # [batch]
        loss_dequant = log_probs_dequant.mean(dim=1)
        # [batch, nsamples, dim]
        z, kl = self.discriminator.sampling_and_KL(x, y=y, nsamples=nsamples)
        # if self.training:
        #     z = z[:, 0] if z is not None else z
        #     u = u[:, 0:1]
        #     # [batch, channels, height, width]
        #     x = preprocess(data, n_bits, u).squeeze(1)
        #     # [batch]
        #     log_probs_gen = self.generator.log_probability(x, h=z)
        # else:
        size = data.size()
        # [batch*nsamples, channels, height, width]
        x = preprocess(data, n_bits, u).view(-1, size[1], size[2], size[3])
        # [batch*nsamples, dim]
        z = z.view(-1, z.size(2)) if z is not None else z
        # [batch]
        log_probs_gen = self.generator.log_probability(x, h=z).view(size[0], nsamples).mean(dim=1)

        loss_gen = log_probs_gen * -1.
        if kl is None:
            kl = loss_gen.new_zeros(loss_gen.size(0))
        return loss_gen, kl, loss_dequant


class WolfModel(nn.Module):
    """
    Variational Auto-Encoding Generative Flow
    """

    def __init__(self, core: WolfCore):
        super(WolfModel, self).__init__()
        self.core = core
        self.distribured_enabled = False

    def _get_core(self):
        return self.core.module if self.distribured_enabled else self.core

    def sync(self):
        core = self._get_core()
        core.sync()

    def init(self, x, y=None, init_scale=1.0):
        core = self._get_core()
        core.init(x, y=y, init_scale=init_scale)

    def init_distributed(self, rank, local_rank):
        assert not self.distribured_enabled
        self.distribured_enabled = True
        print("Initializing Distributed, rank {}, local rank {}".format(rank, local_rank))
        dist.init_process_group(backend='nccl', rank=rank)
        torch.cuda.set_device(local_rank)
        self.core = DistributedDataParallel(convert_syncbn_model(self.core))

    def enable_allreduce(self):
        assert self.distribured_enabled
        self.core.enable_allreduce()

    def disable_allreduce(self):
        assert self.distribured_enabled
        self.core.disable_allreduce()

    def encode_global(self, data, y=None, n_bits=8, nsamples=1, random=False):
        """

        Args:
            data: Tensor [batch, channels, height, width]
                input data
            y: Tensor or None
                class labels for x or None.
            n_bits: int (default 8)
                number of bits for image data.
            nsamples: int (default 1)
                number of samples for each image.
            random: bool (default False)
                incorporating randomness.

        Returns: Tensor
            tensor for global encoding [batch, nsamples, dim] or None

        """
        return self._get_core().encode_global(data, y=y, n_bits=n_bits, nsamples=nsamples, random=random)

    def encode(self, data, y=None, n_bits=8, nsamples=1, random=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            data: Tensor [batch, channels, height, width]
                input data
            y: Tensor or None
                class labels for x or None.
            n_bits: int (default 8)
                number of bits for image data.
            nsamples: int (default 1)
                number of samples for each image.
            random: bool (default False)
                incorporating randomness.

        Returns: Tensor1, Tensor2
            Tensor1: epsilon [batch, channels, height width]
            Tensor2: z [batch, dim] or None

        """
        return self._get_core().encode(data, y=y, n_bits=n_bits, nsamples=nsamples, random=random)

    def decode(self, epsilon, z=None, n_bits=8) -> torch.Tensor:
        """

        Args:
            epsilon: Tensor [batch, channels, height, width]
                epslion for generation
            z: Tensor or None [batch, dim]
                conditional input
            n_bits: int (default 8)
                number of bits for image data.

        Returns: generated tensor [nums, channels, height, width]

        """
        return self._get_core().decode(epsilon, z=z, n_bits=n_bits)

    def synthesize(self, nums, image_size, tau=1.0, n_bits=8, device=torch.device('cpu')) -> torch.Tensor:
        """

        Args:
            nums: int
                number of synthesis
            image_size: size of tuple
                the size of the synthesized images with shape [channels, height, width]
            tau: float (default 1.0)
                temperature
            n_bits: int (default 8)
                number of bits for image data.
            device: torch.device
                device to store the synthesis

        Returns: generated tensor [nums, channels, height, width]

        """
        return self._get_core().synthesize(nums, image_size, tau=tau, n_bits=n_bits, device=device)

    def loss(self, data, y=None, n_bits=8, nsamples=1) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            data: Tensor [batch, channels, height, width]
                input data.
            y: Tensor or None
                class labels for x or None.
            n_bits: int (default 8)
                number of bits for image data.
            nsamples: int (default 1)
                number of samples for compute the loss.

        Returns: Tensor1, Tensor2, Tensor3
            Tensor1: generation loss [batch]
            Tensor2: KL [batch]
            Tensor3: dequantization loss [batch]

        """
        core = self._get_core() if not self.training else self.core
        return core(data, y=y, n_bits=n_bits, nsamples=nsamples)

    def to_device(self, device):
        assert not self.distribured_enabled
        self.core.discriminator.to_device(device)
        return self.to(device)

    def save(self, model_path):
        model = {'core': self._get_core().state_dict()}
        model_name = os.path.join(model_path, 'model.pt')
        torch.save(model, model_name)

    @classmethod
    def load(cls, model_path, device):
        params = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
        flowae = WolfModel.from_params(params).to_device(device)
        model_name = os.path.join(model_path, 'model.pt')
        model = torch.load(model_name, map_location=device)
        flowae.core.load_state_dict(model['core'])
        return flowae

    @classmethod
    def from_params(cls, params: Dict) -> "WolfModel":
        # discriminator
        disc_params = params.pop('discriminator')
        discriminator = Discriminator.by_name(disc_params.pop('type')).from_params(disc_params)
        # dequantizer
        dequant_params = params.pop('dequantizer')
        dequantizer = DeQuantizer.by_name(dequant_params.pop('type')).from_params(dequant_params)
        # generator
        generator_params = params.pop('generator')
        generator = Generator.from_params(generator_params)

        return WolfModel(WolfCore(generator, discriminator, dequantizer))
