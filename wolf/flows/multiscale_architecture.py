from overrides import overrides
from typing import Tuple
import torch
import torch.nn as nn

from wolf.flows.flow import Flow
from wolf.flows.couplings import NICE2d
from wolf.flows.permutation import Conv1x1Flow
from wolf.flows.normalization import ActNorm2dFlow
from wolf.utils import squeeze2d, unsqueeze2d, split2d, unsplit2d


class MultiScalePrior(Flow):
    """
    Prior in multi-scale architecture
    """
    def __init__(self, in_channels, hidden_channels, h_channels, factor, transform, alpha,
                 inverse, coupling_type, h_type, activation, normalize, num_groups):
        super(MultiScalePrior, self).__init__(inverse)
        self.conv1x1 = Conv1x1Flow(in_channels, inverse=inverse)
        self.coupling = NICE2d(in_channels, hidden_channels=hidden_channels, h_channels=h_channels,
                               transform=transform, alpha=alpha, inverse=inverse, factor=factor,
                               type=coupling_type, h_type=h_type, split_type='continuous', order='up',
                               activation=activation, normalize=normalize, num_groups=num_groups)
        out_channels = in_channels // factor
        self.z1_channels = self.coupling.z1_channels
        assert out_channels + self.z1_channels == in_channels
        self.actnorm = ActNorm2dFlow(out_channels, inverse=inverse)

    def sync(self):
        self.conv1x1.sync()

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # conv1x1
        out, logdet_accum = self.conv1x1.forward(input)
        # coupling
        out, logdet = self.coupling.forward(out, h=h)
        logdet_accum = logdet_accum + logdet
        # actnorm
        out1, out2 = split2d(out, self.z1_channels)
        out2, logdet = self.actnorm.forward(out2)
        logdet_accum = logdet_accum + logdet
        out = unsplit2d([out1, out2])
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # actnorm
        out1, out2 = split2d(input, self.z1_channels)
        out2, logdet_accum = self.actnorm.backward(out2)
        out = unsplit2d([out1, out2])
        # coupling
        out, logdet = self.coupling.backward(out, h=h)
        logdet_accum = logdet_accum + logdet
        # conv1x1
        out, logdet = self.conv1x1.backward(out)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # conv1x1
        out, logdet_accum = self.conv1x1.init(data, init_scale=init_scale)
        # coupling
        out, logdet = self.coupling.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        # actnorm
        out1, out2 = split2d(out, self.z1_channels)
        out2, logdet = self.actnorm.init(out2, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        out = unsplit2d([out1, out2])
        return out, logdet_accum


class MultiScaleExternal(Flow):
    """
    Multi-scale architecture external block (bottom or top).
    """
    def __init__(self, flow_step, num_steps, in_channels, hidden_channels, h_channels,
                 transform='affine', alpha=1.0, inverse=False, kernel_size=(2, 3),
                 coupling_type='conv', h_type=None, activation='relu', normalize=None, num_groups=None):
        super(MultiScaleExternal, self).__init__(inverse)
        steps = [flow_step(in_channels, hidden_channels=hidden_channels, h_channels=h_channels,
                           transform=transform, alpha=alpha, inverse=inverse, coupling_type=coupling_type,
                           h_type=h_type, activation=activation, normalize=normalize, num_groups=num_groups,
                           kernel_size=kernel_size)
                 for _ in range(num_steps)]

        self.steps = nn.ModuleList(steps)

    def sync(self):
        for step in self.steps:
            step.sync()

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out = input
        # [batch]
        logdet_accum = input.new_zeros(input.size(0))
        for step in self.steps:
            out, logdet = step.forward(out, h=h)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        for step in reversed(self.steps):
            out, logdet = step.backward(out, h=h)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out = data
        # [batch]
        logdet_accum = data.new_zeros(data.size(0))
        for step in self.steps:
            out, logdet = step.init(out, h=h, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class MultiScaleInternal(Flow):
    """
    Multi-scale architecture internal block.
    """
    def __init__(self, flow_step, num_steps, in_channels, hidden_channels, h_channels,
                 factor=2, transform='affine', prior_transform='affine', alpha=1.0,
                 inverse=False, kernel_size=(2, 3), coupling_type='conv', h_type=None,
                 activation='relu', normalize=None, num_groups=None):
        super(MultiScaleInternal, self).__init__(inverse)
        num_layers = len(num_steps)
        assert num_layers < factor
        self.layers = nn.ModuleList()
        self.priors = nn.ModuleList()
        channel_step = in_channels // factor
        for num_step in num_steps:
            layer = [flow_step(in_channels, hidden_channels=hidden_channels, h_channels=h_channels,
                               transform=transform, alpha=alpha, inverse=inverse, coupling_type=coupling_type, h_type=h_type,
                               activation=activation, normalize=normalize, num_groups=num_groups, kernel_size=kernel_size)
                     for _ in range(num_step)]
            self.layers.append(nn.ModuleList(layer))
            prior = MultiScalePrior(in_channels, hidden_channels=hidden_channels, h_channels=h_channels,
                                    transform=prior_transform, alpha=alpha, inverse=inverse, factor=factor,
                                    coupling_type=coupling_type, h_type=h_type, activation=activation,
                                    normalize=normalize, num_groups=num_groups)
            self.priors.append(prior)
            in_channels = in_channels - channel_step
            assert in_channels == prior.z1_channels
            factor = factor - 1
        self.z_channels = in_channels
        assert len(self.layers) == len(self.priors)

    def sync(self):
        for layer, prior in zip(self.layers, self.priors):
            for step in layer:
                step.sync()
            prior.sync()

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out = input
        # [batch]
        logdet_accum = input.new_zeros(input.size(0))
        outputs = []
        for layer, prior in zip(self.layers, self.priors):
            for step in layer:
                out, logdet = step.forward(out, h=h)
                logdet_accum = logdet_accum + logdet
            out, logdet = prior.forward(out, h=h)
            logdet_accum = logdet_accum + logdet
            # split
            out1, out2 = split2d(out, prior.z1_channels)
            outputs.append(out2)
            out = out1

        outputs.append(out)
        outputs.reverse()
        out = unsplit2d(outputs)
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out = input
        outputs = []
        for prior in self.priors:
            out1, out2 = split2d(out, prior.z1_channels)
            outputs.append(out2)
            out = out1

        # [batch]
        logdet_accum = out.new_zeros(out.size(0))
        for layer, prior in zip(reversed(self.layers), reversed(self.priors)):
            out2 = outputs.pop()
            out = unsplit2d([out, out2])
            out, logdet = prior.backward(out, h=h)
            logdet_accum = logdet_accum + logdet
            for step in reversed(layer):
                out, logdet = step.backward(out, h=h)
                logdet_accum = logdet_accum + logdet

        assert len(outputs) == 0
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out = data
        # [batch]
        logdet_accum = data.new_zeros(data.size(0))
        outputs = []
        for layer, prior in zip(self.layers, self.priors):
            for step in layer:
                out, logdet = step.init(out, h=h, init_scale=init_scale)
                logdet_accum = logdet_accum + logdet
            out, logdet = prior.init(out, h=h, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
            # split
            out1, out2 = split2d(out, prior.z1_channels)
            outputs.append(out2)
            out = out1

        outputs.append(out)
        outputs.reverse()
        out = unsplit2d(outputs)
        return out, logdet_accum


class MultiScaleArchitecture(Flow):
    """
    Multi-scale Architecture.
    """

    def __init__(self, flow_step, levels, num_steps, in_channels, factors, hidden_channels,
                 h_channels=0, inverse=False, transform='affine', prior_transform='affine',
                 alpha=1.0, kernel_size=None, coupling_type='conv', h_type=None,
                 activation='relu', normalize=None, num_groups=None):
        super(MultiScaleArchitecture, self).__init__(inverse)
        assert levels > 1, 'Multi-scale architecture should have at least 2 levels.'
        assert levels == len(num_steps)
        factors = [0] + factors + [0]
        assert levels == len(factors)
        assert levels == len(hidden_channels)
        if normalize == 'group_norm':
            assert levels == len(num_groups)

        blocks = []
        self.levels = levels
        self.internals = levels - 2
        self.squeeze_h = h_type is not None and h_type.startswith('local')

        for level in range(levels):
            hidden_channel = hidden_channels[level]
            n_groups = num_groups[level] if normalize == 'group_norm' else None
            if level == 0:
                # bottom
                block = MultiScaleExternal(flow_step, num_steps[level], in_channels,
                                           hidden_channels=hidden_channel, h_channels=h_channels,
                                           transform=transform, alpha=alpha, inverse=inverse,
                                           kernel_size=kernel_size, coupling_type=coupling_type, h_type=h_type,
                                           activation=activation, normalize=normalize, num_groups=n_groups)
                blocks.append(block)
            elif level == levels - 1:
                # top
                in_channels = in_channels * 4
                if self.squeeze_h:
                    h_channels = h_channels * 4
                block = MultiScaleExternal(flow_step, num_steps[level], in_channels,
                                           hidden_channels=hidden_channel, h_channels=h_channels,
                                           transform=transform, alpha=alpha, inverse=inverse,
                                           kernel_size=kernel_size, coupling_type=coupling_type, h_type=h_type,
                                           activation=activation, normalize=normalize, num_groups=n_groups)
                blocks.append(block)
            else:
                # internal
                in_channels = in_channels * 4
                if self.squeeze_h:
                    h_channels = h_channels * 4
                block = MultiScaleInternal(flow_step, num_steps[level], in_channels,
                                           hidden_channels=hidden_channel, h_channels=h_channels,
                                           factor=factors[level], inverse=inverse, kernel_size=kernel_size,
                                           transform=transform, prior_transform=prior_transform,
                                           alpha=alpha, coupling_type=coupling_type, h_type=h_type,
                                           activation=activation, normalize=normalize, num_groups=n_groups)
                blocks.append(block)
                in_channels = block.z_channels
        self.blocks = nn.ModuleList(blocks)

    def sync(self):
        for block in self.blocks:
            block.sync()

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        outputs = []
        for i, block in enumerate(self.blocks):
            out, logdet = block.forward(out, h=h)
            logdet_accum = logdet_accum + logdet
            if i < self.levels - 1:
                if i > 0:
                    # split when block is not bottom or top
                    out1, out2 = split2d(out, block.z_channels)
                    outputs.append(out2)
                    out = out1
                # squeeze when block is not top
                out = squeeze2d(out, factor=2)
                if self.squeeze_h:
                    h = squeeze2d(h, factor=2)

        out = unsqueeze2d(out, factor=2)
        for _ in range(self.internals):
            out2 = outputs.pop()
            out = unsqueeze2d(unsplit2d([out, out2]), factor=2)
        assert len(outputs) == 0
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        out = input
        for i in range(self.levels - 1):
            if i > 0:
                out1, out2 = split2d(out, self.blocks[i].z_channels)
                outputs.append(out2)
                out = out1
            out = squeeze2d(out, factor=2)
            if self.squeeze_h:
                h = squeeze2d(h, factor=2)

        logdet_accum = input.new_zeros(input.size(0))
        for i, block in enumerate(reversed(self.blocks)):
            if i > 0:
                out = unsqueeze2d(out, factor=2)
                if self.squeeze_h:
                    h = unsqueeze2d(h, factor=2)
                if i < self.levels - 1:
                    out2 = outputs.pop()
                    out = unsplit2d([out, out2])
            out, logdet = block.backward(out, h=h)
            logdet_accum = logdet_accum + logdet
        assert len(outputs) == 0
        return out, logdet_accum

    @overrides
    def init(self, data: torch.Tensor, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = data.new_zeros(data.size(0))
        out = data
        outputs = []
        for i, block in enumerate(self.blocks):
            out, logdet = block.init(out, h=h, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
            if i < self.levels - 1:
                if i > 0:
                    # split when block is not bottom or top
                    out1, out2 = split2d(out, block.z_channels)
                    outputs.append(out2)
                    out = out1
                # squeeze when block is not top
                out = squeeze2d(out, factor=2)
                if self.squeeze_h:
                    h = squeeze2d(h, factor=2)

        out = unsqueeze2d(out, factor=2)
        for _ in range(self.internals):
            out2 = outputs.pop()
            out = unsqueeze2d(unsplit2d([out, out2]), factor=2)
        assert len(outputs) == 0
        return out, logdet_accum
