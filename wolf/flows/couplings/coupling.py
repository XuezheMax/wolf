__author__ = 'max'

from overrides import overrides
from typing import Tuple, Dict
import torch

from wolf.flows.couplings.blocks import NICEConvBlock, MCFBlock, NICEMLPBlock
from wolf.flows.couplings.blocks import LocalLinearCondNet, GlobalLinearCondNet, GlobalAttnCondNet
from wolf.flows.flow import Flow
from wolf.flows.couplings.transform import Additive, Affine, NLSQ, ReLU, SymmELU


class NICE1d(Flow):
    """
    NICE Flow for 1D data
    """
    def __init__(self, in_features, hidden_features=None, inverse=False, split_type='continuous',
                 order='up', transform='affine', alpha=1.0, type='mlp', activation='elu'):
        super(NICE1d, self).__init__(inverse)
        self.in_features = in_features
        self.factor = 2
        assert split_type in ['continuous', 'skip']
        assert in_features % self.factor == 0
        assert order in ['up', 'down']
        self.split_type = split_type
        self.up = order == 'up'

        if hidden_features is None:
            hidden_features = min(8 * in_features, 512)

        out_features = in_features // self.factor
        in_features = in_features - out_features
        self.z1_features = in_features if self.up else out_features

        assert transform in ['additive', 'affine']
        if transform == 'additive':
            self.transform = Additive()
            self.analytic_bwd = True
        elif transform == 'affine':
            self.transform = Affine(dim=-1, alpha=alpha)
            self.analytic_bwd = True
            out_features = out_features * 2
        else:
            raise ValueError('unknown transform: {}'.format(transform))

        assert type in ['mlp']
        if type == 'mlp':
            self.net = NICEMLPBlock(in_features, out_features, hidden_features, activation)

    def split(self, z):
        split_dim = z.dim() - 1
        split_type = self.split_type
        dim = z.size(split_dim)
        if split_type == 'continuous':
            return z.split([self.z1_features, dim - self.z1_features], dim=split_dim)
        elif split_type == 'skip':
            idx1 = torch.tensor(list(range(0, dim, 2))).to(z.device)
            idx2 = torch.tensor(list(range(1, dim, 2))).to(z.device)
            z1 = z.index_select(split_dim, idx1)
            z2 = z.index_select(split_dim, idx2)
            return z1, z2
        else:
            raise ValueError('unknown split type: {}'.format(split_type))

    def unsplit(self, z1, z2):
        split_dim = z1.dim() - 1
        split_type = self.split_type
        if split_type == 'continuous':
            return torch.cat([z1, z2], dim=split_dim)
        elif split_type == 'skip':
            z = torch.cat([z1, z2], dim=split_dim)
            dim = z1.size(split_dim)
            idx = torch.tensor([i // 2 if i % 2 == 0 else i // 2 + dim for i in range(dim * 2)]).to(z.device)
            return z.index_select(split_dim, idx)
        else:
            raise ValueError('unknown split type: {}'.format(split_type))

    def calc_params(self, z: torch.Tensor):
        params = self.net(z)
        return params

    def init_net(self, z: torch.Tensor, init_scale=1.0):
        params = self.net.init(z, init_scale=init_scale)
        return params

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
        # [batch, length, in_channels]
        z1, z2 = self.split(input)
        # [batch, length, features]
        z, zp = (z1, z2) if self.up else (z2, z1)

        params = self.transform.calc_params(self.calc_params(z))
        zp, logdet = self.transform.fwd(zp, params)

        z1, z2 = (z, zp) if self.up else (zp, z)
        return self.unsplit(z1, z2), logdet

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
        if self.analytic_bwd:
            return self.backward_analytic(input)
        else:
            return self.backward_iterative(input)

    def backward_analytic(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, length, in_channels]
        z1, z2 = self.split(z)
        # [batch, length, features]
        z, zp = (z1, z2) if self.up else (z2, z1)

        params = self.transform.calc_params(self.calc_params(z))
        zp, logdet = self.transform.bwd(zp, params)

        z1, z2 = (z, zp) if self.up else (zp, z)
        return self.unsplit(z1, z2), logdet

    def backward_iterative(self, z: torch.Tensor, maxIter=100) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, length, in_channels]
        z1, z2 = self.split(z)
        # [batch, length, features]
        z, zp = (z1, z2) if self.up else (z2, z1)

        params = self.transform.calc_params(self.calc_params(z))
        zp_org = zp
        eps = 1e-6
        for iter in range(maxIter):
            new_zp, logdet = self.transform.bwd(zp, params)
            new_zp = zp_org - new_zp
            diff = torch.abs(new_zp - zp).max().item()
            zp = new_zp
            if diff < eps:
                break

        _, logdet = self.transform.fwd(zp, params)
        z1, z2 = (z, zp) if self.up else (zp, z)
        return self.unsplit(z1, z2), logdet * -1.0

    @overrides
    def init(self, data: torch.Tensor, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # [batch, length, in_channels]
            z1, z2 = self.split(data)
            # [batch, length, features]
            z, zp = (z1, z2) if self.up else (z2, z1)

            params = self.transform.calc_params(self.init_net(z, init_scale=init_scale))
            zp, logdet = self.transform.fwd(zp, params)

            z1, z2 = (z, zp) if self.up else (zp, z)
            return self.unsplit(z1, z2), logdet

    @overrides
    def extra_repr(self):
        return 'inverse={}, in_features={}, split={}, order={}, factor={}, transform={}'.format(self.inverse, self.in_features,
                                                                                                self.split_type, 'up' if self.up else 'down',
                                                                                                self.factor, self.transform)

    @classmethod
    def from_params(cls, params: Dict) -> "NICE1d":
        return NICE1d(**params)


class NICE2d(Flow):
    """
    NICE Flow for 2D image data
    """
    def __init__(self, in_channels, hidden_channels=None, h_channels=0, inverse=False,
                 split_type='continuous', order='up', factor=2, transform='affine', alpha=1.0,
                 type='conv', h_type=None, activation='relu', normalize=None, num_groups=None):
        super(NICE2d, self).__init__(inverse)
        self.in_channels = in_channels
        self.factor = factor
        assert split_type in ['continuous', 'skip']
        if split_type == 'skip':
            assert factor == 2
            if in_channels % factor == 1:
                split_type = 'continuous'
        assert order in ['up', 'down']
        self.split_type = split_type
        self.up = order == 'up'

        if hidden_channels is None:
            hidden_channels = min(8 * in_channels, 512)

        out_channels = in_channels // factor
        in_channels = in_channels - out_channels
        self.z1_channels = in_channels if self.up else out_channels

        assert transform in ['additive', 'affine', 'relu', 'nlsq', 'symm_elu']
        if transform == 'additive':
            self.transform = Additive()
            self.analytic_bwd = True
        elif transform == 'affine':
            self.transform = Affine(dim=1, alpha=alpha)
            self.analytic_bwd = True
            out_channels = out_channels * 2
        elif transform == 'relu':
            self.transform = ReLU(dim=1)
            self.analytic_bwd = True
            out_channels = out_channels * 2
        elif transform == 'nlsq':
            self.transform = NLSQ(dim=1)
            self.analytic_bwd = True
            out_channels = out_channels * 5
        elif transform == 'symm_elu':
            self.transform = SymmELU(dim=1)
            self.analytic_bwd = False
            out_channels = out_channels * 2
        else:
            raise ValueError('unknown transform: {}'.format(transform))

        assert type in ['conv']
        if type == 'conv':
            self.net = NICEConvBlock(in_channels, out_channels, hidden_channels, activation,
                                     normalize=normalize, num_groups=num_groups)

        assert h_type in [None, 'local_linear', 'global_linear', 'global_attn']
        if h_type is None:
            assert h_channels == 0
            self.h_net = None
        elif h_type == 'local_linear':
            self.h_net = LocalLinearCondNet(h_channels, hidden_channels, kernel_size=3)
        elif h_type == 'global_linear':
            self.h_net = GlobalLinearCondNet(h_channels, hidden_channels)
        elif h_type == 'global_attn':
            self.h_net = GlobalAttnCondNet(h_channels, in_channels, hidden_channels)
        else:
            raise ValueError('unknown conditional transform: {}'.format(h_type))

    def split(self, z):
        split_dim = 1
        split_type = self.split_type
        dim = z.size(split_dim)
        if split_type == 'continuous':
            return z.split([self.z1_channels, dim - self.z1_channels], dim=split_dim)
        elif split_type == 'skip':
            idx1 = torch.tensor(list(range(0, dim, 2))).to(z.device)
            idx2 = torch.tensor(list(range(1, dim, 2))).to(z.device)
            z1 = z.index_select(split_dim, idx1)
            z2 = z.index_select(split_dim, idx2)
            return z1, z2
        else:
            raise ValueError('unknown split type: {}'.format(split_type))

    def unsplit(self, z1, z2):
        split_dim = 1
        split_type = self.split_type
        if split_type == 'continuous':
            return torch.cat([z1, z2], dim=split_dim)
        elif split_type == 'skip':
            z = torch.cat([z1, z2], dim=split_dim)
            dim = z1.size(split_dim)
            idx = torch.tensor([i // 2 if i % 2 == 0 else i // 2 + dim for i in range(dim * 2)]).to(z.device)
            return z.index_select(split_dim, idx)
        else:
            raise ValueError('unknown split type: {}'.format(split_type))

    def calc_params(self, z: torch.Tensor, h=None):
        params = self.net(z, h=h)
        return params

    def init_net(self, z: torch.Tensor, h=None, init_scale=1.0):
        params = self.net.init(z, h=h, init_scale=init_scale)
        return params

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]
            h: Tensor
                conditional input (default: None)

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        # [batch, length, in_channels]
        z1, z2 = self.split(input)
        # [batch, length, features]
        z, zp = (z1, z2) if self.up else (z2, z1)

        if self.h_net is not None:
            h = self.h_net(h, x=z)
        else:
            h = None

        params = self.transform.calc_params(self.calc_params(z, h=h))
        zp, logdet = self.transform.fwd(zp, params)

        z1, z2 = (z, zp) if self.up else (zp, z)
        return self.unsplit(z1, z2), logdet

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]
            h: Tensor
                conditional input (default: None)

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        if self.analytic_bwd:
            return self.backward_analytic(input, h=h)
        else:
            return self.backward_iterative(input, h=h)

    def backward_analytic(self, z: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, length, in_channels]
        z1, z2 = self.split(z)
        # [batch, length, features]
        z, zp = (z1, z2) if self.up else (z2, z1)

        if self.h_net is not None:
            h = self.h_net(h, x=z)
        else:
            h = None

        params = self.transform.calc_params(self.calc_params(z, h=h))
        zp, logdet = self.transform.bwd(zp, params)

        z1, z2 = (z, zp) if self.up else (zp, z)
        return self.unsplit(z1, z2), logdet

    def backward_iterative(self, z: torch.Tensor, h=None, maxIter=100) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, length, in_channels]
        z1, z2 = self.split(z)
        # [batch, length, features]
        z, zp = (z1, z2) if self.up else (z2, z1)

        if self.h_net is not None:
            h = self.h_net(h, x=z)
        else:
            h = None

        params = self.transform.calc_params(self.calc_params(z, h=h))
        zp_org = zp
        eps = 1e-6
        for iter in range(maxIter):
            new_zp, logdet = self.transform.bwd(zp, params)
            new_zp = zp_org - new_zp
            diff = torch.abs(new_zp - zp).max().item()
            zp = new_zp
            if diff < eps:
                break

        _, logdet = self.transform.fwd(zp, params)
        z1, z2 = (z, zp) if self.up else (zp, z)
        return self.unsplit(z1, z2), logdet * -1.0

    @overrides
    def init(self, data: torch.Tensor, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # [batch, length, in_channels]
            z1, z2 = self.split(data)
            # [batch, length, features]
            z, zp = (z1, z2) if self.up else (z2, z1)

            if self.h_net is not None:
                h = self.h_net(h, x=z)
            else:
                h = None

            params = self.transform.calc_params(self.init_net(z, h=h, init_scale=init_scale))
            zp, logdet = self.transform.fwd(zp, params)

            z1, z2 = (z, zp) if self.up else (zp, z)
            return self.unsplit(z1, z2), logdet

    @overrides
    def extra_repr(self):
        return 'inverse={}, in_channels={}, split={}, order={}, factor={}, transform={}'.format(self.inverse, self.in_channels,
                                                                                                self.split_type, 'up' if self.up else 'down',
                                                                                                self.factor, self.transform)

    @classmethod
    def from_params(cls, params: Dict) -> "NICE2d":
        return NICE2d(**params)


class MaskedConvFlow(Flow):
    """
    Masked Convolutional Flow
    """

    def __init__(self, in_channels, kernel_size, hidden_channels=None, h_channels=None,
                 h_type=None, activation='relu', order='A', transform='affine', alpha=1.0, inverse=False):
        super(MaskedConvFlow, self).__init__(inverse)
        self.in_channels = in_channels
        if hidden_channels is None:
            if in_channels <= 96:
                hidden_channels = 4 * in_channels
            else:
                hidden_channels = min(2 * in_channels, 512)
        out_channels = in_channels
        assert transform in ['additive', 'affine', 'relu', 'nlsq', 'symm_elu']
        if transform == 'additive':
            self.transform = Additive()
            self.analytic_bwd = True
        elif transform == 'affine':
            self.transform = Affine(dim=1, alpha=alpha)
            self.analytic_bwd = True
            out_channels = out_channels * 2
        elif transform == 'relu':
            self.transform = ReLU(dim=1)
            self.analytic_bwd = True
            out_channels = out_channels * 2
        elif transform == 'nlsq':
            self.transform = NLSQ(dim=1)
            self.analytic_bwd = True
            out_channels = out_channels * 5
        elif transform == 'symm_elu':
            self.transform = SymmELU(dim=1)
            self.analytic_bwd = False
            out_channels = out_channels * 2
        else:
            raise ValueError('unknown transform: {}'.format(transform))
        self.kernel_size = kernel_size
        self.order = order
        self.net = MCFBlock(in_channels, out_channels, kernel_size, hidden_channels, order, activation)

        assert h_type in [None, 'local_linear', 'global_linear', 'global_attn']
        if h_type is None:
            assert h_channels is None or h_channels == 0
            self.h_net = None
        elif h_type == 'local_linear':
            self.h_net = LocalLinearCondNet(h_channels, hidden_channels, kernel_size=3)
        elif h_type == 'global_linear':
            # TODO remove global linear
            self.h_net = GlobalLinearCondNet(h_channels, hidden_channels)
        elif h_type == 'global_attn':
            # TODO add global attn
            self.h_net = None
        else:
            raise ValueError('unknown conditional transform: {}'.format(h_type))

    def calc_params(self, x: torch.Tensor, h=None, shifted=True):
        params = self.net(x, h=h, shifted=shifted)
        return params

    def init_net(self, x, h=None, init_scale=1.0):
        params = self.net.init(x, h=h, init_scale=init_scale)
        return params

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]
            h: Tensor
                conditional input (default: None)

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        if self.h_net is not None:
            h = self.h_net(h)
        else:
            h = None

        params = self.transform.calc_params(self.calc_params(input, h=h))
        out, logdet = self.transform.fwd(input, params)
        return out, logdet

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]
            h: Tensor
                conditional input (default: None)

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        if self.analytic_bwd:
            return self.backward_analytic(input, h=h)
        else:
            return self.backward_iterative(input, h=h)

    def backward_analytic(self, z: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.h_net is not None:
            bs, _, H, W = z.size()
            h = self.h_net(h)
            hh = h + h.new_zeros(bs, 1, H, W)
        else:
            h = hh = None
        if self.order == 'A':
            out = self.backward_height(z, hh=hh, reverse=False)
        elif self.order == 'B':
            out = self.backward_height(z, hh=hh, reverse=True)
        elif self.order == 'C':
            out = self.backward_width(z, hh=hh, reverse=False)
        else:
            out = self.backward_width(z, hh=hh, reverse=True)

        params = self.transform.calc_params(self.calc_params(out, h=h))
        _, logdet = self.transform.fwd(out, params)
        return out, logdet.mul(-1.0)

    def backward_iterative(self, z: torch.Tensor, h=None, maxIter=100) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.h_net is not None:
            h = self.h_net(h)
        else:
            h = None

        z_org = z
        eps = 1e-6
        for iter in range(maxIter):
            params = self.transform.calc_params(self.calc_params(z, h=h))
            new_z, logdet = self.transform.bwd(z, params)
            new_z = z_org - new_z
            diff = torch.abs(new_z - z).max().item()
            z = new_z
            if diff < eps:
                break

        params = self.transform.calc_params(self.calc_params(z, h=h))
        z_recon, logdet = self.transform.fwd(z, params)
        return z, logdet * -1.0

    def backward_height(self, input: torch.Tensor, hh=None, reverse=False) -> torch.Tensor:
        batch, channels, H, W = input.size()

        kH, kW = self.kernel_size
        cW = kW // 2
        out = input.new_zeros(batch, channels, H + kH, W + 2 * cW)

        itr = reversed(range(H)) if reverse else range(H)
        for h in itr:
            curr_h = h if reverse else h + kH
            s_h = h + 1 if reverse else h
            t_h = h + kH + 1 if reverse else h + kH
            # [batch, channels, kH, width+2*cW]
            out_curr = out[:, :, s_h:t_h]
            hh_curr = None if hh is None else hh[:, :, h:h + 1]
            # [batch, channels, width]
            in_curr = input[:, :, h]

            # [batch, channels, 1, width]
            params = self.calc_params(out_curr, h=hh_curr, shifted=False)
            params = self.transform.calc_params(params.squeeze(2))
            # [batch, channels, width]
            new_out, _ = self.transform.bwd(in_curr, params)
            out[:, :, curr_h, cW:W + cW] = new_out

        out = out[:, :, :H, cW:cW + W] if reverse else out[:, :, kH:, cW:cW + W]
        return out

    def backward_width(self, input: torch.Tensor, hh=None, reverse=False) -> torch.Tensor:
        batch, channels, H, W = input.size()

        kH, kW = self.kernel_size
        cH = kH // 2
        out = input.new_zeros(batch, channels, H + 2 * cH, W + kW)

        itr = reversed(range(W)) if reverse else range(W)
        for w in itr:
            curr_w = w if reverse else w + kW
            s_w = w + 1 if reverse else w
            t_w = w + kW + 1 if reverse else w + kW
            # [batch, channels, height+2*cH, kW]
            out_curr = out[:, :, :, s_w:t_w]
            hh_curr = None if hh is None else hh[:, :, :, w:w + 1]
            # [batch, channels, height]
            in_curr = input[:, :, :, w]

            # [batch, channels, height, 1]
            params = self.calc_params(out_curr, h=hh_curr, shifted=False)
            params = self.transform.calc_params(params.squeeze(3))
            # [batch, channels, height]
            new_out, _ = self.transform.bwd(in_curr, params)
            out[:, :, cH:H + cH, curr_w] = new_out

        out = out[:, :, cH:cH + H, :W] if reverse else out[:, :, cH:cH + H, kW:]
        return out

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if self.h_net is not None:
                h = self.h_net(h)
            else:
                h = None

            params = self.transform.calc_params(self.init_net(data, h=h, init_scale=init_scale))
            out, logdet = self.transform.fwd(data, params)
            return out, logdet

    @overrides
    def extra_repr(self):
        return 'inverse={}, in_channels={}, order={}, kernel={}, transform={}'.format(self.inverse, self.in_channels, self.order,
                                                                                      self.kernel_size, self.transform)

    @classmethod
    def from_params(cls, params: Dict) -> "MaskedConvFlow":
        return MaskedConvFlow(**params)


NICE1d.register('nice1d')
NICE2d.register('nice2d')
MaskedConvFlow.register('masc')
