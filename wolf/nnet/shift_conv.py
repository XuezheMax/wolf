__author__ = 'max'

from overrides import overrides
import torch.nn as nn
import torch.nn.functional as F


class ShiftedConv2d(nn.Conv2d):
    """
    Conv2d with shift operation.
    A -> top
    B -> bottom
    C -> left
    D -> right
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), dilation=1, groups=1, bias=True, order='A'):
        assert len(stride) == 2
        assert len(kernel_size) == 2
        assert order in {'A', 'B', 'C', 'D'}, 'unknown order: {}'.format(order)
        if order in {'A', 'B'}:
            assert kernel_size[1] % 2 == 1, 'kernel width cannot be even number: {}'.format(kernel_size)
        else:
            assert kernel_size[0] % 2 == 1, 'kernel height cannot be even number: {}'.format(kernel_size)

        self.order = order
        if order == 'A':
            # left, right, top, bottom
            self.shift_padding = ((kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2, kernel_size[0], 0)
            # top, bottom, left, right
            self.cut = (0, -1, 0, 0)
        elif order == 'B':
            # left, right, top, bottom
            self.shift_padding = ((kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2, 0, kernel_size[0])
            # top, bottom, left, right
            self.cut = (1, 0, 0, 0)
        elif order == 'C':
            # left, right, top, bottom
            self.shift_padding = (kernel_size[1], 0, (kernel_size[0] - 1) // 2, (kernel_size[0] - 1) // 2)
            # top, bottom, left, right
            self.cut = (0, 0, 0, -1)
        elif order == 'D':
            # left, right, top, bottom
            self.shift_padding = (0, kernel_size[1], (kernel_size[0] - 1) // 2, (kernel_size[0] - 1) // 2)
            # top, bottom, left, right
            self.cut = (0, 0, 1, 0)
        else:
            self.shift_padding = None
            raise ValueError('unknown order: {}'.format(order))

        super(ShiftedConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=0,
                                            stride=stride, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input, shifted=True):
        if shifted:
            input = F.pad(input, self.shift_padding)
            bs, channels, height, width = input.size()
            t, b, l, r = self.cut
            input = input[:, :, t:height + b, l:width + r]
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    @overrides
    def extra_repr(self):
        s = super(ShiftedConv2d, self).extra_repr()
        s += ', order={order}'
        s += ', shift_padding={shift_padding}'
        s += ', cut={cut}'
        return s.format(**self.__dict__)
