import torch
from torch.nn import Module


class Conv2dSame(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True,
                 padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)
        )

    def forward(self, x):
        """
        print(x.size(),
              'in_channels:', self.in_channels,
              'out_channels:', self.out_channels,
              'kernel_size:', self.kernel_size,
              'stride:', self.stride, )
        """
        return self.net(x)


class MaxPool2dSame(Module):
    def __init__(self, kernel_size, stride=None, dilation=1, return_indices=False, ceil_mode=False,
                 padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.MaxPool2d(kernel_size, stride)
        )

    def forward(self, x):
        return self.net(x)


class MaybeRescale(Module):
    def __init__(self, rescale=True):
        super().__init__()
        self.rescale = rescale

    def forward(self, inputs):
        if self.rescale:
            outputs = inputs / 255
        else:
            outputs = inputs
        return outputs


""" Layers useful for searching. """


def pad_to_size(tensor, sizes, axes=-1, before_and_after=False,
                mode='constant', constant_values=0):
    """ Pads tensor to a given size along specified axis. """
    if isinstance(sizes, int):
        sizes = [sizes]
    if isinstance(axes, int):
        axes = [axes]
    for i, ax in enumerate(axes):
        if ax < 0:
            axes[i] = len(tensor.shape) + ax
    size_deltas = [sizes[i] - tensor.shape[ax] for i, ax in enumerate(axes)]

    size_paddings = {ax: (delta // (1 + before_and_after), (delta + 1) // 2 * before_and_after)
                     for ax, delta in zip(axes, size_deltas)}
    paddings = []

    for i in [3, 2, 1, 0]:
        tup = size_paddings.get(i, (0, 0))
        paddings.extend(tup)
    return torch.nn.functional.pad(tensor, paddings, mode=mode, value=constant_values)


class PadToSize(Module):
    """ layer for padding tensors to a given size. """

    def __init__(self, sizes, axes=1, before_and_after=False, mode="constant", constant_values=0):
        super().__init__()
        self.sizes = sizes
        self.axes = axes
        self.before_and_after = before_and_after
        self.mode = mode
        self.constant_values = constant_values

    def forward(self, inputs):
        return pad_to_size(inputs, sizes=self.sizes, axes=self.axes, before_and_after=self.before_and_after,
                           mode=self.mode, constant_values=self.constant_values)


class PadFeatureMaps(Module):
    """ Pads a tensor consisting of a batch of images. """

    def __init__(self, sizes):
        self.sizes = sizes
        if len(self.sizes) != 3:
            raise ValueError("sizes must be of length 3, " f"got len(sizes)={len(self.sizes)}")
        super().__init__()
        self.l1 = PadToSize(sizes[:2], axes=(1, 2), before_and_after=True)  # 1,2
        self.l2 = PadToSize(sizes[-1], axes=3)  # 3

    def forward(self, inputs):
        x = self.l1(inputs)
        x = self.l2(x)
        return x
