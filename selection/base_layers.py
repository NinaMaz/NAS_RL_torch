from itertools import chain

import torch

from .custom_layers import Conv2dSame, MaxPool2dSame, MaybeRescale, PadToSize, PadFeatureMaps
from .select_layers import SelectLayer


def impala_select_layers():
    """ Returns layers for searching based on impala model. """
    layers = []
    n = 9
    for i in range(1, n + 1):
        conv = SelectLayer(Conv2dSame, {"out_channels": [16, 24, 32, 48], "kernel_size": [1, 2, 3, 4, 5]})
        layers.append(conv)
        layers.append(torch.nn.ReLU())
        layers.append(PadToSize(48))
        if i == n // 3:
            layers.append(MaxPool2dSame(kernel_size=3, stride=2))  # torch.nn.MaxPool2D(kernel_size=3, strides=2))
        if i == 2 * n // 3:
            layers.append(MaxPool2dSame(kernel_size=3, stride=2))  # torch.nn.MaxPool2D(kernel_size=3, strides=2))

    layers.append(MaxPool2dSame(kernel_size=3, stride=2))
    layers.append(PadFeatureMaps((12, 12, 48)))
    return layers


def impala_base_layers():
    return ([MaybeRescale(),
             Conv2dSame(in_channels=4, out_channels=24, kernel_size=1),
             torch.nn.ReLU()]
            + impala_select_layers()
            + [torch.nn.Flatten(),
               torch.nn.Linear(in_features=12 * 12 * 48, out_features=256),
               # SelectLayer(torch.nn.Linear, {"out_features": [256]}),
               torch.nn.ReLU(),
               torch.nn.Linear(in_features=256, out_features=256),
               # SelectLayer(torch.nn.Linear, {"out_features": [256]}),
               torch.nn.ReLU()])


def nature_dqn_base_layers():
    return [
        MaybeRescale(),
        torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=3136, out_features=512),
        torch.nn.ReLU()]


# W2=(W1−F+2P)/S+1
# H2=(H1−F+2P)/S+1
# D2=K

def selectable_nature_dqn_base_layers():
    # num_l: number of selectable_layers
    # num_c_i: number of choices per parameter i
    # param_i: number of parameter per layer i

    # Arch space = num_l  * (num_c_0 ** num_c_1 ** num_c_3 **... num_c_n); such that num_c_i > num_c_j for every i>j
    # Arch len = summation_over_i{ param_i }

    kernel_size_choices = [2, 5]

    return [
        [
            # First block for SPOS model
            MaybeRescale(),
            torch.nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            torch.nn.ReLU()
        ],
        # Middle block for SPOS model
        list(chain.from_iterable((
            SelectLayer(Conv2dSame, {'out_channels':[64], 'kernel_size':kernel_size_choices}),
            torch.nn.ReLU())
            for i in range(5))),
        [
            # Last block for the SPOS model

            PadFeatureMaps((11, 11, 64)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=11 * 11 * 64, out_features=512),
            torch.nn.ReLU(),
        ]
    ]


def small_base_layers():
    return ([MaybeRescale(),
             Conv2dSame(in_channels=4, out_channels=24, kernel_size=1),
             torch.nn.ReLU()]
            + [
                SelectLayer(Conv2dSame, {"out_channels": [16, 24, 32, 48], "kernel_size": [1, 2, 3, 4, 5]}),
                torch.nn.ReLU()
            ]
            + [
                PadFeatureMaps((12, 12, 48)),
                torch.nn.Flatten(),
                torch.nn.Linear(in_features=12 * 12 * 48, out_features=256),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=256, out_features=256),
                torch.nn.ReLU()])


def selectable_spos_layers():
    first = [MaybeRescale(),
             Conv2dSame(in_channels=4, out_channels=4, kernel_size=1),
             torch.nn.ReLU(), ]
    blocks = [SelectLayer(Conv2dSame, {"out_channels": [4], "kernel_size": [3, 5, 7], "stride": [1, 2]})]

    last = [torch.nn.ReLU(),
            MaxPool2dSame(kernel_size=3, stride=2),
            PadFeatureMaps((12, 12, 48)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=12 * 12 * 48, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=256),
            torch.nn.ReLU()]

    return first, blocks, last