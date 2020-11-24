import math
from functools import partial
from itertools import product

import torch
from torch.nn import Module
from torch.nn import init

from base.base import Selectable
from .custom_layers import Conv2dSame
from .spaces import MultiDiscrete, SearchSpace, SkipSpace


def reset_parameters(module, scale=2e-4):
    if isinstance(module, torch.nn.Linear):
        init.orthogonal_(module.weight, gain=math.sqrt(2))
        if module.bias is not None:
            init.zeros_(module.bias, )


def instantiate_model(model, x):
    # populate parameters for the optimizer by ensuring that all layers exist
    with torch.no_grad():
        model(x)


def check_partial_class(cls, klass):
    return isinstance(cls, partial) and issubclass(cls.func, klass)


class SelectLayer(Module, Selectable):
    """ Layer with selectable keyword arguments specified by choices. """

    def __init__(self, layer_class, choices, **kwargs):
        super().__init__()
        if set(choices.keys()) & set(kwargs.keys()):
            raise ValueError(f"choices and kwargs cannot have overlapping keys: {set(choices.keys()) & set(kwargs.keys())}")
        self.layer_class = layer_class
        self.choices = choices

        self.prototypes = {}
        for vals in product(*self.choices.values()):
            key = tuple(zip(self.choices.keys(), vals))
            kws = dict(key)
            kws.update(kwargs)
            self.prototypes[key] = partial(layer_class, **kws)

        self.layers = torch.nn.ModuleDict()

        self.space = MultiDiscrete([len(val) for val in self.choices.values()])

    def select(self, selection):
        super().select(selection)
        return self.prototypes[self.get_key(selection)]

    def get_key(self, selection):
        return tuple((key, val[selection[j]]) for j, (key, val) in enumerate(self.choices.items()))

    def forward(self, inputs, selection):
        """ Applies layer to given inputs with specified selection. """
        key = "_".join(map(repr, self.get_key(selection)))
        if key not in self.layers:
            ctor = self.select(selection)
            if check_partial_class(ctor, torch.nn.Conv2d) or check_partial_class(ctor, Conv2dSame):
                inputs_shape = inputs.shape[1]

            elif check_partial_class(ctor, torch.nn.Linear):
                inputs_shape = inputs.shape[-1]

            else:
                klass = getattr(ctor, "func", None)
                raise TypeError(f"Unrecognized module {klass}")

            module = ctor(inputs_shape).to(inputs)
            module.apply(reset_parameters)
            self.layers[key] = module

        return self.layers[key](inputs)


class SelectModelFromLayers(Module):
    def __init__(self, output_feats, base_layers, skip_connections=True):
        super().__init__()
        self.base_layers = torch.nn.ModuleList(base_layers)
        self.skip_connections = skip_connections
        self.outputs_inst = torch.nn.ModuleList()
        if isinstance(output_feats, (list, tuple)):
            self.outputs = [partial(torch.nn.Linear, out_features=feat) for feat in output_feats]
        else:
            self.outputs = partial(torch.nn.Linear, out_features=output_feats)
        self.selection = None
        spaces = []
        for n, layer in enumerate(filter(lambda layer: isinstance(layer, Selectable), self.base_layers)):
            spaces.append(layer.space)
            if self.skip_connections and n:
                spaces.append(SkipSpace(n=1, value=1))
        self.space = SearchSpace(spaces)

    def select(self, selection):
        self.selection = selection
        return selection

    def forward(self, inputs):
        selection = self.selection
        layer_input = inputs
        list_layer_input = [layer_input]
        activations = [layer_input]
        layer_output = [layer_input]

        i = 0
        nskip = 0
        for layer in self.base_layers:

            made_selection = False
            list_layer_input = [layer_output]
            if isinstance(layer, Selectable):
                made_selection = True
                slen = len(layer.space)

                # skip_connections
                nskip = len(activations) - 1

                if made_selection and nskip:
                    list_layer_input.extend(activations[j] for j in range(nskip) if selection[i + slen + j])
                    layer_input = torch.stack(list_layer_input, dim=0).sum(dim=0)

                layer_output = layer(layer_input, selection[i:i + slen])  # .select(selection[i:i + slen])
                i += nskip
                i += slen
            else:
                layer_output = layer(layer_input)

            layer_input = layer_output

            if self.skip_connections:
                if not hasattr(layer, 'weight') and i < len(selection) - 1 and not isinstance(layer, Selectable):
                    activations = [layer(act) for act in activations]

                if made_selection:
                    activations.append(layer_output)

        if len(self.outputs_inst) == 0:
            if isinstance(self.outputs, list):
                for out in self.outputs:
                    self.outputs_inst.append(out(in_features=layer_output.shape[-1]))
            else:
                self.outputs_inst.append(self.outputs(in_features=layer_output.shape[-1]))
            self.outputs_inst.to(inputs)

        outputs = [out(layer_output) for out in self.outputs_inst]
        return outputs
