from __future__ import print_function

import torch.nn as nn

from multi_categorical_gans.methods.general.multi_categorical import MultiCategorical
from multi_categorical_gans.methods.general.single_output import SingleOutput


class Generator(nn.Module):

    def __init__(self, noise_size, output_size, hidden_sizes=[], bn_decay=0.01):
        super(Generator, self).__init__()

        hidden_activation = nn.ReLU()

        previous_layer_size = noise_size
        hidden_layers = []

        for layer_number, layer_size in enumerate(hidden_sizes):
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))
            if layer_number > 0 and bn_decay > 0:
                hidden_layers.append(nn.BatchNorm1d(layer_size, momentum=(1 - bn_decay)))
            hidden_layers.append(hidden_activation)
            previous_layer_size = layer_size

        if len(hidden_layers) > 0:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        else:
            self.hidden_layers = None

        if type(output_size) is int:
            self.output = SingleOutput(previous_layer_size, output_size)
        elif type(output_size) is list:
            self.output = MultiCategorical(previous_layer_size, output_size)
        else:
            raise Exception("Invalid output size.")

    def forward(self, noise, training=False, temperature=None):
        if self.hidden_layers is None:
            hidden = noise
        else:
            hidden = self.hidden_layers(noise)

        return self.output(hidden, training=training, temperature=temperature)
