from __future__ import print_function

import torch.nn as nn

from multi_categorical_gans.methods.general.multi_categorical import MultiCategorical
from multi_categorical_gans.methods.general.single_output import SingleOutput


class Decoder(nn.Module):

    def __init__(self, code_size, output_size, hidden_sizes=[]):
        super(Decoder, self).__init__()

        hidden_activation = nn.Tanh()

        previous_layer_size = code_size
        hidden_layers = []

        for layer_size in hidden_sizes:
            hidden_layers.append(nn.Linear(previous_layer_size, layer_size))
            hidden_layers.append(hidden_activation)
            previous_layer_size = layer_size

        if len(hidden_layers) > 0:
            self.hidden_layers = nn.Sequential(*hidden_layers)
        else:
            self.hidden_layers = None

        if type(output_size) is int:
            self.output_layer = SingleOutput(previous_layer_size, output_size, activation=nn.Sigmoid())
        elif type(output_size) is list:
            self.output_layer = MultiCategorical(previous_layer_size, output_size)
        else:
            raise Exception("Invalid output size.")

    def forward(self, code, training=False, temperature=None):
        if self.hidden_layers is None:
            hidden = code
        else:
            hidden = self.hidden_layers(code)

        return self.output_layer(hidden, training=training, temperature=temperature)
