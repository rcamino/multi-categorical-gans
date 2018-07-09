from __future__ import print_function

import torch.nn as nn


class SingleOutput(nn.Module):

    def __init__(self, previous_layer_size, output_size, activation=None):
        super(SingleOutput, self).__init__()
        if activation is None:
            self.model = nn.Linear(previous_layer_size, output_size)
        else:
            self.model = nn.Sequential(nn.Linear(previous_layer_size, output_size), activation)

    def forward(self, hidden, training=False, temperature=None):
        return self.model(hidden)
