from __future__ import print_function

import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, code_size=128, num_hidden_layers=2, bn_decay=0.01):
        super(Generator, self).__init__()

        self.modules = []
        self.batch_norms = []

        for layer_number in range(num_hidden_layers):
            self.add_generator_module("hidden_{:d}".format(layer_number + 1), code_size, nn.ReLU(), bn_decay)
        self.add_generator_module("output", code_size, nn.Tanh(), bn_decay)

    def add_generator_module(self, name, code_size, activation, bn_decay):
        batch_norm = nn.BatchNorm1d(code_size, momentum=(1 - bn_decay))
        module = nn.Sequential(
            nn.Linear(code_size, code_size, bias=False),  # bias is not necessary because of the batch normalization
            batch_norm,
            activation
        )
        self.modules.append(module)
        self.add_module(name, module)
        self.batch_norms.append(batch_norm)

    def batch_norm_train(self, mode=True):
        for batch_norm in self.batch_norms:
            batch_norm.train(mode=mode)

    def forward(self, noise):
        outputs = noise

        for module in self.modules:
            # Cannot write "outputs += module(outputs)" because it is an inplace operation (no differentiable)
            outputs = module(outputs) + outputs  # shortcut connection
        return outputs
