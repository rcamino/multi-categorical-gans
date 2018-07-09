from __future__ import print_function

import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_sizes=(256, 128)):
        super(Discriminator, self).__init__()

        hidden_activation = nn.LeakyReLU()

        previous_layer_size = input_size * 2
        layers = []

        for layer_size in hidden_sizes:
            layers.append(nn.Linear(previous_layer_size, layer_size))
            layers.append(hidden_activation)
            previous_layer_size = layer_size

        layers.append(nn.Linear(previous_layer_size, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def minibatch_averaging(self, inputs):
        """
        This method is explained in the MedGAN paper.
        """
        mean_per_feature = torch.mean(inputs, 0)
        mean_per_feature_repeated = mean_per_feature.repeat(len(inputs), 1)
        return torch.cat((inputs, mean_per_feature_repeated), 1)

    def forward(self, inputs):
        inputs = self.minibatch_averaging(inputs)
        return self.model(inputs).view(-1)
