from __future__ import print_function

import torch
import torch.nn as nn

from multi_categorical_gans.methods.general.decoder import Decoder
from multi_categorical_gans.methods.general.encoder import Encoder


class AutoEncoder(nn.Module):

    def __init__(self, input_size, code_size=128, encoder_hidden_sizes=[], decoder_hidden_sizes=[],
                 variable_sizes=None):

        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(input_size,
                               code_size,
                               hidden_sizes=encoder_hidden_sizes)

        self.decoder = Decoder(code_size,
                               (input_size if variable_sizes is None else variable_sizes),
                               hidden_sizes=decoder_hidden_sizes)

    def forward(self, inputs, normalize_code=False, training=False, temperature=None):
        code = self.encode(inputs, normalize_code=normalize_code)
        reconstructed = self.decode(code, training=training, temperature=temperature)
        return code, reconstructed

    def encode(self, inputs, normalize_code=False):
        code = self.encoder(inputs)

        if normalize_code:
            norms = torch.norm(code, 2, 1)
            code = torch.div(code, norms.unsqueeze(1).expand_as(code))

        return code

    def decode(self, code, training=False, temperature=None):
        return self.decoder(code, training=training, temperature=temperature)
