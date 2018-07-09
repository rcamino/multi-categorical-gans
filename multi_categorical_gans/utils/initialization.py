import torch.nn as nn

from multi_categorical_gans.utils.cuda import load_without_cuda


def initialize_weights(module):
    if type(module) == nn.Linear:
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif type(module) == nn.BatchNorm1d:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)


def load_or_initialize(module, state_dict_path):
    if state_dict_path is not None:
        load_without_cuda(module, state_dict_path)
    else:
        module.apply(initialize_weights)
