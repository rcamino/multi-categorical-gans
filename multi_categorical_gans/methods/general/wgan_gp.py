from __future__ import print_function

import torch

from torch.autograd.variable import Variable

from multi_categorical_gans.utils.cuda import to_cuda_if_available


def calculate_gradient_penalty(discriminator, penalty, real_data, fake_data):
    real_data = real_data.data
    fake_data = fake_data.data

    alpha = torch.rand(len(real_data), 1)
    alpha = alpha.expand(real_data.size())
    alpha = to_cuda_if_available(alpha)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = Variable(interpolates, requires_grad=True)
    discriminator_interpolates = discriminator(interpolates)

    gradients = torch.autograd.grad(outputs=discriminator_interpolates,
                                    inputs=interpolates,
                                    grad_outputs=to_cuda_if_available(torch.ones_like(discriminator_interpolates)),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty
