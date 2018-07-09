from __future__ import print_function

import argparse
import torch

import numpy as np

from torch.autograd.variable import Variable

from multi_categorical_gans.methods.general.generator import Generator

from multi_categorical_gans.utils.categorical import load_variable_sizes_from_metadata
from multi_categorical_gans.utils.commandline import parse_int_list
from multi_categorical_gans.utils.cuda import to_cuda_if_available, to_cpu_if_available, load_without_cuda


def sample(generator, temperature, num_samples, num_features, batch_size=100, noise_size=128):
    generator = to_cuda_if_available(generator)

    generator.train(mode=False)

    samples = np.zeros((num_samples, num_features), dtype=np.float32)

    start = 0
    while start < num_samples:
        with torch.no_grad():
            noise = Variable(torch.FloatTensor(batch_size, noise_size).normal_())
            noise = to_cuda_if_available(noise)
            batch_samples = generator(noise, training=False, temperature=temperature)
        batch_samples = to_cpu_if_available(batch_samples)
        batch_samples = batch_samples.data.numpy()

        # do not go further than the desired number of samples
        end = min(start + batch_size, num_samples)
        # limit the samples taken from the batch based on what is missing
        samples[start:end, :] = batch_samples[:min(batch_size, end - start), :]

        # move to next batch
        start = end
    return samples


def main():
    options_parser = argparse.ArgumentParser(description="Sample data with MedGAN.")

    options_parser.add_argument("generator", type=str, help="Generator input file.")

    options_parser.add_argument("metadata", type=str,
                                help="Information about the categorical variables in json format.")

    options_parser.add_argument("num_samples", type=int, help="Number of output samples.")
    options_parser.add_argument("num_features", type=int, help="Number of output features.")
    options_parser.add_argument("data", type=str, help="Output data.")

    options_parser.add_argument(
        "--noise_size",
        type=int,
        default=128,
        help="Dimension of the generator input noise."
    )

    options_parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Amount of samples per batch."
    )

    options_parser.add_argument(
        "--generator_hidden_sizes",
        type=str,
        default="256,128",
        help="Size of each hidden layer in the generator separated by commas (no spaces)."
    )

    options_parser.add_argument(
        "--generator_bn_decay",
        type=float,
        default=0.01,
        help="Generator batch normalization decay."
    )

    options_parser.add_argument(
        "--temperature",
        type=float,
        default=0.666,
        help="Gumbel-Softmax temperature."
    )

    options = options_parser.parse_args()

    generator = Generator(
        options.noise_size,
        load_variable_sizes_from_metadata(options.metadata),
        hidden_sizes=parse_int_list(options.generator_hidden_sizes),
        bn_decay=options.generator_bn_decay
    )

    load_without_cuda(generator, options.generator)

    data = sample(
        generator,
        options.temperature,
        options.num_samples,
        options.num_features,
        batch_size=options.batch_size,
        noise_size=options.noise_size,
    )

    np.save(options.data, data)


if __name__ == "__main__":
    main()
