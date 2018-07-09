from __future__ import print_function

import argparse
import torch

import numpy as np

from torch.autograd.variable import Variable
from torch.optim import Adam

from multi_categorical_gans.datasets.dataset import Dataset
from multi_categorical_gans.datasets.formats import data_formats, loaders

from multi_categorical_gans.methods.general.autoencoder import AutoEncoder

from multi_categorical_gans.utils.categorical import load_variable_sizes_from_metadata, categorical_variable_loss
from multi_categorical_gans.utils.commandline import DelayedKeyboardInterrupt, parse_int_list
from multi_categorical_gans.utils.cuda import to_cuda_if_available, to_cpu_if_available
from multi_categorical_gans.utils.initialization import load_or_initialize
from multi_categorical_gans.utils.logger import Logger


def pre_train(autoencoder,
              train_data,
              val_data,
              output_path,
              output_loss_path,
              batch_size=100,
              start_epoch=0,
              num_epochs=100,
              l2_regularization=0.001,
              learning_rate=0.001,
              variable_sizes=None,
              temperature=None
              ):
    autoencoder = to_cuda_if_available(autoencoder)

    optim = Adam(autoencoder.parameters(), weight_decay=l2_regularization, lr=learning_rate)

    logger = Logger(output_loss_path)

    for epoch_index in range(start_epoch, num_epochs):
        logger.start_timer()
        train_loss = pre_train_epoch(autoencoder, train_data, batch_size, optim, variable_sizes, temperature)
        logger.log(epoch_index, num_epochs, "autoencoder", "train_mean_loss", np.mean(train_loss))

        logger.start_timer()
        val_loss = pre_train_epoch(autoencoder, val_data, batch_size, None, variable_sizes, temperature)
        logger.log(epoch_index, num_epochs, "autoencoder", "validation_mean_loss", np.mean(val_loss))

        # save models for the epoch
        with DelayedKeyboardInterrupt():
            torch.save(autoencoder.state_dict(), output_path)
            logger.flush()

    logger.close()


def pre_train_epoch(autoencoder, data, batch_size, optim=None, variable_sizes=None, temperature=None):
    autoencoder.train(mode=(optim is not None))

    training = optim is not None

    losses = []
    for batch in data.batch_iterator(batch_size):
        if optim is not None:
            optim.zero_grad()

        batch = Variable(torch.from_numpy(batch))
        batch = to_cuda_if_available(batch)

        _, batch_reconstructed = autoencoder(batch, training=training, temperature=temperature, normalize_code=False)

        loss = categorical_variable_loss(batch_reconstructed, batch, variable_sizes)
        loss.backward()

        if training:
            optim.step()

        loss = to_cpu_if_available(loss)
        losses.append(loss.data.numpy())
        del loss
    return losses


def losses_by_class_to_string(losses_by_class):
    return ", ".join(["{:.5f}".format(np.mean(losses)) for losses in losses_by_class])


def main():
    options_parser = argparse.ArgumentParser(description="Pre-train MedGAN or MC-MedGAN. "
                                                         + "Define 'metadata' and 'temperature' to use MC-MedGAN.")

    options_parser.add_argument("data", type=str, help="Training data. See 'data_format' parameter.")

    options_parser.add_argument("output_model", type=str, help="Model output file.")
    options_parser.add_argument("output_loss", type=str, help="Loss output file.")

    options_parser.add_argument("--input_model", type=str, help="Model input file.", default=None)

    options_parser.add_argument("--metadata", type=str,
                                help="Information about the categorical variables in json format." +
                                     " Only used if temperature is also provided.")

    options_parser.add_argument(
        "--validation_proportion",
        type=float,
        default=.1,
        help="Ratio of data for validation."
    )

    options_parser.add_argument(
        "--data_format",
        type=str,
        default="sparse",
        choices=data_formats,
        help="Either a dense numpy array, a sparse csr matrix or any of those formats in split into several files."
    )

    options_parser.add_argument(
        "--code_size",
        type=int,
        default=128,
        help="Dimension of the autoencoder latent space."
    )

    options_parser.add_argument(
        "--encoder_hidden_sizes",
        type=str,
        default="",
        help="Size of each hidden layer in the encoder separated by commas (no spaces)."
    )

    options_parser.add_argument(
        "--decoder_hidden_sizes",
        type=str,
        default="",
        help="Size of each hidden layer in the decoder separated by commas (no spaces)."
    )

    options_parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Amount of samples per batch."
    )

    options_parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of epochs."
    )

    options_parser.add_argument(
        "--l2_regularization",
        type=float,
        default=0.001,
        help="L2 regularization weight for every parameter."
    )

    options_parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Adam learning rate."
    )

    options_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Gumbel-Softmax temperature. Only used if metadata is also provided."
    )

    options_parser.add_argument("--seed", type=int, help="Random number generator seed.", default=42)

    options = options_parser.parse_args()

    if options.seed is not None:
        np.random.seed(options.seed)
        torch.manual_seed(options.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(options.seed)

    features = loaders[options.data_format](options.data)
    data = Dataset(features)
    train_data, val_data = data.split(1.0 - options.validation_proportion)

    if options.metadata is not None and options.temperature is not None:
        variable_sizes = load_variable_sizes_from_metadata(options.metadata)
        temperature = options.temperature
    else:
        variable_sizes = None
        temperature = None

    autoencoder = AutoEncoder(
        features.shape[1],
        code_size=options.code_size,
        encoder_hidden_sizes=parse_int_list(options.encoder_hidden_sizes),
        decoder_hidden_sizes=parse_int_list(options.decoder_hidden_sizes),
        variable_sizes=variable_sizes
    )

    load_or_initialize(autoencoder, options.input_model)

    pre_train(
        autoencoder,
        train_data,
        val_data,
        options.output_model,
        options.output_loss,
        batch_size=options.batch_size,
        num_epochs=options.num_epochs,
        l2_regularization=options.l2_regularization,
        learning_rate=options.learning_rate,
        variable_sizes=variable_sizes,
        temperature=temperature
    )


if __name__ == "__main__":
    main()
