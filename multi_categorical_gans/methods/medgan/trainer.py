from __future__ import print_function

import argparse
import torch

import numpy as np

from torch.autograd.variable import Variable
from torch.optim import Adam
from torch.nn import BCELoss

from multi_categorical_gans.datasets.dataset import Dataset
from multi_categorical_gans.datasets.formats import data_formats, loaders

from multi_categorical_gans.methods.general.autoencoder import AutoEncoder
from multi_categorical_gans.methods.medgan.discriminator import Discriminator
from multi_categorical_gans.methods.medgan.generator import Generator

from multi_categorical_gans.utils.categorical import load_variable_sizes_from_metadata
from multi_categorical_gans.utils.commandline import DelayedKeyboardInterrupt, parse_int_list
from multi_categorical_gans.utils.cuda import to_cuda_if_available, to_cpu_if_available, load_without_cuda
from multi_categorical_gans.utils.initialization import load_or_initialize
from multi_categorical_gans.utils.logger import Logger


def train(autoencoder,
          generator,
          discriminator,
          train_data,
          val_data,
          output_ae_path,
          output_gen_path,
          output_disc_path,
          output_loss_path,
          batch_size=1000,
          start_epoch=0,
          num_epochs=1000,
          num_disc_steps=2,
          num_gen_steps=1,
          code_size=128,
          l2_regularization=0.001,
          learning_rate=0.001,
          temperature=None
          ):
    autoencoder, generator, discriminator = to_cuda_if_available(autoencoder, generator, discriminator)

    optim_gen = Adam(list(generator.parameters()) + list(autoencoder.decoder.parameters()),
                     weight_decay=l2_regularization, lr=learning_rate)

    optim_disc = Adam(discriminator.parameters(), weight_decay=l2_regularization, lr=learning_rate)

    criterion = BCELoss()

    logger = Logger(output_loss_path, append=start_epoch > 0)

    for epoch_index in range(start_epoch, num_epochs):
        logger.start_timer()

        # train
        autoencoder.train(mode=True)
        generator.train(mode=True)
        discriminator.train(mode=True)

        disc_losses = []
        gen_losses = []

        more_batches = True
        train_data_iterator = train_data.batch_iterator(batch_size)

        while more_batches:
            # train discriminator
            generator.batch_norm_train(mode=False)

            for _ in range(num_disc_steps):
                # next batch
                try:
                    batch = next(train_data_iterator)
                except StopIteration:
                    more_batches = False
                    break

                # using "one sided smooth labels" is one trick to improve GAN training
                label_zeros = Variable(torch.zeros(len(batch)))
                smooth_label_ones = Variable(torch.FloatTensor(len(batch)).uniform_(0.9, 1))

                label_zeros, smooth_label_ones = to_cuda_if_available(label_zeros, smooth_label_ones)

                optim_disc.zero_grad()

                # first train the discriminator only with real data
                real_features = Variable(torch.from_numpy(batch))
                real_features = to_cuda_if_available(real_features)
                real_pred = discriminator(real_features)
                real_loss = criterion(real_pred, smooth_label_ones)
                real_loss.backward()

                # then train the discriminator only with fake data
                noise = Variable(torch.FloatTensor(len(batch), code_size).normal_())
                noise = to_cuda_if_available(noise)
                fake_code = generator(noise)
                fake_features = autoencoder.decode(fake_code, training=True, temperature=temperature)
                fake_features = fake_features.detach()  # do not propagate to the generator
                fake_pred = discriminator(fake_features)
                fake_loss = criterion(fake_pred, label_zeros)
                fake_loss.backward()

                # finally update the discriminator weights
                # using two separated batches is another trick to improve GAN training
                optim_disc.step()

                disc_loss = real_loss + fake_loss
                disc_loss = to_cpu_if_available(disc_loss)
                disc_losses.append(disc_loss.data.numpy())

                del disc_loss
                del fake_loss
                del real_loss

            # train generator
            generator.batch_norm_train(mode=True)

            for _ in range(num_gen_steps):
                optim_gen.zero_grad()

                noise = Variable(torch.FloatTensor(len(batch), code_size).normal_())
                noise = to_cuda_if_available(noise)
                gen_code = generator(noise)
                gen_features = autoencoder.decode(gen_code, training=True, temperature=temperature)
                gen_pred = discriminator(gen_features)

                smooth_label_ones = Variable(torch.FloatTensor(len(batch)).uniform_(0.9, 1))
                smooth_label_ones = to_cuda_if_available(smooth_label_ones)

                gen_loss = criterion(gen_pred, smooth_label_ones)
                gen_loss.backward()

                optim_gen.step()

                gen_loss = to_cpu_if_available(gen_loss)
                gen_losses.append(gen_loss.data.numpy())

                del gen_loss

        # validate discriminator
        autoencoder.train(mode=False)
        generator.train(mode=False)
        discriminator.train(mode=False)

        correct = 0.0
        total = 0.0
        for batch in val_data.batch_iterator(batch_size):
            # real data discriminator accuracy
            with torch.no_grad():
                real_features = Variable(torch.from_numpy(batch))
                real_features = to_cuda_if_available(real_features)
                real_pred = discriminator(real_features)
            real_pred = to_cpu_if_available(real_pred)
            correct += (real_pred.data.numpy().ravel() > .5).sum()
            total += len(real_pred)

            # fake data discriminator accuracy
            with torch.no_grad():
                noise = Variable(torch.FloatTensor(len(batch), code_size).normal_())
                noise = to_cuda_if_available(noise)
                fake_code = generator(noise)
                fake_features = autoencoder.decode(fake_code, training=False, temperature=temperature)
                fake_pred = discriminator(fake_features)
            fake_pred = to_cpu_if_available(fake_pred)
            correct += (fake_pred.data.numpy().ravel() < .5).sum()
            total += len(fake_pred)

        # log epoch metrics for current class
        logger.log(epoch_index, num_epochs, "discriminator", "train_mean_loss", np.mean(disc_losses))
        logger.log(epoch_index, num_epochs, "generator", "train_mean_loss", np.mean(gen_losses))
        logger.log(epoch_index, num_epochs, "discriminator", "validation_accuracy", correct / total)

        # save models for the epoch
        with DelayedKeyboardInterrupt():
            torch.save(autoencoder.state_dict(), output_ae_path)
            torch.save(generator.state_dict(), output_gen_path)
            torch.save(discriminator.state_dict(), output_disc_path)
            logger.flush()

    logger.close()


def main():
    options_parser = argparse.ArgumentParser(description="Train MedGAN or MC-MedGAN. "
                                                         + "Define 'metadata' and 'temperature' to use MC-MedGAN.")

    options_parser.add_argument("data", type=str, help="Training data. See 'data_format' parameter.")

    options_parser.add_argument("input_autoencoder", type=str, help="Autoencoder input file.")
    options_parser.add_argument("output_autoencoder", type=str, help="Autoencoder output file.")
    options_parser.add_argument("output_generator", type=str, help="Generator output file.")
    options_parser.add_argument("output_discriminator", type=str, help="Discriminator output file.")
    options_parser.add_argument("output_loss", type=str, help="Loss output file.")

    options_parser.add_argument("--input_generator", type=str, help="Generator input file.", default=None)
    options_parser.add_argument("--input_discriminator", type=str, help="Discriminator input file.", default=None)

    options_parser.add_argument("--metadata", type=str,
                                help="Information about the categorical variables in json format." +
                                     " Only used if temperature is also provided.")

    options_parser.add_argument(
        "--validation_proportion", type=float,
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
        default=1000,
        help="Amount of samples per batch."
    )

    options_parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="Starting epoch."
    )

    options_parser.add_argument(
        "--num_epochs",
        type=int,
        default=1000,
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
        "--generator_hidden_layers",
        type=int,
        default=2,
        help="Number of hidden layers in the generator."
    )

    options_parser.add_argument(
        "--generator_bn_decay",
        type=float,
        default=0.99,
        help="Generator batch normalization decay."
    )

    options_parser.add_argument(
        "--discriminator_hidden_sizes",
        type=str,
        default="256,128",
        help="Size of each hidden layer in the discriminator separated by commas (no spaces)."
    )

    options_parser.add_argument(
        "--num_discriminator_steps",
        type=int,
        default=2,
        help="Number of successive training steps for the discriminator."
    )

    options_parser.add_argument(
        "--num_generator_steps",
        type=int,
        default=1,
        help="Number of successive training steps for the generator."
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

    load_without_cuda(autoencoder, options.input_autoencoder)

    generator = Generator(
        code_size=options.code_size,
        num_hidden_layers=options.generator_hidden_layers,
        bn_decay=options.generator_bn_decay
    )

    load_or_initialize(generator, options.input_generator)

    discriminator = Discriminator(
        features.shape[1],
        hidden_sizes=parse_int_list(options.discriminator_hidden_sizes)
    )

    load_or_initialize(discriminator, options.input_discriminator)

    train(
        autoencoder,
        generator,
        discriminator,
        train_data,
        val_data,
        options.output_autoencoder,
        options.output_generator,
        options.output_discriminator,
        options.output_loss,
        batch_size=options.batch_size,
        start_epoch=options.start_epoch,
        num_epochs=options.num_epochs,
        num_disc_steps=options.num_discriminator_steps,
        num_gen_steps=options.num_generator_steps,
        code_size=options.code_size,
        l2_regularization=options.l2_regularization,
        learning_rate=options.learning_rate,
        temperature=temperature
    )


if __name__ == "__main__":
    main()
