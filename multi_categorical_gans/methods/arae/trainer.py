from __future__ import print_function

import argparse
import torch

import numpy as np

from torch.autograd.variable import Variable
from torch.optim import Adam

from multi_categorical_gans.datasets.dataset import Dataset
from multi_categorical_gans.datasets.formats import data_formats, loaders

from multi_categorical_gans.methods.general.autoencoder import AutoEncoder
from multi_categorical_gans.methods.general.generator import Generator
from multi_categorical_gans.methods.general.discriminator import Discriminator
from multi_categorical_gans.methods.general.wgan_gp import calculate_gradient_penalty

from multi_categorical_gans.utils.categorical import load_variable_sizes_from_metadata, categorical_variable_loss
from multi_categorical_gans.utils.commandline import DelayedKeyboardInterrupt, parse_int_list
from multi_categorical_gans.utils.cuda import to_cuda_if_available, to_cpu_if_available
from multi_categorical_gans.utils.initialization import load_or_initialize
from multi_categorical_gans.utils.logger import Logger


def add_noise_to_code(code, noise_radius):
    if noise_radius > 0:
        means = torch.zeros_like(code)
        gauss_noise = torch.normal(means, noise_radius)
        return code + to_cuda_if_available(Variable(gauss_noise))
    else:
        return code


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
          num_ae_steps=1,
          num_disc_steps=2,
          num_gen_steps=1,
          noise_size=128,
          l2_regularization=0.001,
          learning_rate=0.001,
          ae_noise_radius=0.2,
          ae_noise_anneal=0.995,
          normalize_code=True,
          variable_sizes=None,
          temperature=None,
          penalty=0.1
          ):
    autoencoder, generator, discriminator = to_cuda_if_available(autoencoder, generator, discriminator)

    optim_ae = Adam(autoencoder.parameters(), weight_decay=l2_regularization, lr=learning_rate)
    optim_gen = Adam(generator.parameters(), weight_decay=l2_regularization, lr=learning_rate)
    optim_disc = Adam(discriminator.parameters(), weight_decay=l2_regularization, lr=learning_rate)

    logger = Logger(output_loss_path, append=start_epoch > 0)

    for epoch_index in range(start_epoch, num_epochs):
        logger.start_timer()

        # train
        autoencoder.train(mode=True)
        generator.train(mode=True)
        discriminator.train(mode=True)

        ae_losses = []
        disc_losses = []
        gen_losses = []

        more_batches = True
        train_data_iterator = train_data.batch_iterator(batch_size)

        while more_batches:
            # train autoencoder
            for _ in range(num_ae_steps):
                try:
                    batch = next(train_data_iterator)
                except StopIteration:
                    more_batches = False
                    break

                autoencoder.zero_grad()

                batch_original = Variable(torch.from_numpy(batch))
                batch_original = to_cuda_if_available(batch_original)
                batch_code = autoencoder.encode(batch_original, normalize_code=normalize_code)
                batch_code = add_noise_to_code(batch_code, ae_noise_radius)

                batch_reconstructed = autoencoder.decode(batch_code, training=True, temperature=temperature)

                ae_loss = categorical_variable_loss(batch_reconstructed, batch_original, variable_sizes)
                ae_loss.backward()

                optim_ae.step()

                ae_loss = to_cpu_if_available(ae_loss)
                ae_losses.append(ae_loss.data.numpy())

            # train discriminator
            for _ in range(num_disc_steps):
                try:
                    batch = next(train_data_iterator)
                except StopIteration:
                    more_batches = False
                    break

                discriminator.zero_grad()
                autoencoder.zero_grad()

                # first train the discriminator only with real data
                real_features = Variable(torch.from_numpy(batch))
                real_features = to_cuda_if_available(real_features)
                real_code = autoencoder.encode(real_features, normalize_code=normalize_code)
                real_code = add_noise_to_code(real_code, ae_noise_radius)
                real_pred = discriminator(real_code)
                real_loss = - real_pred.mean(0).view(1)
                real_loss.backward()

                # then train the discriminator only with fake data
                noise = Variable(torch.FloatTensor(len(batch), noise_size).normal_())
                noise = to_cuda_if_available(noise)
                fake_code = generator(noise)
                fake_code = fake_code.detach()  # do not propagate to the generator
                fake_pred = discriminator(fake_code)
                fake_loss = fake_pred.mean(0).view(1)
                fake_loss.backward()

                # this is the magic from WGAN-GP
                gradient_penalty = calculate_gradient_penalty(discriminator, penalty, real_code, fake_code)
                gradient_penalty.backward()

                optim_ae.step()
                optim_disc.step()

                disc_loss = real_loss + fake_loss + gradient_penalty
                disc_loss = to_cpu_if_available(disc_loss)
                disc_losses.append(disc_loss.data.numpy())

                del disc_loss
                del gradient_penalty
                del fake_loss
                del real_loss

            # train generator
            for _ in range(num_gen_steps):
                generator.zero_grad()

                noise = Variable(torch.FloatTensor(len(batch), noise_size).normal_())
                noise = to_cuda_if_available(noise)
                gen_code = generator(noise)
                fake_pred = discriminator(gen_code)
                fake_loss = - fake_pred.mean(0).view(1)
                fake_loss.backward()

                optim_gen.step()

                fake_loss = to_cpu_if_available(fake_loss)
                gen_losses.append(fake_loss.data.numpy()[0])

                del fake_loss

        # log epoch metrics for current class
        logger.log(epoch_index, num_epochs, "autoencoder", "train_mean_loss", np.mean(ae_losses))
        logger.log(epoch_index, num_epochs, "discriminator", "train_mean_loss", np.mean(disc_losses))
        logger.log(epoch_index, num_epochs, "generator", "train_mean_loss", np.mean(gen_losses))

        # save models for the epoch
        with DelayedKeyboardInterrupt():
            torch.save(autoencoder.state_dict(), output_ae_path)
            torch.save(generator.state_dict(), output_gen_path)
            torch.save(discriminator.state_dict(), output_disc_path)
            logger.flush()

        ae_noise_radius *= ae_noise_anneal

    logger.close()


def main():
    options_parser = argparse.ArgumentParser(description="Train ARAE or MC-ARAE. "
                                                         + "Define 'metadata' and 'temperature' to use MC-ARAE.")

    options_parser.add_argument("data", type=str, help="Training data. See 'data_format' parameter.")

    options_parser.add_argument("output_autoencoder", type=str, help="Autoencoder output file.")
    options_parser.add_argument("output_generator", type=str, help="Generator output file.")
    options_parser.add_argument("output_discriminator", type=str, help="Discriminator output file.")
    options_parser.add_argument("output_loss", type=str, help="Loss output file.")

    options_parser.add_argument("--input_autoencoder", type=str, help="Autoencoder input file.", default=None)
    options_parser.add_argument("--input_generator", type=str, help="Generator input file.", default=None)
    options_parser.add_argument("--input_discriminator", type=str, help="Discriminator input file.", default=None)

    options_parser.add_argument("--metadata", type=str,
                                help="Information about the categorical variables in json format.")

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
        help="Either a dense numpy array or a sparse csr matrix."
    )

    options_parser.add_argument(
        "--code_size",
        type=int,
        default=128,
        help="Dimension of the autoencoder latent space."
    )

    options_parser.add_argument(
        "--noise_size",
        type=int,
        default=128,
        help="Dimension of the generator input noise."
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
        "--start_epoch",
        type=int,
        default=0,
        help="Starting epoch."
    )

    options_parser.add_argument(
        "--num_epochs",
        type=int,
        default=5000,
        help="Number of epochs."
    )

    options_parser.add_argument(
        "--l2_regularization",
        type=float,
        default=0,
        help="L2 regularization weight for every parameter."
    )

    options_parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Adam learning rate."
    )

    options_parser.add_argument(
        "--generator_hidden_sizes",
        type=str,
        default="100,100,100",
        help="Size of each hidden layer in the generator separated by commas (no spaces)."
    )

    options_parser.add_argument(
        "--bn_decay",
        type=float,
        default=0.9,
        help="Batch normalization decay for the generator and discriminator."
    )

    options_parser.add_argument(
        "--discriminator_hidden_sizes",
        type=str,
        default="100",
        help="Size of each hidden layer in the discriminator separated by commas (no spaces)."
    )

    options_parser.add_argument(
        "--num_autoencoder_steps",
        type=int,
        default=1,
        help="Number of successive training steps for the autoencoder."
    )

    options_parser.add_argument(
        "--num_discriminator_steps",
        type=int,
        default=1,
        help="Number of successive training steps for the discriminator."
    )

    options_parser.add_argument(
        "--num_generator_steps",
        type=int,
        default=1,
        help="Number of successive training steps for the generator."
    )

    options_parser.add_argument(
        "--autoencoder_noise_radius",
        type=float,
        default=0,
        help="Gaussian noise standard deviation for the latent code (autoencoder regularization)."
    )

    options_parser.add_argument(
        "--autoencoder_noise_anneal",
        type=float,
        default=0.995,
        help="Anneal the noise radius by this value after every epoch."
    )

    options_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Gumbel-Softmax temperature."
    )

    options_parser.add_argument(
        "--penalty",
        type=float,
        default=0.1,
        help="WGAN-GP gradient penalty lambda."
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

    load_or_initialize(autoencoder, options.input_autoencoder)

    generator = Generator(
        options.noise_size,
        options.code_size,
        hidden_sizes=parse_int_list(options.generator_hidden_sizes),
        bn_decay=options.bn_decay
    )

    load_or_initialize(generator, options.input_generator)

    discriminator = Discriminator(
        options.code_size,
        hidden_sizes=parse_int_list(options.discriminator_hidden_sizes),
        bn_decay=0,  # no batch normalization for the critic
        critic=True
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
        num_ae_steps=options.num_autoencoder_steps,
        num_disc_steps=options.num_discriminator_steps,
        num_gen_steps=options.num_generator_steps,
        noise_size=options.noise_size,
        l2_regularization=options.l2_regularization,
        learning_rate=options.learning_rate,
        ae_noise_radius=options.autoencoder_noise_radius,
        ae_noise_anneal=options.autoencoder_noise_anneal,
        variable_sizes=variable_sizes,
        temperature=temperature,
        penalty=options.penalty
    )


if __name__ == "__main__":
    main()
