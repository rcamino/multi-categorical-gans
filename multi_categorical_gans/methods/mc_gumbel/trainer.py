from __future__ import division
from __future__ import print_function

import argparse
import torch

import numpy as np

from torch.autograd.variable import Variable
from torch.nn import BCELoss
from torch.optim import Adam

from multi_categorical_gans.datasets.dataset import Dataset
from multi_categorical_gans.datasets.formats import data_formats, loaders

from multi_categorical_gans.methods.general.discriminator import Discriminator
from multi_categorical_gans.methods.general.generator import Generator

from multi_categorical_gans.utils.categorical import load_variable_sizes_from_metadata
from multi_categorical_gans.utils.commandline import DelayedKeyboardInterrupt, parse_int_list
from multi_categorical_gans.utils.cuda import to_cuda_if_available, to_cpu_if_available
from multi_categorical_gans.utils.initialization import load_or_initialize
from multi_categorical_gans.utils.logger import Logger


def train(generator,
          discriminator,
          train_data,
          val_data,
          output_gen_path,
          output_disc_path,
          output_loss_path,
          batch_size=1000,
          start_epoch=0,
          num_epochs=1000,
          num_disc_steps=2,
          num_gen_steps=1,
          noise_size=128,
          l2_regularization=0.001,
          learning_rate=0.001,
          temperature=0.666
          ):
    generator, discriminator = to_cuda_if_available(generator, discriminator)

    optim_gen = Adam(generator.parameters(), weight_decay=l2_regularization, lr=learning_rate)
    optim_disc = Adam(discriminator.parameters(), weight_decay=l2_regularization, lr=learning_rate)

    criterion = BCELoss()

    logger = Logger(output_loss_path, append=start_epoch > 0)

    for epoch_index in range(start_epoch, num_epochs):
        logger.start_timer()

        # train
        generator.train(mode=True)
        discriminator.train(mode=True)

        disc_losses = []
        gen_losses = []

        more_batches = True
        train_data_iterator = train_data.batch_iterator(batch_size)

        while more_batches:
            # train discriminator
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
                noise = Variable(torch.FloatTensor(len(batch), noise_size).normal_())
                noise = to_cuda_if_available(noise)
                fake_features = generator(noise, training=True, temperature=temperature)
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
            for _ in range(num_gen_steps):
                optim_gen.zero_grad()

                noise = Variable(torch.FloatTensor(len(batch), noise_size).normal_())
                noise = to_cuda_if_available(noise)
                gen_features = generator(noise, training=True, temperature=temperature)
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
                noise = Variable(torch.FloatTensor(len(batch), noise_size).normal_())
                noise = to_cuda_if_available(noise)
                fake_features = generator(noise, training=False, temperature=temperature)
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
            torch.save(generator.state_dict(), output_gen_path)
            torch.save(discriminator.state_dict(), output_disc_path)
            logger.flush()

    logger.close()


def main():
    options_parser = argparse.ArgumentParser(description="Train MC-Gumbel.")

    options_parser.add_argument("data", type=str, help="Training data. See 'data_format' parameter.")

    options_parser.add_argument("metadata", type=str,
                                help="Information about the categorical variables in json format.")

    options_parser.add_argument("output_generator", type=str, help="Generator output file.")
    options_parser.add_argument("output_discriminator", type=str, help="Discriminator output file.")
    options_parser.add_argument("output_loss", type=str, help="Loss output file.")

    options_parser.add_argument("--input_generator", type=str, help="Generator input file.", default=None)
    options_parser.add_argument("--input_discriminator", type=str, help="Discriminator input file.", default=None)

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
        "--noise_size",
        type=int,
        default=128,
        help=""
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
        "--generator_hidden_sizes",
        type=str,
        default="256,128",
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
        default=0.666,
        help="Gumbel-Softmax temperature."
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

    variable_sizes = load_variable_sizes_from_metadata(options.metadata)

    generator = Generator(
        options.noise_size,
        variable_sizes,
        hidden_sizes=parse_int_list(options.generator_hidden_sizes),
        bn_decay=options.bn_decay
    )

    load_or_initialize(generator, options.input_generator)

    discriminator = Discriminator(
        features.shape[1],
        hidden_sizes=parse_int_list(options.discriminator_hidden_sizes),
        bn_decay=options.bn_decay,
        critic=False
    )

    load_or_initialize(discriminator, options.input_discriminator)

    train(
        generator,
        discriminator,
        train_data,
        val_data,
        options.output_generator,
        options.output_discriminator,
        options.output_loss,
        batch_size=options.batch_size,
        start_epoch=options.start_epoch,
        num_epochs=options.num_epochs,
        num_disc_steps=options.num_discriminator_steps,
        num_gen_steps=options.num_generator_steps,
        noise_size=options.noise_size,
        l2_regularization=options.l2_regularization,
        learning_rate=options.learning_rate,
        temperature=options.temperature
    )


if __name__ == "__main__":
    main()
