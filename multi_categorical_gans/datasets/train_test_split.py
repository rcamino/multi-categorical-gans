from __future__ import print_function

import argparse

import numpy as np

from multi_categorical_gans.datasets.formats import data_formats, loaders, savers


def train_test_split(features, train_size, percent=False, shuffle=False):
    num_samples = features.shape[0]
    if percent:
        assert 0 < train_size < 100, "Invalid percent value."
        limit = int(num_samples * (train_size / 100.0))
    else:
        assert 0 < train_size < num_samples, "Invalid size."
        limit = train_size

    if shuffle:
        index = np.arange(num_samples)
        np.random.shuffle(index)
        features = features[index, :]

    return features[:limit, :], features[limit:, :]


def main():
    options_parser = argparse.ArgumentParser(description="Split features file into train and test files.")

    options_parser.add_argument("features", type=str, help="Input features file.")
    options_parser.add_argument("train_size", type=int, help="Number of samples for the train part.")
    options_parser.add_argument("train_features", type=str, help="Output train features file.")
    options_parser.add_argument("test_features", type=str, help="Output test features file.")

    options_parser.add_argument(
        "--data_format",
        type=str,
        default="sparse",
        choices=data_formats,
        help="Either a dense numpy array or a sparse csr matrix."
    )

    options_parser.add_argument("--percent", default=False, action="store_true",
                                help="Interpret the train size as a percentage.")

    options_parser.add_argument("--shuffle", default=False, action="store_true",
                                help="Shuffle the dataset before the split.")

    options = options_parser.parse_args()

    loader = loaders[options.data_format]
    saver = savers[options.data_format]

    train_features, test_features = train_test_split(
        loader(options.features, transform=False),
        options.train_size,
        percent=options.percent,
        shuffle=options.shuffle
    )

    saver(options.train_features, train_features)
    saver(options.test_features, test_features)

    print("Train features:", train_features.shape)
    print("Test features:", test_features.shape)


if __name__ == "__main__":
    main()
