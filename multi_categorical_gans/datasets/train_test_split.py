from __future__ import print_function

import argparse

from multi_categorical_gans.datasets.formats import data_formats, loaders, savers
from sklearn.model_selection import train_test_split


def main():
    options_parser = argparse.ArgumentParser(description="Split features file into train and test files.")

    options_parser.add_argument("features", type=str, help="Input features file.")
    options_parser.add_argument("train_size", type=float, help="Number or proportion of samples for the train part.")
    options_parser.add_argument("train_features", type=str, help="Output train features file.")
    options_parser.add_argument("test_features", type=str, help="Output test features file.")

    options_parser.add_argument(
        "--features_format",
        type=str,
        default="sparse",
        choices=data_formats,
        help="Either a dense numpy array or a sparse csr matrix."
    )

    options_parser.add_argument("--labels", type=str, help="Input labels file.")
    options_parser.add_argument("--train_labels", type=str, help="Output train labels file.")
    options_parser.add_argument("--test_labels", type=str, help="Output test labels file.")

    options_parser.add_argument(
        "--labels_format",
        type=str,
        default="sparse",
        choices=data_formats,
        help="Either a dense numpy array or a sparse csr matrix."
    )

    options_parser.add_argument("--shuffle", default=False, action="store_true",
                                help="Shuffle the dataset before the split.")

    options = options_parser.parse_args()

    features_loader = loaders[options.features_format]
    features_saver = savers[options.features_format]
    features = features_loader(options.features, transform=False)

    if 0 < options.train_size < 1:
        test_size = 1 - options.train_size
    elif options.train_size > 1:
        test_size = len(features) - options.train_size
    else:
        raise Exception("Invalid train size.")

    if options.labels is None:
        train_features, test_features = train_test_split(features,
                                                         train_size=options.train_size,
                                                         test_size=test_size,
                                                         shuffle=options.shuffle)
    else:
        labels_loader = loaders[options.labels_format]
        labels_saver = savers[options.labels_format]
        labels = labels_loader(options.labels, transform=False)

        train_features, test_features, train_labels, test_labels = train_test_split(features,
                                                                                    labels,
                                                                                    train_size=options.train_size,
                                                                                    test_size=test_size,
                                                                                    shuffle=options.shuffle)

    features_saver(options.train_features, train_features)
    features_saver(options.test_features, test_features)

    print("Train features:", train_features.shape, "Test features:", test_features.shape)

    if options.labels is not None:
        labels_saver(options.train_labels, train_labels)
        labels_saver(options.test_labels, test_labels)

        print("Train labels:", train_labels.shape, "Test labels:", test_labels.shape)


if __name__ == "__main__":
    main()
