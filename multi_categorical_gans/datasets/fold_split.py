from __future__ import print_function

import argparse
import os

from multi_categorical_gans.datasets.formats import data_formats, loaders, savers
from sklearn.model_selection import KFold


def main():
    options_parser = argparse.ArgumentParser(description="Split features file into train and test files.")

    options_parser.add_argument("features", type=str, help="Input features file.")
    options_parser.add_argument("folds", type=int, help="Number of folds.")
    options_parser.add_argument("output_directory", type=str, help="Output directory path for the folds.")
    options_parser.add_argument("features_template", type=str, help="Output features file name template. ")

    options_parser.add_argument(
        "--features_format",
        type=str,
        default="sparse",
        choices=data_formats,
        help="Either a dense numpy array or a sparse csr matrix."
    )

    options_parser.add_argument("--labels", type=str, help="Input labels file.")
    options_parser.add_argument("--labels_template", type=str, help="Output labels file name template. ")

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

    if options.labels is not None:
        labels_loader = loaders[options.labels_format]
        labels_saver = savers[options.labels_format]
        labels = labels_loader(options.labels, transform=False)

    k_fold = KFold(n_splits=options.folds, shuffle=options.shuffle)
    for fold_number, (train_index, test_index) in enumerate(k_fold.split(features)):
        train_features, test_features = features[train_index, :], features[test_index, :]
        template = os.path.join(options.output_directory, options.features_template)

        features_saver(template.format(name="train", number=fold_number, total=options.folds), train_features)
        features_saver(template.format(name="test", number=fold_number, total=options.folds), test_features)

        print("Train features:", train_features.shape, "Test features:", test_features.shape)

        if options.labels is not None:
            train_labels, test_labels = labels[train_index], labels[test_index]
            template = os.path.join(options.output_directory, options.labels_template)

            labels_saver(template.format(name="train", number=fold_number, total=options.folds), train_labels)
            labels_saver(template.format(name="test", number=fold_number, total=options.folds), test_labels)

            print("Train labels:", train_labels.shape, "Test labels:", test_labels.shape)


if __name__ == "__main__":
    main()
