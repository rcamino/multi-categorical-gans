from __future__ import division
from __future__ import print_function

import argparse
import json
import torch

import numpy as np

from scipy.sparse import csr_matrix, save_npz
from torch.distributions.one_hot_categorical import OneHotCategorical


distribution_types = ["probs", "logits", "uniform"]


class Variable(object):

    def __init__(self, distributions):
        self.distributions = distributions

    def sample(self, previous_sample):
        k = previous_sample.argmax().item()
        distribution = self.distributions[k]
        return distribution.sample()


def add_one(ones, rows, cols, i, j, sample):
    k = sample.argmax().item()
    ones.append(1)
    rows.append(i)
    cols.append(j + k)


def generate_one_hot_variable(distribution, distribution_type):
    assert distribution_type in distribution_types
    variable = OneHotCategorical(**{distribution_type: torch.FloatTensor(distribution)})
    assert all([prob > 0 for prob in variable.probs])
    return variable


def print_matrix_stats(matrix, num_samples, num_features):
    num_ones = matrix.sum()
    num_positions = num_samples * num_features

    num_ones_per_row = np.asarray(matrix.sum(axis=1)).ravel()
    num_ones_per_column = np.asarray(matrix.sum(axis=0)).ravel()

    print("Min:", matrix.min())
    print("Max:", matrix.max())
    print("Rows:", matrix.shape[0])
    print("Columns:", matrix.shape[1])
    print("Mean ones per row:", num_ones_per_row.mean())
    print("Mean ones per column:", num_ones_per_column.mean())
    print("Total ones:", num_ones)
    print("Total positions:", num_positions)
    print("Total ratio of ones:", num_ones / num_positions)
    print("Empty rows:", np.sum(num_ones_per_row == 0))
    print("Full rows:", np.sum(num_ones_per_row == num_features))
    print("Empty columns:", np.sum(num_ones_per_column == 0))
    print("Full columns:", np.sum(num_ones_per_column == num_samples))


def generate_one_hot(num_samples, num_variables, min_variable_size, max_variable_size, metadata_path, output_path,
                     class_distribution=2, class_distribution_type="uniform", seed=None):

    if seed is not None:
        np.random.seed(seed)

    assert 2 <= min_variable_size <= max_variable_size
    assert class_distribution is not None
    if class_distribution_type == "uniform":
        num_classes = int(class_distribution[0])
        class_distribution = [1.0 / num_classes for _ in range(num_classes)]
        class_distribution_type = "probs"

    # generate classes
    class_variable = generate_one_hot_variable(class_distribution, class_distribution_type)
    num_classes = class_variable.event_shape[0]

    # generate variables
    variables = []
    variable_sizes = [num_classes]
    num_features = num_classes
    last_variable_size = num_classes
    for _ in range(num_variables):
        if min_variable_size == max_variable_size:
            variable_size = min_variable_size
        else:
            variable_size = np.random.randint(low=min_variable_size, high=max_variable_size + 1)

        variable_sizes.append(variable_size)
        distributions = {}
        for input_value in range(last_variable_size):
            logits = torch.FloatTensor(size=(variable_size,)).normal_(0, 1)
            distributions[input_value] = OneHotCategorical(logits=logits)

        variables.append(Variable(distributions))
        num_features += variable_size
        last_variable_size = variable_size

    # generate metadata
    metadata = {
        "seed": seed,
        "variable_sizes": variable_sizes,
        "class_probs": class_variable.probs.tolist(),
        "variable_probs": [[sub_variable.probs.tolist() for sub_variable in variable.distributions.values()]
                           for variable in variables]
    }

    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    # generate data
    ones = []
    rows = []
    cols = []

    for i in range(num_samples):
        j = 0
        class_sample = class_variable.sample()
        add_one(ones, rows, cols, i, j, class_sample)
        j += class_sample.shape[0]
        previous_sample = class_sample
        for variable in variables:
            sample = variable.sample(previous_sample)
            add_one(ones, rows, cols, i, j, sample)
            j += sample.shape[0]
            previous_sample = sample

    output = csr_matrix((ones, (rows, cols)), shape=(num_samples, num_features), dtype=np.uint8)

    print_matrix_stats(output, num_samples, num_features)

    save_npz(output_path, output)


def main():
    options_parser = argparse.ArgumentParser(description="Generate one hot encoded data with cascade dependencies.")

    options_parser.add_argument("num_samples", type=int, help="Number of output samples.")

    options_parser.add_argument("num_variables", type=int, help="Number of output categorical variables.")

    options_parser.add_argument("metadata_path", type=str,
                                help="Output data file path indicating the class distribution and the variable maps.")

    options_parser.add_argument("output_path", type=str,
                                help="Output data file path in sparse format.")

    options_parser.add_argument("--min_variable_size", type=int, default=2,
                                help="Minimum random size of each categorical variable. Should be at least 2.")

    options_parser.add_argument("--max_variable_size", type=int, default=10,
                                help="Maximum random size of each categorical variable.")

    options_parser.add_argument("--seed", type=int, help="Random number generator seed.", default=42)

    options_parser.add_argument("--class_distribution", type=str, default="2",
                                help="Defines the distribution of the class variable. See 'class_distribution_type'.")

    options_parser.add_argument("--class_distribution_type", type=str, default="uniform", choices=distribution_types,
                                help="If uniform, same probability is assigned to every class;" +
                                     " the 'class_distribution' should be the number of classes." +
                                     "\nIf probs, explicit probabilities per class" +
                                     " are defined in 'class_distribution' separated by commas." +
                                     "\nIf logits, the values separated by commas defined in 'class_distribution'" +
                                     " will be used as softmax logits."
                                )

    options = options_parser.parse_args()

    generate_one_hot(options.num_samples,
                     options.num_variables,
                     options.min_variable_size,
                     options.max_variable_size,
                     options.metadata_path,
                     options.output_path,
                     [float(x) for x in options.class_distribution.split(",")],
                     options.class_distribution_type,
                     options.seed
                     )


if __name__ == "__main__":
    main()
