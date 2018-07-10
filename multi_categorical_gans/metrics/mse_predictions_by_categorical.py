from __future__ import print_function

import argparse

import numpy as np

from multi_categorical_gans.datasets.formats import data_formats, loaders
from multi_categorical_gans.utils.categorical import load_variable_sizes_from_metadata, separate_categorical
from multi_categorical_gans.utils.prediction import prediction_score

from sklearn.metrics.regression import mean_squared_error


def predictions_by_categorical(train, test, variable_sizes):
    prediction_scores = []
    for selected_index, variable_size in enumerate(variable_sizes):
        train_X, train_y = separate_categorical(train, variable_sizes, selected_index)
        test_X, test_y = separate_categorical(test, variable_sizes, selected_index)
        prediction_scores.append(prediction_score(
            train_X, train_y, test_X, test_y,
            metric="accuracy", model="random_forest_classifier"
        ))
    return np.array(prediction_scores)


def plot_predictions_by_categorical(data_x, data_y, data_test, variable_sizes):
    score_y_by_categorical = predictions_by_categorical(data_y, data_test, variable_sizes)
    score_x_by_categorical = predictions_by_categorical(data_x, data_test, variable_sizes)
    mse = mean_squared_error(score_x_by_categorical, score_y_by_categorical)
    return score_x_by_categorical, score_y_by_categorical, mse


def main():
    options_parser = argparse.ArgumentParser(description="Plot predictions by categorical variable for two samples.")

    options_parser.add_argument("data_x", type=str, help="Data for x-axis. See 'data_format_x' parameter.")
    options_parser.add_argument("data_y", type=str, help="Data for y-axis. See 'data_format_y' parameter.")

    options_parser.add_argument("data_test", type=str,
                                help="Data for prediction evaluation. See 'data_format_test' parameter.")

    options_parser.add_argument("metadata", type=str,
                                help="Information about the categorical variables in json format.")

    options_parser.add_argument("--output", type=str, help="Output numpy data path.")

    options_parser.add_argument("--data_format_x", type=str, choices=data_formats, default="dense")
    options_parser.add_argument("--data_format_y", type=str, choices=data_formats, default="dense")
    options_parser.add_argument("--data_format_test", type=str, choices=data_formats, default="dense")

    options_parser.add_argument("--seed", type=int, default=42, help="Random number generator seed.")

    options = options_parser.parse_args()

    np.random.seed(options.seed)

    variable_sizes = load_variable_sizes_from_metadata(options.metadata)

    score_x_by_categorical, score_y_by_categorical, mse = plot_predictions_by_categorical(
        loaders[options.data_format_x](options.data_x),
        loaders[options.data_format_y](options.data_y),
        loaders[options.data_format_test](options.data_test),
        variable_sizes
    )

    if options.output is not None:
        np.savez(options.output, x=score_x_by_categorical, y=score_y_by_categorical, mse=mse)

    print("MSE: ", mse)


if __name__ == "__main__":
    main()
