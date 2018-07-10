from __future__ import print_function

import argparse

import numpy as np

from multi_categorical_gans.datasets.formats import data_formats, loaders
from multi_categorical_gans.utils.prediction import prediction_score

from sklearn.metrics.regression import mean_squared_error


def predictions_by_dimension(train, test):
    num_features = train.shape[1]
    class_indices = list(range(num_features))

    prediction_scores = []
    for class_index in class_indices:
        used_feature_indices = [index for index in range(num_features) if index != class_index]
        prediction_scores.append(prediction_score(
            train[:, used_feature_indices], train[:, class_index],
            test[:, used_feature_indices], test[:, class_index],
            metric="f1", model="random_forest_classifier"
        ))
    return np.array(prediction_scores)


def plot_predictions_by_dimension(data_x, data_y, data_test):
    score_y_by_dimension = predictions_by_dimension(data_y, data_test)
    score_x_by_dimension = predictions_by_dimension(data_x, data_test)
    mse = mean_squared_error(score_x_by_dimension, score_y_by_dimension)
    return score_x_by_dimension, score_y_by_dimension, mse


def main():
    options_parser = argparse.ArgumentParser(description="Plot predictions by dimension for two samples.")

    options_parser.add_argument("data_x", type=str, help="Data for x-axis. See 'data_format_x' parameter.")
    options_parser.add_argument("data_y", type=str, help="Data for y-axis. See 'data_format_y' parameter.")

    options_parser.add_argument("data_test", type=str,
                                help="Data for prediction evaluation. See 'data_format_test' parameter.")

    options_parser.add_argument("--output", type=str, help="Output numpy data path.")

    options_parser.add_argument("--data_format_x", type=str, choices=data_formats, default="dense")
    options_parser.add_argument("--data_format_y", type=str, choices=data_formats, default="dense")
    options_parser.add_argument("--data_format_test", type=str, choices=data_formats, default="dense")

    options_parser.add_argument("--seed", type=int, default=42, help="Random number generator seed.")

    options = options_parser.parse_args()

    np.random.seed(options.seed)

    score_x_by_dimension, score_y_by_dimension, mse = plot_predictions_by_dimension(
        loaders[options.data_format_x](options.data_x),
        loaders[options.data_format_y](options.data_y),
        loaders[options.data_format_test](options.data_test)
    )

    if options.output is not None:
        np.savez(options.output, x=score_x_by_dimension, y=score_y_by_dimension, mse=mse)

    print("MSE: ", mse)


if __name__ == "__main__":
    main()
