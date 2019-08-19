from __future__ import print_function

import argparse
import csv
import json

from scipy.sparse import load_npz


def uscensus_reverse_transform(input_path, output_path, metadata_path):
    with open(metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)

    features = load_npz(input_path)

    csv_file = open(output_path, "w")
    output = csv.DictWriter(csv_file, ["caseid"] + metadata["variables"])
    output.writeheader()

    for row_value_indices in features:
        _, selected_value_indices = row_value_indices.nonzero()
        # there should be one value per variable
        assert len(selected_value_indices) == len(metadata["variables"])

        row_dict = dict()

        for selected_value_index in selected_value_indices:
            variable, value = metadata["index_to_value"][selected_value_index]
            row_dict[variable] = value

        output.writerow(row_dict)

    csv_file.close()


def main():
    options_parser = argparse.ArgumentParser(
        description="Transform the USCensus feature matrix into text data."
                    + " Dataset: https://archive.ics.uci.edu/ml/datasets/US+Census+Data+(1990)."
    )

    options_parser.add_argument("input", type=str, help="Output features in sparse scipy matrix format.")
    options_parser.add_argument("output", type=str, help="Input USCensus data in text format.")
    options_parser.add_argument("metadata", type=str, help="Metadata in json format.")

    options = options_parser.parse_args()

    uscensus_reverse_transform(options.input, options.output, options.metadata)


if __name__ == "__main__":
    main()
