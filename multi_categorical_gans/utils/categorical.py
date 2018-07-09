import json

import torch
import torch.nn.functional as F


def load_variable_sizes_from_metadata(metadata_path):
    with open(metadata_path, "r") as metadata_file:
        metadata = json.load(metadata_file)
    return metadata["variable_sizes"]


def categorical_variable_loss(reconstructed, original, variable_sizes):
    # by default use loss for binary variables
    if variable_sizes is None:
        return F.binary_cross_entropy(reconstructed, original)
    # use the variable sizes when available
    else:
        loss = 0
        start = 0
        continuous_size = 0
        for variable_size in variable_sizes:
            # if it is a categorical variable
            if variable_size > 1:
                # add loss from the accumulated continuous variables
                if continuous_size > 0:
                    end = start + continuous_size
                    batch_reconstructed_variable = reconstructed[:, start:end]
                    batch_target = original[:, start:end]
                    loss += F.mse_loss(batch_reconstructed_variable, batch_target)
                    start = end
                    continuous_size = 0
                # add loss from categorical variable
                end = start + variable_size
                batch_reconstructed_variable = reconstructed[:, start:end]
                batch_target = torch.argmax(original[:, start:end], dim=1)
                loss += F.cross_entropy(batch_reconstructed_variable, batch_target)
                start = end
            # if not, accumulate continuous variables
            else:
                continuous_size += 1

        # add loss from the remaining accumulated continuous variables
        if continuous_size > 0:
            end = start + continuous_size
            batch_reconstructed_variable = reconstructed[:, start:end]
            batch_target = original[:, start:end]
            loss += F.mse_loss(batch_reconstructed_variable, batch_target)

        return loss
