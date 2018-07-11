# Metrics

An example using the `MIX SMALL` dataset. Mean squared error metrics are returned via standard output.
The `--output` parameter can be used to write into a numpy compressed object the x and y values for plots.

## Probabilities by dimension

```bash
python multi_categorical_gans/metrics/mse_probabilities_by_dimension.py \
    --data_format_x=sparse --data_format_y=dense \
    data/synthetic/mix_small/synthetic-test.features.npz \
    samples/arae/synthetic/mix_small/sample.features.npy
```

## Predictions by dimension

```bash
python multi_categorical_gans/metrics/mse_predictions_by_dimension.py \
    --data_format_x=sparse --data_format_y=dense --data_format_test=sparse \
    data/synthetic/mix_small/synthetic-train.features.npz \
    samples/arae/synthetic/mix_small/sample.features.npy \
    data/synthetic/mix_small/synthetic-test.features.npz
```

## Predictions by categorical

```bash
python multi_categorical_gans/metrics/mse_predictions_by_categorical.py \
    --data_format_x=sparse --data_format_y=dense --data_format_test=sparse \
    data/synthetic/mix_small/synthetic-train.features.npz \
    samples/arae/synthetic/mix_small/sample.features.npy \
    data/synthetic/mix_small/synthetic-test.features.npz \
    data/synthetic/mix_small/metadata.json
```