# Datasets

In this package you will find scripts to process or generate the datasets from the paper:

- [Synthetic data generation](synthetic/)
- [US Census 1990](uscensus/)

## Loading and saving

We work either with dense or sparse numpy arrays. The module  `multi_categorical_gans.datasets.formats` presents some
functions to operate with both data formats in an abstract way.

## Train and test split

Examples of how to split a dataset into 90% train and 10% test:

```bash
python multi_categorical_gans/datasets/train_test_split.py \
    data/synthetic/fixed_2/synthetic.features.npz \
    0.9 \
    data/synthetic/fixed_2/synthetic-train.features.npz \
    data/synthetic/fixed_2/synthetic-test.features.npz
```

```bash
python multi_categorical_gans/datasets/train_test_split.py \
    data/synthetic/fixed_10/synthetic.features.npz \
    0.9 \
    data/synthetic/fixed_10/synthetic-train.features.npz \
    data/synthetic/fixed_10/synthetic-test.features.npz
```

```bash
python multi_categorical_gans/datasets/train_test_split.py \
    data/synthetic/mix_small/synthetic.features.npz \
    0.9 \
    data/synthetic/mix_small/synthetic-train.features.npz \
    data/synthetic/mix_small/synthetic-test.features.npz
```

```bash
python multi_categorical_gans/datasets/train_test_split.py \
    data/synthetic/mix_big/synthetic.features.npz \
    0.9 \
    data/synthetic/mix_big/synthetic-train.features.npz \
    data/synthetic/mix_big/synthetic-test.features.npz
```

```bash
python multi_categorical_gans/datasets/train_test_split.py \
    data/uscensus/USCensus1990.features.npz \
    0.9 \
    data/uscensus/USCensus1990-train.features.npz \
    data/uscensus/USCensus1990-test.features.npz
```

For more information about the split run:

```bash
python multi_categorical_gans/datasets/train_test_split.py -h
```

## The dataset wrapper

The class `multi_categorical_gans.datasets.dataset.Dataset` can wrap a dense numpy array to provide simple operations
for training, like `split(proportion)` (useful for validation) or `batch_iterator(batch_size, shuffle=True)`.