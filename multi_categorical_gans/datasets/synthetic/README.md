# Synthetic data generation

We generated several synthetic datasets for our experiments.

We decided to save the data in `data/synthetic`:

```bash
mkdir -p data/synthetic/fixed_2
mkdir -p data/synthetic/fixed_10
mkdir -p data/synthetic/mix_small
mkdir -p data/synthetic/mix_big
```

The basic arguments for the script that generates synthetic datasets are:

```bash
python multi_categorical_gans/datasets/synthetic/generate.py
usage: generate.py [-h] [--min_variable_size MIN_VARIABLE_SIZE]
                   [--max_variable_size MAX_VARIABLE_SIZE] [--seed SEED]
                   [--class_distribution CLASS_DISTRIBUTION]
                   [--class_distribution_type {probs,logits,uniform}]
                   num_samples num_variables metadata_path output_path

```

The first variable can be considered as a class or label.
It has a fixed categorical distribution that can be defined with the `class_distribution` and `class_distribution_type` parameters:

- when `class_distribution_type=uniform`, `class_distribution` must be an integer defining the number of classes;
- when `class_distribution_type=probs`, `class_distribution` must be a list of comma separated positive floats
adding up to one that defines the probability of each class;
- when `class_distribution_type=logits`, `class_distribution` must be a list of comma separated floats
that will be used as input for a softmax that will define the probability of each class;


For the following variables, one categorical distribution is defined at random for each possible value of the previous variable.
The parameters `min_variable_size` and `max_variable_size` define the range for the number of possible values of every variable.
During the generation of a sample, the categorical distribution is selected depending on the value drawn for the previous variable.

To generate a dataset similar to the one called `FIXED 2` in the paper:

```bash
python multi_categorical_gans/datasets/synthetic/generate.py 10000 9 \
    data/synthetic/fixed_2/metadata.json \
    data/synthetic/fixed_2/synthetic.features.npz \
    --min_variable_size=2 --max_variable_size=2
```

To generate a dataset similar to the one called `FIXED 10` in the paper:

```bash
python multi_categorical_gans/datasets/synthetic/generate.py 10000 9 \
    data/synthetic/fixed_10/metadata.json \
    data/synthetic/fixed_10/synthetic.features.npz \
    --min_variable_size=10 --max_variable_size=10
```

To generate a dataset similar to the one called `MIX SMALL` in the paper:

```bash
python multi_categorical_gans/datasets/synthetic/generate.py 10000 9 \
    data/synthetic/mix_small/metadata.json \
    data/synthetic/mix_small/synthetic.features.npz \
    --min_variable_size=2 --max_variable_size=10
```

To generate a dataset similar to the one called `MIX BIG` in the paper:

```bash
python multi_categorical_gans/datasets/synthetic/generate.py 10000 99 \
    data/synthetic/mix_big/metadata.json \
    data/synthetic/mix_big/synthetic.features.npz \
    --min_variable_size=2 --max_variable_size=10
```

For more information about the transformation run:

```bash
python multi_categorical_gans/datasets/synthetic/generate.py -h
```