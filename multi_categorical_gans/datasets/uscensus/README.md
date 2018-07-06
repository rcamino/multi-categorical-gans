# US Census 1990

This is one of the datasets we used for our experiments.

We decided to save the data in `data/uscensus`:

```bash
mkdir -p data/uscensus
```

To download the data you can visit the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/US+Census+Data+(1990))
or download it directly:

```bash
cd data/uscensus
wget https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt
```

To transform the csv data one-hot-encoding each categorical variable we run:

```bash
python multi_categorical_gans/datasets/uscensus/transform.py \
    data/uscensus/USCensus1990.data.txt \
    data/uscensus/USCensus1990.features.npz \
    data/uscensus/metadata.json
```

For more information about the transformation run:

```bash
python multi_categorical_gans/datasets/uscensus/transform.py -h
```