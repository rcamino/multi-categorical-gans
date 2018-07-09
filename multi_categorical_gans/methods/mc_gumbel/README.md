# MC-Gumbel

We decided to save the models in `models/mc-gumbel` and the samples in `samples/mc-gumbel`.
An example using the `MIX SMALL` dataset:

```bash
mkdir -p models/mc-gumbel/synthetic/mix_small
mkdir -p samples/mc-gumbel/synthetic/mix_small
```

Training:

```bash
python multi_categorical_gans/methods/mc_gumbel/trainer.py \
    --data_format=sparse \
    --noise_size=10 \
    --batch_size=100 \
    --num_epochs=1000 \
    --l2_regularization=0 \
    --learning_rate=1e-3 \
    --generator_hidden_sizes=100,100,100 \
    --bn_decay=0.9 \
    --discriminator_hidden_sizes=100 \
    --num_discriminator_steps=2 \
    --num_generator_steps=1 \
    --seed=123 \
    --temperature=0.666 \
    data/synthetic/mix_small/synthetic-train.features.npz \
    data/synthetic/mix_small/metadata.json \
    models/mc-gumbel/synthetic/mix_small/generator.torch \
    models/mc-gumbel/synthetic/mix_small/discriminator.torch \
    models/mc-gumbel/synthetic/mix_small/loss.csv
```

Sampling:

```bash
python multi_categorical_gans/methods/mc_gumbel/sampler.py \
    --noise_size=10 \
    --batch_size=1000 \
    --generator_hidden_sizes=100,100,100 \
    --generator_bn_decay=0.9 \
    --temperature=0.666 \
    models/mc-gumbel/synthetic/mix_small/generator.torch \
    data/synthetic/mix_small/metadata.json \
    10000 65 \
    samples/mc-gumbel/synthetic/mix_small/sample.features.npy
```