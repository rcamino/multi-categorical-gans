# MedGAN and MC-MedGAN

Based on the code from [mp2893/medgan](https://github.com/mp2893/medgan).

# MedGAN

An example using the `MIX SMALL` dataset:

```bash
mkdir -p models/medgan/synthetic/mix_small
mkdir -p samples/medgan/synthetic/mix_small
```

Pre-training:

```bash
python multi_categorical_gans/methods/medgan/pre_trainer.py \
    --data_format=sparse \
    --code_size=65 \
    --encoder_hidden_sizes="" \
    --decoder_hidden_sizes="" \
    --batch_size=100 \
    --num_epochs=100 \
    --l2_regularization=0 \
    --learning_rate=1e-3 \
    --seed=123 \
    data/synthetic/mix_small/synthetic-train.features.npz \
    models/medgan/synthetic/mix_small/pre-autoencoder.torch \
    models/medgan/synthetic/mix_small/pre-loss.csv
```

Training:

```bash
python multi_categorical_gans/methods/medgan/trainer.py \
    --data_format=sparse \
    --code_size=65 \
    --encoder_hidden_sizes="" \
    --decoder_hidden_sizes="" \
    --batch_size=100 \
    --num_epochs=1000 \
    --l2_regularization=0 \
    --learning_rate=1e-3 \
    --generator_hidden_layers=2 \
    --generator_bn_decay=0.99 \
    --discriminator_hidden_sizes=256,128 \
    --num_discriminator_steps=2 \
    --num_generator_steps=1 \
    --seed=123 \
    data/synthetic/mix_small/synthetic-train.features.npz \
    models/medgan/synthetic/mix_small/pre-autoencoder.torch \
    models/medgan/synthetic/mix_small/autoencoder.torch \
    models/medgan/synthetic/mix_small/generator.torch \
    models/medgan/synthetic/mix_small/discriminator.torch \
    models/medgan/synthetic/mix_small/loss.csv
```

Sampling:

```bash
python multi_categorical_gans/methods/medgan/sampler.py \
    --code_size=65 \
    --encoder_hidden_sizes="" \
    --decoder_hidden_sizes="" \
    --batch_size=100 \
    --generator_hidden_layers=2 \
    --generator_bn_decay=0.99 \
    models/medgan/synthetic/mix_small/autoencoder.torch \
    models/medgan/synthetic/mix_small/generator.torch \
    10000 65 \
    samples/medgan/synthetic/mix_small/sample.features.npy
```

# MC-MedGAN

Now the metadata (with the variable size information) and the gumbel-softmax temperature is needed.

An example using the `MIX SMALL` dataset:

```bash
mkdir -p models/mc-medgan/synthetic/mix_small
mkdir -p samples/mc-medgan/synthetic/mix_small
```

Pre-training:

```bash
python multi_categorical_gans/methods/medgan/pre_trainer.py \
    --metadata=data/synthetic/mix_small/metadata.json \
    --data_format=sparse \
    --code_size=65 \
    --encoder_hidden_sizes="" \
    --decoder_hidden_sizes="" \
    --batch_size=100 \
    --num_epochs=100 \
    --l2_regularization=0 \
    --learning_rate=1e-3 \
    --temperature=0.666 \
    --seed=123 \
    data/synthetic/mix_small/synthetic-train.features.npz \
    models/mc-medgan/synthetic/mix_small/pre-autoencoder.torch \
    models/mc-medgan/synthetic/mix_small/pre-loss.csv
```

Training:

```bash
python multi_categorical_gans/methods/medgan/trainer.py \
    --metadata=data/synthetic/mix_small/metadata.json \
    --data_format=sparse \
    --code_size=65 \
    --encoder_hidden_sizes="" \
    --decoder_hidden_sizes="" \
    --batch_size=100 \
    --num_epochs=1000 \
    --l2_regularization=0 \
    --learning_rate=1e-3 \
    --generator_hidden_layers=2 \
    --generator_bn_decay=0.99 \
    --discriminator_hidden_sizes=256,128 \
    --num_discriminator_steps=2 \
    --num_generator_steps=1 \
    --temperature=0.666 \
    --seed=123 \
    data/synthetic/mix_small/synthetic-train.features.npz \
    models/mc-medgan/synthetic/mix_small/pre-autoencoder.torch \
    models/mc-medgan/synthetic/mix_small/autoencoder.torch \
    models/mc-medgan/synthetic/mix_small/generator.torch \
    models/mc-medgan/synthetic/mix_small/discriminator.torch \
    models/mc-medgan/synthetic/mix_small/loss.csv
```

Sampling:

```bash
python multi_categorical_gans/methods/medgan/sampler.py \
    --metadata=data/synthetic/mix_small/metadata.json \
    --code_size=65 \
    --encoder_hidden_sizes="" \
    --decoder_hidden_sizes="" \
    --batch_size=100 \
    --generator_hidden_layers=2 \
    --generator_bn_decay=0.99 \
    --temperature=0.666 \
    models/mc-medgan/synthetic/mix_small/autoencoder.torch \
    models/mc-medgan/synthetic/mix_small/generator.torch \
    10000 65 \
    samples/mc-medgan/synthetic/mix_small/sample.features.npy
```