# ARAE and MC-ARAE

Based on the code from [jakezhaojb/ARAE](https://github.com/jakezhaojb/ARAE).

# ARAE

An example using the `MIX SMALL` dataset:

```bash
mkdir -p models/arae/synthetic/mix_small
mkdir -p samples/arae/synthetic/mix_small
```

Training:

```bash
python multi_categorical_gans/methods/arae/trainer.py \
    --data_format=sparse \
    --code_size=65 \
    --noise_size=10 \
    --encoder_hidden_sizes="" \
    --decoder_hidden_sizes="" \
    --batch_size=100 \
    --num_epochs=5000 \
    --l2_regularization=0 \
    --learning_rate=1e-5 \
    --generator_hidden_sizes=100,100,100 \
    --bn_decay=0.9 \
    --discriminator_hidden_sizes=100 \
    --num_autoencoder_steps=1 \
    --num_discriminator_steps=1 \
    --num_generator_steps=1 \
    --autoencoder_noise_radius=0 \
    --seed=123 \
    data/synthetic/mix_small/synthetic-train.features.npz \
    models/arae/synthetic/mix_small/autoencoder.torch \
    models/arae/synthetic/mix_small/generator.torch \
    models/arae/synthetic/mix_small/discriminator.torch \
    models/arae/synthetic/mix_small/loss.csv
```

Sampling:

```bash
python multi_categorical_gans/methods/arae/sampler.py \
    --code_size=65 \
    --noise_size=10 \
    --encoder_hidden_sizes="" \
    --decoder_hidden_sizes="" \
    --batch_size=1000 \
    --generator_hidden_sizes=100,100,100 \
    --generator_bn_decay=0.9 \
    models/arae/synthetic/mix_small/autoencoder.torch \
    models/arae/synthetic/mix_small/generator.torch \
    10000 65 \
    samples/arae/synthetic/mix_small/sample.features.npy
```

# MC-ARAE

Now the metadata (with the variable size information) and the gumbel-softmax temperature is needed.

An example using the `MIX SMALL` dataset:

```bash
mkdir -p models/mc-arae/synthetic/mix_small
mkdir -p samples/mc-arae/synthetic/mix_small
```

Training:

```bash
python multi_categorical_gans/methods/arae/trainer.py \
    --metadata=data/synthetic/mix_small/metadata.json \
    --data_format=sparse \
    --code_size=65 \
    --noise_size=10 \
    --encoder_hidden_sizes="" \
    --decoder_hidden_sizes="" \
    --batch_size=100 \
    --num_epochs=5000 \
    --l2_regularization=0 \
    --learning_rate=1e-5 \
    --generator_hidden_sizes=100,100,100 \
    --bn_decay=0.99 \
    --discriminator_hidden_sizes=100 \
    --num_autoencoder_steps=1 \
    --num_discriminator_steps=1 \
    --num_generator_steps=1 \
    --autoencoder_noise_radius=0 \
    --seed=123 \
    --temperature=0.666 \
    data/synthetic/mix_small/synthetic-train.features.npz \
    models/mc-arae/synthetic/mix_small/autoencoder.torch \
    models/mc-arae/synthetic/mix_small/generator.torch \
    models/mc-arae/synthetic/mix_small/discriminator.torch \
    models/mc-arae/synthetic/mix_small/loss.csv
```

Sampling:

```bash
python multi_categorical_gans/methods/arae/sampler.py \
    --metadata=data/synthetic/mix_small/metadata.json \
    --code_size=65 \
    --noise_size=10 \
    --encoder_hidden_sizes="" \
    --decoder_hidden_sizes="" \
    --batch_size=1000 \
    --generator_hidden_sizes=100,100,100 \
    --generator_bn_decay=0.99 \
    --temperature=0.666 \
    models/mc-arae/synthetic/mix_small/autoencoder.torch \
    models/mc-arae/synthetic/mix_small/generator.torch \
    10000 65 \
    samples/mc-arae/synthetic/mix_small/sample.features.npy
```