# Multi-Categorical GANs
Code for the paper [Generating Multi-Categorical Samples with Generative Adversarial Networks](https://arxiv.org/abs/1807.01202)

## Pre-requisites

The project was developed using python 2.7.12 with the following packages:

- numpy==1.14.5
- scipy==1.1.0
- torch==0.4.0

Installation with pip:

```bash
pip install -r requirements.txt
```

## Contents
- Datasets
  - [Synthetic data generation](datasets/synthetic/)
  - [US Census 1990](datasets/uscensus/)
- Methods
  - [ARAE and MC-ARAE](arae/)
  - [MedGAN and MC-MedGAN](medgan/)
  - [MC-Gumbel](mc_gumbel/)
  - [MC-WGAN-GP](mc_wgan_gp/)