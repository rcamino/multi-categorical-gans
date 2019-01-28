# Multi-Categorical GANs

Code for the paper [Generating Multi-Categorical Samples with Generative Adversarial Networks](https://arxiv.org/abs/1807.01202)

## Pre-requisites

The project was developed using python 3.6.7 with the following packages:

- future==0.17.1
- numpy==1.16.0
- scikit-learn==0.20.2
- scipy==1.2.0
- torch==1.0.0

Installation with pip:

```bash
pip install -r requirements.txt
```

## Contents
- [Datasets](multi_categorical_gans/datasets)
  - [Synthetic data generation](multi_categorical_gans/datasets/synthetic/)
  - [US Census 1990](multi_categorical_gans/datasets/uscensus/)
- [Methods](multi_categorical_gans/methods)
  - [ARAE and MC-ARAE](multi_categorical_gans/methods/arae/)
  - [MedGAN and MC-MedGAN](multi_categorical_gans/methods/medgan/)
  - [MC-Gumbel](multi_categorical_gans/methods/mc_gumbel/)
  - [MC-WGAN-GP](multi_categorical_gans/methods/mc_wgan_gp/)
- [Metrics](multi_categorical_gans/metrics)

## Changelog

- 2019-01-28: changed to Python 3 as suggested (and still compatible with 2.7 ... I hope).
- 2018-07-25: now we use WGAN-GP for ARAE following the author updates.