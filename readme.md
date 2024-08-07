# GRU-D or GRU with decay

This repo contains an accurate reproduction of the GRU-D model architecture described by [Che et al 2018 in Recurrent Neural Networks for Multivariate Time Series with Missing Values](https://www.nature.com/articles/s41598-018-24271-9).

As of yet, the publicly available Pytorch implementations are not accurate reimplementations of the above paper and fail to reproduce the results.

The files under /models/grud contain modular GRUDCell and GRUD files with similar (but not identical) structure to PyTorch GRU and GRUCell. This should make doplying these modules in your own projects easier.

## Reproduction of Che et al results:

```
python3 grud_classifier.py --mode k_fold -k 5

>>> AUC Scores:  [0.8559997859130807, 0.8564583333333332, 0.8497795039851115, 0.8428457485061258, 0.8183071076857202]
>>> Mean AUC:  0.8446780958846745
>>> Std AUC:  0.014083788833679355

```

## Usage:
Pull the repo and install the necessary dependencies with conda (or install all imports manually):
```
conda env create -f conda_env.yml
```

Activate the newly created environment and run:
```
python3 physionet_preprocessor.py
python3 grud_classifier.py

```

## Notes:

- Currently the output of the GRUD module does not conform to the PyTorch GRU documentation