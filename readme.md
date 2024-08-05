# GRU-D or GRU with decay

This repo contains an accurate reproduction of the GRU-D model architecture described by [Che et al 2018 in Recurrent Neural Networks for Multivariate Time Series with Missing Values](https://www.nature.com/articles/s41598-018-24271-9).

As of yet, the publicly available Pytorch implementations are not accurate reimplementations of the above paper and fail to reproduce the results.

## Reproduction of Che et al results:

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

The above python script will not do a 5-fold training/validation run on the data like in the original paper. Thus, the best AUC you will see can vary between training runs (the datasets are shuffled before each run).