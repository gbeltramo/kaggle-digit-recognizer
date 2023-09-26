#!/usr/bin/env bash

mamba create -y -n ka-pytorch python=3.11
mamba install -y -n ka-pytorch -c pytorch -c nvidia pytorch=2.0.1 torchvision pytorch-cuda=11.7
mamba install -y -n ka-pytorch -c conda-forge numpy numba scipy ipython pytest polars omegaconf scikit-learn matplotlib torchdata jupyter jupyterlab build black kaggle mlflow bumpver graphviz python-graphviz
mamba install -y -n ka-pytorch -c numba icc_rt
