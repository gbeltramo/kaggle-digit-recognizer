#!/usr/bin/env bash

mamba create --name ka-pytorch2 -y python=3.11
mamba install -n ka-pytorch -c pytorch -c nvidia pytorch=2.0.1 torchvision pytorch-cuda=11.7
mamba install -n ka-pytorch -c conda-forge numpy numba scipy ipython pytest \
      polars omegaconf scikit-learn matplotlib jupyter jupyterlab \
      build black kaggle mlflow bumpver graphviz python-graphviz
mamba install -n ka-pytorch -c numba icc_rt

