"""
license: MIT
author: Gabriele Beltramo

Overview
--------
Code used to load the dataset from the "Digit recognizer" Kaggle competition.

"""


import os
import numpy as np
import pathlib
import polars as pl

from sklearn.model_selection import StratifiedKFold
from typing import Tuple


def make(path_to_data: pathlib.PosixPath, images_datatype=np.float32):
    """Read and split the MNIST dataset."""

    assert isinstance(
        path_to_data, pathlib.PosixPath
    ), f"{path_to_data=} needs to be a pathlib.Path"

    images, labels = read(path_to_data, images_datatype)

    train_indices, val_splits = compute_train_val_splits(
        labels=labels, num_splits=10, num_val_splits=3, random_state=0
    )

    train_data = {
        "images": images[train_indices],
        "labels": labels[train_indices],
        "indices": train_indices,
    }
    val_data = {
        idx: {"images": images[val_indices], "labels": labels[val_indices], "indices": val_indices}
        for idx, val_indices in enumerate(val_splits)
    }

    return train_data, val_data


def read(
    path_to_csv: pathlib.PosixPath, images_datatype=np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a MNIST .csv file containing train/test images and labels."""

    assert isinstance(
        path_to_csv, pathlib.PosixPath
    ), f"{path_to_csv=} needs to be a pathlib.PosixPath"

    assert (
        os.path.splitext(str(path_to_csv))[1] == ".csv"
    ), f"{path_to_csv=} extension needs to be `.csv`"

    csv = pl.scan_csv(path_to_csv)

    shape_MNIST_images = (1, 28, 28)
    images = (
        csv.select(pl.col("*").exclude("label").cast(pl.UInt8))
        .collect()
        .to_numpy()
        .reshape(-1, *shape_MNIST_images)
        .astype(images_datatype)
    )

    labels = csv.select(pl.col("label").cast(pl.UInt8)).collect().to_numpy().ravel()

    return images, labels


def compute_train_val_splits(
    labels: np.ndarray, num_splits: int, num_val_splits: int, random_state: int = 0
):
    """Return training/validation data splits using `StratifiedKFold`."""

    cross_val_splits = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)

    indices_splits = []
    for _, idx_ith_fold in cross_val_splits.split(labels, y=labels):
        indices_splits.append(idx_ith_fold)

    # get indices of train and validation splits
    num_train_splits = num_splits - num_val_splits
    train_indices = np.concatenate(indices_splits[:num_train_splits])
    val_splits = indices_splits[num_train_splits:]

    # shuffle indices in splits, setting random_seed
    np.random.seed(random_state)
    train_indices = train_indices[np.random.permutation(len(train_indices))]
    for idx, val_indices in enumerate(val_splits):
        val_splits[idx] = val_indices[np.random.permutation(len(val_indices))]

    return train_indices, val_splits
