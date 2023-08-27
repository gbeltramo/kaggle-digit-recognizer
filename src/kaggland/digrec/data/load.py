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

from typing import Tuple


def read_data(path_to_csv: pathlib.PosixPath) -> Tuple[np.ndarray, np.ndarray]:
    """Read a MNIST .csv file containing train/test images and labels."""

    assert (
        isinstance(path_to_csv, pathlib.PosixPath)
    ), f"{path_to_csv=} needs to be a pathlib.PosixPath"
    
    assert (
        os.path.splitext(str(path_to_csv))[1] == ".csv"
    ), f"{path_to_csv=} extension needs to be `.csv`"

    csv = pl.scan_csv(path_to_csv)

    shape_MNIST_images = (28, 28)
    images = (
        csv.select(pl.col("*").exclude("label").cast(pl.UInt8))
        .collect()
        .to_numpy()
        .reshape(-1, *shape_MNIST_images)
    )

    labels = csv.select(pl.col("label").cast(pl.UInt8)).collect().to_numpy().ravel()

    return images, labels


def transform(images: np.ndarray) -> np.ndarray:
    return images


def load(path_to_csv: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read and transform .csv MNIST data"""
    images, labels = read_data(path_to_csv)
    images = transform(images)
    return images, labels
