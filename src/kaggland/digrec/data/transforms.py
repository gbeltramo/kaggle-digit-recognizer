"""
license: MIT
author: Gabriele Beltramo

Overview
--------
Preprocessing transformations.

"""

import numba
import numpy as np

from typing import Tuple


def train_fn(
    image_label: Tuple[np.ndarray, np.ndarray],
    chan_mean: float,
    chan_std: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """MNIST training preprocessing transformations."""

    image, label = image_label[0], image_label[1]
    image = normalize_channels_images(image, chan_mean, chan_std)
    return image, label.astype(np.int64)


@numba.jit(nopython=True, fastmath=True, cache=True)
def normalize_channels_images(
    image: np.ndarray,
    chan_mean: float,
    chan_std: float,
) -> np.ndarray:
    """Normalize channels of `image`."""

    normalized_image = (image - chan_mean) / chan_std
    return normalized_image


@numba.jit(nopython=True, fastmath=True, cache=True)
def inference_fn(
    images: np.ndarray,
    chan_mean: float,
    chan_std: float,
) -> np.ndarray:
    """MNIST inferencing preprocessing transformations."""

    for idx in len(images):
        # in-place channel normalization
        images[idx] -= chan_mean
        images[idx] /= chan_std
    return images
