"""
license: MIT
author: Gabriele Beltramo

Overview
--------
Utility functions for 2D images.

"""

import numpy as np

from typing import Tuple


def compute_channels_mean_std(
    images: np.ndarray, np_type=np.float32
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the mean and standard deviation of the channels of a set of 2D images.

    Note that this function assumes that the channels are axis=1 of the Numpy array `images`.
    """

    assert len(images) >= 1, "At least 1 image is needed"
    assert images.ndim == 4, f"{images.shape=} needs to be (N, num_channels, L, W)"

    chan_mean = np.mean(images, axis=(0, 2, 3)).astype(np_type)
    chan_std = np.std(images, axis=(0, 2, 3)).astype(np_type)
    return chan_mean, chan_std
