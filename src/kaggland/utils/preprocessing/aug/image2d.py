"""
license: MIT
author: Gabriele Beltramo

Overview
--------
Augmentation transformations for 2D images.

"""

import kaggland.utils.preprocessing.aug.res as res
import numba
import numpy as np

from typing import Tuple


def random_translation(
    images: np.ndarray,
    labels: np.ndarray,
    random_transl_range: Tuple[int, int],
    prob_thresh: float,
    random_seed: int = 0,
) -> res.Aug:
    """Randomly translate a mini-batch of 2d images in-place."""

    assert images.ndim == 3, f"{images.shape=} needs to be (batch_size, L, W)"
    assert random_transl_range[0] < random_transl_range[1], "Invalid translation offsets"
    assert abs(random_transl_range[0]) < images.shape[1], "The x-translation offset is too big"
    assert random_transl_range[1] < images.shape[2], "The y-translation offset is too big"

    np.random.seed(random_seed)
    probs = np.random.random(len(images))
    transl_offsets = np.random.randint(
        low=random_transl_range[0],
        high=random_transl_range[1] + 1,
        size=(len(images), 2),
        dtype=np.int16,
    )

    selected = np.flatnonzero(probs > prob_thresh).astype(np.uint8)
    numba_translation(images, transl_offsets, selected)

    # NOTE labels does not require change

    return res.Aug(selected=selected, params={"offsets": transl_offsets})


@numba.jit(nopython=True, fastmath=True, cache=True)
def numba_translation(images, transl_offsets, selected):
    for idx in selected:
        x_offset, y_offset = transl_offsets[idx]
        img = images[idx]
        x_len, y_len = img.shape

        if x_offset >= 0:
            img[x_offset:, :] = img[0 : x_len - x_offset, :]
            img[0:x_offset, :] = 0
        else:
            img[0:x_offset, :] = img[-(x_len + x_offset) :, :]
            img[x_offset:, :] = 0

        if y_offset >= 0:
            img[:, y_offset:] = img[:, 0 : y_len - y_offset]
            img[:, 0:y_offset] = 0
        else:
            img[:, 0:y_offset] = img[:, -(y_len + y_offset) :]
            img[:, y_offset:] = 0


def random_rotation():
    pass


def random_dilation():
    pass


def random_erosion():
    pass


def random_noise():
    pass
