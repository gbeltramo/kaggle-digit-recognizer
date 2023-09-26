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
    image: np.ndarray,
    label: np.ndarray,
    max_translation_offset: int,
    prob_thresh: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly translate one 2d images in-place."""

    assert image.ndim == 2, f"{image.shape=} needs to be (L, W)"
    assert max_translation_offset < image.shape[0], "The maximum translation offset is too big"
    assert max_translation_offset < image.shape[1], "The maximum translation offset is too big"
    assert max_translation_offset >= 1, "The maximum translation offset needs to be greater than 0"

    prob = np.random.random()

    if prob > prob_thresh:
        transl_offset = _generate_random_translation_offset(
            max_translation_offset=max_translation_offset,
            random_seed=random_seed,
        )
        print(f"{transl_offset=}")

        _numba_translation(image[:], transl_offset)

        # NOTE label does not require change

    return image, label


def _generate_random_translation_offset(max_translation_offset: int, random_seed: int = 0):
    np.random.seed(random_seed)
    transl_offset = np.random.randint(
        low=1,
        high=max_translation_offset + 1,
        size=2,
        dtype=np.int32,
    )

    # Flip translation offsets with 50% probability
    if np.random.random() > 0.5:
        transl_offset[0] *= -1

    if np.random.random() > 0.5:
        transl_offset[1] *= -1

    return transl_offset


@numba.jit(nopython=True, fastmath=True, cache=True)
def _numba_translation(image, transl_offset):
    y_offset, x_offset = transl_offset
    y_len, x_len = image.shape

    if y_offset > 0:
        image[y_offset:, :] = image[0 : y_len - y_offset, :]
        image[0:y_offset, :] = 0
    elif y_offset < 0:
        image[0:y_offset, :] = image[-(y_len + y_offset) :, :]
        image[y_offset:, :] = 0

    if x_offset > 0:
        image[:, x_offset:] = image[:, 0 : x_len - x_offset]
        image[:, 0:x_offset] = 0
    elif x_offset < 0:
        image[:, 0:x_offset] = image[:, -(x_len + x_offset) :]
        image[:, x_offset:] = 0


def random_batch_translation(
    images: np.ndarray,
    labels: np.ndarray,
    max_translation_offset: int,
    prob_thresh: float,
    random_seed: int = 0,
) -> res.Aug:
    """Randomly translate a mini-batch of 2d images in-place."""

    if len(images) == 0:
        return res.Aug()

    assert images.ndim == 3, f"{images.shape=} needs to be (batch_size, L, W)"
    assert max_translation_offset < images.shape[1], "The maximum translation offset is too big"
    assert max_translation_offset < images.shape[2], "The maximum translation offset is too big"
    assert max_translation_offset >= 1, "The maximum translation offset needs to be greater than 0"

    np.random.seed(random_seed)
    probs = np.random.random(len(images))

    transl_offsets = _generate_random_batch_translation_offsets(
        max_translation_offset=max_translation_offset,
        num_offsets=len(images),
        random_seed=random_seed,
    )

    selected = np.flatnonzero(probs > prob_thresh).astype(np.int32)
    _numba_batch_translation(images, transl_offsets, selected)

    # NOTE labels does not require change

    return res.Aug(selected=selected, params={"offsets": transl_offsets})


def _generate_random_batch_translation_offsets(
    max_translation_offset: int, num_offsets: int, random_seed: int = 0
):
    np.random.seed(random_seed)
    transl_offsets = np.random.randint(
        low=1,
        high=max_translation_offset + 1,
        size=(num_offsets, 2),
        dtype=np.int32,
    )

    # Flip translation offsets with 50% probability
    indices_negative_y_offsets = np.random.choice(num_offsets, size=num_offsets // 2, replace=False)
    indices_negative_x_offsets = np.random.choice(num_offsets, size=num_offsets // 2, replace=False)
    transl_offsets[indices_negative_y_offsets, 0] *= -1
    transl_offsets[indices_negative_x_offsets, 1] *= -1

    return transl_offsets


@numba.jit(nopython=True, fastmath=True, cache=True)
def _numba_batch_translation(images, transl_offsets, selected):
    for idx in selected:
        _numba_translation(images[idx], transl_offsets[idx])


def random_rotation():
    pass


def random_dilation():
    pass


def random_erosion():
    pass


def random_noise():
    pass
