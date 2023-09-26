import kaggland.utils.preprocessing.aug.image2d as aug_image2d
import numpy as np


def test_random_batch_translation():
    np.random.seed(0)
    batch_size = 16
    images_shape = (28, 28)
    images = np.random.randint(low=0, high=256, size=(batch_size, *images_shape), dtype=np.uint8)

    aug_res = aug_image2d.random_batch_translation(
        images=images,
        labels=None,
        max_translation_offset=5,
        prob_thresh=0.5,
        random_seed=0,
    )

    print(f"{aug_res=}")
