import kaggland.digrec.data.data as data
import numpy as np
import os
import pathlib
import pytest

path_to_data = (
    pathlib.Path(os.getenv("HOME")) / "data" / "kaggle" / "digit-recognizer" / "train.csv"
)


def test_input_data_make():
    with pytest.raises(AssertionError):
        data.make(str(path_to_data))


def test_data_labels_are_0_to_9():
    train_data, val_data = data.make(path_to_data)
    train_images = train_data["images"]
    train_labels = train_data["labels"]

    assert (np.arange(10) == np.unique(train_labels)).all()

    for _, val_img_lab in val_data.items():
        val_labels = val_img_lab["labels"]
        assert (np.arange(10) == np.unique(val_labels)).all()


def test_all_indices():
    train_data, val_data = data.make(path_to_data)
    train_indices = train_data["indices"]
    val_splits = tuple(value["indices"] for value in val_data.values())

    indices_splits = np.concatenate((train_indices, *val_splits))
    assert (np.unique(indices_splits) == np.arange(len(indices_splits))).all()
