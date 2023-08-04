import kaggland.digrec.data.load
import numpy as np
import pathlib

train_data = pathlib.Path.home() / "data" / "kaggle" / "digit-recognizer" / "train.csv"


def test_read_data():
    images, labels = kaggland.digrec.data.load.read_data(str(train_data))

    assert images[0].shape == (28, 28)
    assert images[-1].shape == (28, 28)
    assert len(images) == len(labels)
    assert type(images[0, 0, 0]) is np.uint8
    assert type(labels[0]) is np.uint8


def test_transform():
    images, labels = kaggland.digrec.data.load.read_data(str(train_data))
    images = kaggland.digrec.data.load.transform(images)

    assert isinstance(images, np.ndarray)


def test_load():
    images, labels = kaggland.digrec.data.load.load(str(train_data))

    assert images[0].shape == (28, 28)
    assert images[-1].shape == (28, 28)
    assert len(images) == len(labels)
    assert type(images[0, 0, 0]) is np.uint8
    assert type(labels[0]) is np.uint8
