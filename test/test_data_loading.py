import kaggland.digrec.data.load


def test_test():
    res = kaggle.digrec.data.load.test("Gabriele")
    assert "Your name" in res
