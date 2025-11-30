from hello import hello


def test_default():
    assert hello() == "hello, world"


def test_argument():
    assert hello("Pasan") == "hello, Pasan"


def test_argument_loop():
    for name in ["Hermione", "Harry", "Ron"]:
        assert hello(name) == f"hello, {name}"