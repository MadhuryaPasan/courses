from hello import hello

def test_default():
    assert hello() == "hello, world"


def test_argument():
    assert hello("Pasan") == "hello, Pasan"


# because this is inside a folder need to add a new file named __init__.py
# this tell python to treet this folder as a package
# so now in the terminal run `pytest test` (test is the folder name). (* terminal need to be on the Unit test folder. do not navigate to test folder)