# ==========================================
#  TESTING IN PYTHON: A STEP-BY-STEP GUIDE
# ==========================================
# Assumption: There is a file named calculator.py with a function square()
from calculator import square


# ==========================================
# LEVEL 1: The Manual Method
# ==========================================
# Problem: This works, but writing 'if' statements for everything is tedious.
# It also doesn't explicitly tell us "TEST FAILED" unless we print it.

# def main():
#     test_square()

# def test_square():
#     if square(2) != 4:
#         print("2 squared was not 4")
#     if square(3) != 9:
#         print("3 squared was not 9")

# if __name__ == "__main__":
#     main()


# ==========================================
# LEVEL 2: Using the `assert` Keyword
# ==========================================
# Improvement: Uses the built-in `assert` keyword.
# Problem: If the first assertion fails (AssertionError), the program crashes immediately.
# It stops running and doesn't check the rest of the numbers.

# def main():
#     test_square()

# def test_square():
#     assert square(2) == 4
#     assert square(3) == 9

# if __name__ == "__main__":
#     main()


# ==========================================
# LEVEL 3: Using `try` and `except`
# ==========================================
# Improvement: Prevents the program from crashing so all tests can run.
# Problem: This works, but look how much code we have to write!
# It is too long (verbose) and hard to maintain.

# def main():
#     test_square()

# def test_square():
#     try:
#         assert square(2) == 4
#     except AssertionError:
#         print("2 squared was not 4")

#     try:
#         assert square(3) == 9
#     except AssertionError:
#         print("3 squared was not 9")

#     try:
#         assert square(-2) == 4
#     except AssertionError:
#         print("-2 squared was not 4")

#     try:
#         assert square(-3) == 9
#     except AssertionError:
#         print("-3 squared was not 9")

#     try:
#         assert square(0) == 0
#     except AssertionError:
#         print("0 squared was not 0")

# if __name__ == "__main__":
#     main()


# ==========================================
# LEVEL 4: The Best Solution (`pytest`)
# ==========================================
# Solution: We use the external library `pytest`.
# - No need for try/except blocks.
# - No need for `if __name__ == "__main__"`.
# - Tests are separated into categories (Positive, Negative, Zero, Errors).

import pytest

def test_positive():
    assert square(2) == 4
    assert square(3) == 9

def test_negative():
    assert square(-2) == 4
    assert square(-3) == 9

def test_zero():
    assert square(0) == 0

def test_str():
    with pytest.raises(TypeError): # checks if the function handles TypeErrors correctly
        square("cat")

# ------------------------------------------
# HOW TO RUN THIS:
# Type the following in your terminal:
# pytest test_calculator.py
# ------------------------------------------

























# # using pytest
# import pytest
# from calculator import square


# # def test_square():
# #     assert square(2) == 4
# #     assert square(3) == 9
# #     assert square(-2) == 4
# #     assert square(-3) == 9
# #     assert square(0) == 0


# # * a better way


# def test_positive():
#     assert square(2) == 4
#     assert square(3) == 9


# def test_negative():
#     assert square(-2) == 4
#     assert square(-3) == 9


# def test_zero():
#     assert square(0) == 0


# def test_str():
#     with pytest.raises(TypeError): # check for type erros
#         square("cat")


# # to run this `pytest test_calculator.py`

# ###################################################
# # * ____this is manual method____

# # from calculator import square

# # def main():
# #     test_square()

# # def test_square():
# #     if square(2) != 4:
# #         print("2 squared was not 4")
# #     if square(3) != 9:
# #         print("3 squared was not 9")


# # if __name__ == "__main__":
# #     main()


# # *____ Using `assert`____
# # * this program show `AssertionError`

# # # ! this is not user friendly
# # from calculator import square


# # def main():
# #     test_square()


# # def test_square():
# #     assert square(2) == 4
# #     assert square(3) == 9


# # if __name__ == "__main__":
# #     main()

# #! need to add try extept to handle error
# #! but this is also not good because this hase lot of codes
# # from calculator import square


# # def main():
# #     test_square()


# # def test_square():
# #     try:
# #         assert square(2) == 4
# #     except AssertionError:
# #         print("2 squared was not 4")

# #     try:
# #         assert square(3) == 9
# #     except AssertionError:
# #         print("3 squared was not 9")

# #     try:
# #         assert square(-2) == 4
# #     except AssertionError:
# #         print("-2 squared was not 4")

# #     try:
# #         assert square(-3) == 9
# #     except AssertionError:
# #         print("-3 squared was not 9")

# #     try:
# #         assert square(0) == 0
# #     except AssertionError:
# #         print("0 squared was not 0")


# # if __name__ == "__main__":
# #     main()
