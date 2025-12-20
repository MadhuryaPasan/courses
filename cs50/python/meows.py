# testing type errors with `mypy`
# * run this on terminal `mypy meows.py`
# def meow(n: int):
#     for _ in range(n):
#         print("meow")


# number: int = input("Number: ")
# meow(number)
# =========================================================================

# correct code
# def meow(n: int):
#     for _ in range(n):
#         print("meow")


# number: int = int(input("Number: "))
# meow(number)
# =========================================================================


# def meow(n: int):
#     for _ in range(n):
#         print("meow")


# number: int = int(input("Number: "))
# meows: str = meow(number)
# print(meows)

"""
! Problem:

now the output of this code is like this ->
`meow
meow
meow
None`

now this display `None` at the end. this is because the funtion did not have any return value. so assining the return value of the funtion to `meows` return None as the value.
"""
# =========================================================================
# * define the return value type of a function


# def meow(n: int) -> None:
#     for _ in range(n):
#         print("meow")


# number: int = int(input("Number: "))
# meows: str = meow(number)
# print(meows)

# now with this mypy can detect the error


# =========================================================================
# * add a type to return value


def meow(n: int) -> str:
    return "meow\n" * n


number: int = int(input("Number: "))
meows: str = meow(number)
print(meows, end="")
