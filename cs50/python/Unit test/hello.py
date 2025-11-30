def main():
    name = input("What's your name? ")
    print(hello(name))


def hello(to="world"):
    return f"hello, {to}"


if __name__ == "__main__":
    main()


#! Always try to remove side effects if possible
# * not here the problem is hello() dose not have a return value because of that it will not work on the test_hello.py
# def main():
#     name = input("What's your name? ")
#     hello(name)


# def hello(to="world"):
#     print("hello,", to)


# if __name__ == "__main__":
#     main()
