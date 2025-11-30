def main():
    hello("world")
    goodbye("world")


def hello(name):
    print(f"hello, {name}")


def goodbye(name):
    print(f"goodbye, {name}")


# main() # !if this file is called by a other file the because of the main() the other file will also display the output of the sayings file 

#* to prevent this need to modify main like this

if __name__ == "__main__":
    main()