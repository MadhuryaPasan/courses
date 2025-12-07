import sys

# print("hello, my name is", sys.argv[1])

# if user did not enter a value at the end it will give an a exception (IndexError)
# *this is one way to solve this

# try:
#     print("hello, my name is", sys.argv[1])
# except IndexError:
#     print("Too few arguments")

# *this is one way to solve this
# if len(sys.argv) < 2:
#     print("Too few arguments")
# elif len(sys.argv) > 2:
#     print("Too many arguments")
# else:
#     print("hello, my nane is", sys.argv[1])


# *this is the best practice
# * `sys.exit` -> exist program right away

# check for errors
# if len(sys.argv) < 2:
#     sys.exit("Too few arguments")
# elif len(sys.argv) > 2:
#     sys.exit("Too many arguments")

# # Print name tags
# print("hello, my name is", sys.argv[1])

# *iterate over a list of args
if len(sys.argv) < 2:
    sys.exit("Too few arguments")

for arg in sys.argv[1:]: # get slice of the list (without this it will print the name of the script also)
    print("hello, my name is", arg)
