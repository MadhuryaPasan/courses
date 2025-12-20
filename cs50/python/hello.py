# # Ask user for their name
# name = input("what's your name? ")


# # # say hello to user
# # print("-----------------------")
# # print("hello, "+ name)
# # print("hello,", name)


# options in print function
# %%
name = "Pasan"
print("hello, ", end="")
print(name)
# * these two print apeare on the same line because of `end = ""`


# %%
name = "Pasan"
print("hello, ", end="\n")  # * this line ends with a new line
print(name)


# %%
print(
    "hello, ", end="???"
)  # * this line ends with a ??? and the next line in the same line
print(name)


# %%
print("hello,", name, sep=" ")  # * seperate with a space


# %%
print("hello,", name, sep="???")  # * seperate with a ???


# %%
# say hello to user
print(
    'hello,"frined'
)  # * escape caracters (allows to include " inside of a strint that use "")

# %%
# remove witespace from str
name = "    Pasan    "
print("Before: ____", name, "____")
name = name.strip()
print("After: ____", name, "____")

# %%
# capitalize user's input
name = "pasan perera"
name = name.capitalize()  # first word only
print(name)
name = "pasan perera"
name = name.title()
print(name)

# %%
# remove witespace from str and capiltalize user's name (adding two thing in one line)
name = "    pasan    "
name = name.strip().title()  # left to right
print(f"hello, {name}")

# %%
# add witespace from str and capitalize user's name to the input it self to make it more easy
name = input("what is your name?").strip().title()
print(f"hello, {name}")
