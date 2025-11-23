# # Ask user for their name
# name = input("what's your name? ")


# # # say hello to user
# # print("-----------------------")
# # print("hello, "+ name)
# # print("hello,", name)


# # # options in print function
# # print("-----------------------")
# # print("hello, ", end="")
# # print(name)
# # print("-----------------------")
# # print("hello, ", end="\n")
# # print(name)
# # print("-----------------------")
# # print("hello, ", end="???")
# # print(name)
# # print("-----------------------")
# # print("-----------------------")
# # print("hello,",name,sep=' ')
# # print("hello,",name,sep='???')



# # #say hello to user
# # print("hello,\"frined") #escape caracters


# # #remove witespace from str
# # name=name.strip()
# # # capitalize user's input
# # # name=name.capitalize() # first word only
# # name = name.title()


# # remove witespace from str and capiltalize user's name (adding two thing in one line)
# name = name.strip().title() # left to right
# print(f"hello,{name}")

# add witespace from str and capitalize user's name to the input it self to make it more easy
name = input("what is your name?").strip().title()

print(f"hello {name}")


