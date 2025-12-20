# CS50 Python Introduction

This notebook contains comprehensive notes and code examples from the CS50 Introduction to Python course. The content is organized by topic for easy reference.

## Getting Started: Hello, World!

In the grand tradition of programming, our first step is to greet the world. But let's add a CS50 twist by making it interactive.

### Asking for User Input

We can use the `input()` function to get information from the user. It's a good practice to clean up the input using `.strip()` to remove accidental whitespace and `.title()` to format the name nicely.

```python
# ask user for their name
name = input("what's is your name?").strip().title()

# split user's name into first name and last name
first, last = name.split(" ")  # split with a space

# say hello to user
print(f"Hello, {first}")
```

## Integers and Arithmetic

Python makes working with numbers straightforward.

### Basic Operations

You can perform arithmetic directly.

```python
1 + 1
```

### Calculator Example

Let's build a simple calculator. We'll start by using variables and then incorporate user input.

```python
# Using variables
x = 1
y = 2
z = x + y
print(z)
```

When we take input from the user, it's always a string. We need to convert it to a number (an integer in this case) using `int()` before we can do math.

```python
# With user input
x = input("What's x?")
y = input("What's y?")
z = int(x) + int(y)  # converting string to int
print(z)
```

We can make this code more concise by converting the input to an integer immediately.

```python
# A more compact version
x = int(input("What's x?"))
y = int(input("What's y?"))
print(x + y)
```

## Floating Point Numbers

For numbers with decimal points, we use floats. The process is similar to integers, but we use the `float()` function for conversion.

```python
x = float(input("What's x?"))
y = float(input("What's y?"))
print(x + y)
```

### Rounding

Sometimes you get more decimal places than you need. The `round()` function is here to help.

You can round to the nearest whole number:

```python
x = float(input("What's x?"))
y = float(input("What's y?"))
z = round(x + y)
print(z)
```

Or round to a specific number of decimal places:

```python
x = float(input("What's x?"))
y = float(input("What's y?"))
z = round(x / y, 2)  # round up to the last 2 decimal points
print(z)
```

### Number Formatting with F-Strings

F-strings are a powerful way to format your output. You can control the number of decimal places and even add commas to make large numbers more readable.

```python
# Format to 2 decimal places
x = float(input("What's x?"))
y = float(input("What's y?"))
z = x / y
print(f"{z:.2f}")

# Add commas for thousands separators
x = float(input("What's x?"))
y = float(input("What's y?"))
z = round(x + y)
print(f"{z:,}")
```

## Functions (`def`)

Functions allow us to define reusable blocks of code. The `def` keyword is used to create a function.

### Basic Function

Here's a simple function that prints a greeting.

```python
def hello():
    print("hello")

name = input("What's your name? ")
hello()
print(name)
```

### Functions with Parameters

We can make functions more flexible by passing in arguments (parameters).

```python
def hello(to):
    print("hello,", to)

name = input("What's your name? ")
hello(name)
```

We can also provide a default value for a parameter, which is used if no argument is provided.

```python
# default value
def hello(to="World"):
    print("hello,", to)

name = input("What's your name? ")
hello(name)
hello() # This will use the default "World"
```

### The `main` Function

It's a standard convention in Python to have a `main` function that acts as the entry point for your program's logic.

```python
def main():
    name = input("What's your name? ")
    hello(name)

def hello(to="World"):
    print("hello,", to)

main()
# By using a main function, you can define your other functions in any order.
```

### Variable Scope

A variable's "scope" determines where it can be accessed. Variables defined inside a function are local to that function.

```python
def main():
    name = input("What's your name? ")
    hello()

def hello():
    # This will cause an error because 'name' is local to main()
    print("hello,", name)

main()
```
To fix this, you must pass the variable as an argument.

### Return Values

Functions can also send data back to the code that called them using the `return` keyword.

```python
def main():
    x = int(input("What's x?"))
    print("x squared is ", square(x))

def square(n):
    return n * n
    # Other ways to do the same thing:
    # return n**2
    # return pow(n, 2)

main()
```

## Conditionals

Conditionals allow your program to make decisions using `if`, `elif` (else if), and `else`.

### Comparison

Let's compare two numbers.

```python
# compare.py
x = int(input("what's x? "))
y = int(input("what's y?"))

if x < y:
    print("x is less than y")
elif x > y:
    print("x is greater than y")
else:
    print("x is equal to y")
```

We can also use logical operators like `or` and `and`.

```python
# Using 'or'
if x < y or x > y:
    print("x is not equal to y")
else:
    print("x is equal to y")

# A simpler way using '!=' (not equal)
if x != y:
    print("x is not equal to y")
else:
    print("x is equal to y")
```

### Example: Grading

Here's how you can use conditionals to assign a grade based on a score.

```python
# grade.py
score = int(input("Score: "))

if score >= 90:
    print("Grade: A")
elif score >= 80:
    print("Grade: B")
elif score >= 70:
    print("Grade: C")
elif score >= 60:
    print("Grade: D")
else:
    print("Grade: F")
```

### Example: Parity (Even or Odd)

The modulo operator (`%`) is perfect for checking if a number is even or odd.

```python
# parity.py
x = int(input("what's x? "))

if x % 2 == 0:
    print("Even")
else:
    print("Odd")
```

Let's wrap this logic in a function. A function that returns `True` or `False` is often called a "boolean function."

```python
def main():
    x = int(input("what's x? "))
    if is_even(x):
        print("Even")
    else:
        print("Odd")

def is_even(n):
    # A boolean expression directly returns True or False
    return n % 2 == 0

main()
```

### Match-Case

Python 3.10 introduced the `match-case` statement, which is a powerful way to handle multiple comparisons, similar to a `switch` statement in other languages.

```python
# house.py
name = input("What's your name?")

match name:
    case "Harry" | "Hermione" | "Ron":
        print("Gryffindor")
    case "Draco":
        print("Slytherin")
    case _: # The underscore is a wildcard for "anything else"
        print("Who?")
```

## Loops

Loops are used to repeat a block of code.

### While Loops

A `while` loop continues as long as a condition is `True`.

```python
# A simple while loop
i = 0
while i < 3:
    print("meow")
    i += 1  # This is shorthand for i = i + 1
```

### For Loops

A `for` loop iterates over a sequence (like a list or a range).

```python
# Looping with a list
for i in [0, 1, 2]:
    print("meow")

# A more common way using range()
for _ in range(3): # Using '_' when you don't need the loop variable
    print("meow")
```

A fun trick with strings:

```python
print("meow\n" * 3, end="")
```

### Getting Valid User Input

We can combine loops with `try-except` to ensure the user gives us the input we need.

```python
def main():
    number = get_number()
    meow(number)

def get_number():
    while True:
        try:
            n = int(input("What's n? "))
            if n > 0:
                return n
        except ValueError:
            print("n must be an integer")

def meow(n):
    for _ in range(n):
        print("meow")

main()
```

## Data Structures

### Lists

A list is an ordered, mutable collection of items.

```python
students = ["Hermione", "Harry", "Ron"]

print(students)
print(students[0]) # Access items by index (starting from 0)

# Loop through a list
for student in students:
    print(student)

# Loop with an index
for i in range(len(students)):
    print(i + 1, students[i])
```

### Dictionaries

A dictionary is an unordered collection of key-value pairs. It's perfect for associating related data.

```python
# Using a dictionary to store houses
students = {
    "Hermione": "Gryffindor",
    "Harry": "Gryffindor",
    "Ron": "Gryffindor",
    "Draco": "Slytherin",
}

print(students["Hermione"]) # Access value by key

# Loop through a dictionary
for student in students:
    print(student, students[student], sep=", ")
```

You can also create a list of dictionaries, which is a very common and powerful pattern.

```python
students = [
    {"name": "Hermione", "house": "Gryffindor", "patronus": "Otter"},
    {"name": "Harry", "house": "Gryffindor", "patronus": "Stag"},
    {"name": "Ron", "house": "Gryffindor", "patronus": "Jack Russell terrier"},
    {"name": "Draco", "house": "Slytherin", "patronus": None},
]

for student in students:
    print(student["name"], student["house"], student["patronus"], sep=", ")
```

## Drawing with Loops: Mario Example

We can use nested loops to create simple visual patterns.

```python
# Printing a column
def print_column(height):
    for _ in range(height):
        print("#")

# Printing a row
def print_row(width):
    print("#" * width)

# Printing a square
def print_square(size):
    # For each row...
    for i in range(size):
        # ...print a row of bricks
        print("#" * size)

print_square(3)
```

## Exceptions

Errors are a part of programming. Learning how to handle them gracefully makes your code more robust.

### Common Exceptions

Here are some of the most common errors you'll encounter:

| Exception | Cause | Example |
| :--- | :--- | :--- |
| **`SyntaxError`** | Invalid Python code. | `if x == 5` (missing `:`) |
| **`TypeError`** | Operation on the wrong type. | `"Age: " + 25` |
| **`ValueError`** | Right type, but inappropriate value. | `int("hello")` |
| **`IndexError`** | Accessing a list index that doesn't exist. | `my_list[10]` |
| **`KeyError`** | Accessing a dictionary key that doesn't exist. | `my_dict["unknown_key"]` |
| **`NameError`** | Using a variable that hasn't been defined. | `print(x)` |

### Try and Except

The `try-except` block is how you handle potential errors.

```python
try:
    x = int(input("What's x?"))
    print(f"x is {x}")
except ValueError:
    print("x is not an integer")
```

It's best practice to put only the code that might cause the error inside the `try` block. You can use an `else` block for code that should run only if no exception occurred.

```python
while True:
    try:
        x = int(input("What's x?"))
    except ValueError:
        print("x is not an integer")
        # You can use 'pass' here to silently ignore the error
    else:
        break

print(f"x is {x}")
```

We can organize this into a clean function.

```python
def main():
    x = get_int("What's x? ")
    print(f"x is {x}")

def get_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            pass # Keep trying without printing an error message

main()
```

## Libraries and Modules

Python's power comes from its extensive collection of libraries.

- **Module**: A single Python file (`.py`).
- **Package**: A directory of modules.
- **Library**: A collection of packages/modules.

### `random` Module

The `random` module is great for games, simulations, and more.

```python
import random

# Choose a random item from a list
coin = random.choice(["head", "tail"])
print(coin)

# Generate a random integer
number = random.randint(1, 10)
print(number)

# Shuffle a list in-place
cards = ["jack", "queen", "king"]
random.shuffle(cards)
for card in cards:
    print(card)
```

### `statistics` Module

Provides functions for mathematical statistics.

```python
import statistics

print(statistics.mean([100, 90]))
```

### Command-Line Arguments (`sys.argv`)

You can run your Python scripts with arguments directly from the terminal. The `sys` module helps you access them.

```python
# In a file named 'sayings.py'
import sys

if len(sys.argv) == 2:
    print("hello,", sys.argv[1])

# To run this from the terminal:
# python sayings.py David
```

### APIs and `requests`

You can fetch data from the internet using APIs. The `requests` library is the standard for this.

```python
# First, install it:
# pip install requests

import requests
import json
import sys

if len(sys.argv) != 2:
    sys.exit()

response = requests.get("https://itunes.apple.com/search?entity=song&limit=5&term=" + sys.argv[1])

o = response.json()
for result in o["results"]:
    print(result["trackName"])
```

## Unit Testing with `pytest`

Unit tests are small, automated tests to check if your functions are working correctly.

Let's say we have a `calculator.py` file:
```python
# calculator.py
def main():
    x = int(input("What's x? "))
    print("x squared is", square(x))

def square(n):
    return n * n

if __name__ == "__main__":
    main()
```

We can create a `test_calculator.py` file to test it. `pytest` automatically discovers and runs tests in files named `test_*.py` or `*_test.py`.

```python
# test_calculator.py
from calculator import square

def test_square():
    assert square(2) == 4
    assert square(3) == 9
    assert square(-2) == 4
    assert square(0) == 0
```
To run the tests, simply type `pytest` in your terminal.

## File I/O

Reading from and writing to files.

### Writing to a File

The `open()` function is used to interact with files. Using the `with` statement is the recommended way, as it handles closing the file for you.

```python
name = input("What's your name? ")

# 'a' stands for "append" mode, which adds to the end of the file.
# 'w' for "write" would overwrite the file each time.
with open("names.txt", "a") as file:
    file.write(f"{name}\n")
```

### Reading from a File

You can read all lines into a list or iterate through the file line by line.

```python
# Reading and sorting names from a file
with open("names.txt") as file:
    for line in sorted(file, reverse=True): # You can sort directly!
        print("hello,", line.rstrip()) # rstrip() removes trailing whitespace, including newlines
```

### Working with CSV Files

The `csv` module makes it easy to work with Comma-Separated Values files.

```python
# Reading a CSV
import csv

students = []
with open("students.csv") as file:
    reader = csv.DictReader(file) # Reads rows as dictionaries
    for row in reader:
        students.append({"name": row["name"], "home": row["home"]})

for student in sorted(students, key=lambda s: s["name"]):
    print(f"{student['name']} is from {student['home']}")
```

`lambda` creates a small, anonymous function, perfect for simple `key` operations in sorting.

```python
# Writing to a CSV
import csv

name = input("What's your name? ")
home = input("Where's your home? ")

with open("students.csv", "a", newline="") as file:
    # fieldnames should match your CSV header
    writer = csv.DictWriter(file, fieldnames=["name", "home"])
    writer.writerow({"name": name, "home": home})
```

## Regular Expressions (Regex)

Regex is a powerful mini-language for finding and manipulating patterns in text. The `re` module is Python's tool for this.

### Validating an Email Address

Let's build up a regex to validate an email address.

```python
import re

email = input("What's your email? ").strip()

# A robust pattern for many common emails
if re.search(r"^\w+@(\w+\.)?\w+\.(com|edu|gov|net|org)$", email, re.IGNORECASE):
    print("Valid")
else:
    print("Invalid")
```
**Pattern Breakdown:**
- `^`: Start of the string.
- `\w+`: One or more "word" characters (letters, numbers, underscore).
- `@`: A literal "@".
- `(\w+\.)?`: An optional subdomain (like `students.` in `students.cs50.org`).
- `\w+`: The domain name.
- `\.`: A literal ".".
- `(com|edu|...)`: A group of allowed top-level domains.
- `$`: End of the string.
- `re.IGNORECASE`: Flag to make the pattern case-insensitive.

### Formatting User Input

Regex is great for parsing and reformatting text. Let's fix names entered as "Last, First".

```python
import re

name = input("What's your name? ").strip()

# The Walrus Operator (:=) assigns and checks in one step
if matches := re.search(r"^(.+), *(.+)$
", name):
    # Group 2 is the first name, Group 1 is the last name
    name = matches.group(2) + " " + matches.group(1)

print(f"hello, {name}")
```

## Object-Oriented Programming (OOP)

OOP allows you to model real-world things by creating custom data types (classes).

### Classes and Objects

A class is a blueprint. An object (or instance) is a specific thing created from that blueprint.

```python
class Student:
    def __init__(self, name, house):
        self.name = name
        self.house = house

    def __str__(self):
        return f"{self.name} from {self.house}"

    @classmethod
    def get(cls): # A class method belongs to the class, not an instance
        name = input("Name: ")
        house = input("House: ")
        return cls(name, house) # Creates a new Student object

def main():
    student = Student.get()
    print(student)

if __name__ == "__main__":
    main()
```
- `__init__`: The constructor, called when a new object is created.
- `self`: Refers to the specific instance of the object being worked on.
- `__str__`: A special method that defines what `print(object)` should display.
- `@classmethod`: A decorator indicating a method that operates on the class itself (`cls`), not an instance (`self`).

### Properties (Getters and Setters)

Properties allow you to add logic to getting and setting attributes, like for validation.

```python
class Student:
    def __init__(self, name, house):
        self.name = name
        self.house = house # This will call the setter below

    # ... (str method) ...

    # Getter for 'house'
    @property
    def house(self):
        return self._house # Note the underscore

    # Setter for 'house'
    @house.setter
    def house(self, house):
        if house not in ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]:
            raise ValueError("Invalid house")
        self._house = house
```

### Inheritance

Inheritance allows a class (child) to inherit attributes and methods from another class (parent).

```python
class Wizard:
    def __init__(self, name):
        if not name:
            raise ValueError("Missing name")
        self.name = name

class Student(Wizard): # Student inherits from Wizard
    def __init__(self, name, house):
        super().__init__(name) # Calls the parent's __init__
        self.house = house

class Professor(Wizard): # Professor also inherits from Wizard
    def __init__(self, name, subject):
        super().__init__(name)
        self.subject = subject
```

### Operator Overloading

You can define what operators like `+`, `-`, etc., do for your custom objects.

```python
class Vault:
    def __init__(self, galleons=0, sickles=0, knuts=0):
        self.galleons = galleons
        self.sickles = sickles
        self.knuts = knuts

    def __str__(self):
        return f"{self.galleons}G, {self.sickles}S, {self.knuts}K"

    # Define what '+' means for Vault objects
    def __add__(self, other):
        galleons = self.galleons + other.galleons
        sickles = self.sickles + other.sickles
        knuts = self.knuts + other.knuts
        return Vault(galleons, sickles, knuts)

potter = Vault(100, 50, 25)
weasley = Vault(25, 50, 100)

total = potter + weasley
print(total) # Output: 125G, 100S, 125K
```

## More Data Structures

### Sets

A set is an unordered collection of **unique** items.

```python
students = [
    {"name": "Hermione", "house": "Gryffindor"},
    {"name": "Harry", "house": "Gryffindor"},
    {"name": "Padma", "house": "Ravenclaw"},
]

# Use a set to automatically find the unique houses
houses = set()
for student in students:
    houses.add(student["house"])

for house in sorted(houses):
    print(house)
```

## Global Variables and Constants

### Global Variables

Variables defined outside of any function are global. To modify them from within a function, you must use the `global` keyword. However, this is often discouraged in favor of passing variables or using classes.

```python
class Account:
    def __init__(self):
        self._balance = 0

    @property
    def balance(self):
        return self._balance

    def deposit(self, n):
        self._balance += n

    def withdraw(self, n):
        self._balance -= n

def main():
    account = Account()
    print("Balance:", account.balance)
    account.deposit(100)
    account.withdraw(50)
    print("Balance:", account.balance)

main()
```

### Constants

In Python, constants are variables that are not intended to be changed. By convention, their names are written in all `UPPERCASE`.

```python
MEOWS = 3

for _ in range(MEOWS):
    print("meow")
```

## Type Hinting with `mypy`

Python is dynamically typed, but you can add "type hints" to your code for clarity and to allow for static analysis. The `mypy` tool can then check your code for type consistency.

```python
# First, install it:
# pip install mypy

# This function expects an integer.
def meow(n: int) -> None: # '-> None' means the function doesn't return anything
    for _ in range(n):
        print("meow")

number: int = int(input("Number: "))
meow(number)
```
If you try to pass a string to `meow()`, `mypy` will catch the error before you even run the code. You can run `mypy` from the terminal: `mypy your_file.py`.

