# Python Basics for Data Science

A comprehensive guide to essential Python concepts for AI/ML and data science applications.

## Table of Contents
1. [Python Fundamentals](#python-fundamentals)
2. [Data Types and Structures](#data-types-and-structures)
3. [Control Flow](#control-flow)
4. [Functions](#functions)
5. [File I/O](#file-io)
6. [Error Handling](#error-handling)
7. [List Comprehensions](#list-comprehensions)
8. [Working with Libraries](#working-with-libraries)
9. [Best Practices](#best-practices)

## Python Fundamentals

### Variables and Assignment

```python
# Basic variable assignment
name = "Data Scientist"
age = 25
salary = 75000.50
is_experienced = True

# Multiple assignment
x, y, z = 1, 2, 3

# Type checking
print(type(name))      # <class 'str'>
print(type(age))       # <class 'int'>
print(type(salary))    # <class 'float'>
print(type(is_experienced))  # <class 'bool'>
```

### Basic Operations

```python
# Arithmetic operations
a, b = 10, 3
print(f"Addition: {a + b}")        # 13
print(f"Subtraction: {a - b}")     # 7
print(f"Multiplication: {a * b}")  # 30
print(f"Division: {a / b}")        # 3.3333...
print(f"Floor division: {a // b}") # 3
print(f"Modulus: {a % b}")         # 1
print(f"Power: {a ** b}")          # 1000

# String operations
first_name = "John"
last_name = "Doe"
full_name = first_name + " " + last_name
print(full_name)  # "John Doe"

# String formatting
age = 30
print(f"I am {age} years old")
print("I am {} years old".format(age))
print("I am %d years old" % age)
```

## Data Types and Structures

### Numbers

```python
# Integers
int_num = 42
big_int = 1234567890123456789

# Floating point
float_num = 3.14159
scientific = 1.23e-4

# Complex numbers
complex_num = 3 + 4j

# Type conversion
float_to_int = int(3.9)    # 3
int_to_float = float(42)   # 42.0
str_to_int = int("123")    # 123
```

### Strings

```python
# String creation
text = "Hello, World!"
text2 = 'Python is awesome'
multiline = """
This is a
multi-line string
"""

# String methods
text = "  hello world  "
print(text.upper())        # "  HELLO WORLD  "
print(text.lower())        # "  hello world  "
print(text.strip())        # "hello world"
print(text.replace("world", "python"))  # "  hello python  "
print(text.split())        # ['hello', 'world']

# String indexing and slicing
text = "Python"
print(text[0])     # 'P'
print(text[-1])    # 'n'
print(text[1:4])   # 'yth'
print(text[:3])    # 'Pyt'
print(text[3:])    # 'hon'
print(text[::2])   # 'Pto'
print(text[::-1])  # 'nohtyP'
```

### Lists

```python
# Creating lists
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
empty = []

# List operations
numbers.append(6)          # [1, 2, 3, 4, 5, 6]
numbers.insert(0, 0)      # [0, 1, 2, 3, 4, 5, 6]
numbers.remove(3)          # [0, 1, 2, 4, 5, 6]
popped = numbers.pop()     # 6, list becomes [0, 1, 2, 4, 5]
numbers.sort()             # [0, 1, 2, 4, 5]
numbers.reverse()          # [5, 4, 2, 1, 0]

# List slicing
numbers = [0, 1, 2, 3, 4, 5]
print(numbers[1:4])    # [1, 2, 3]
print(numbers[::2])    # [0, 2, 4]
print(numbers[::-1])   # [5, 4, 3, 2, 1, 0]

# List comprehension
squares = [x**2 for x in range(5)]  # [0, 1, 4, 9, 16]
evens = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]
```

### Tuples

```python
# Tuples are immutable
coordinates = (10, 20)
person = ("John", 30, "Engineer")

# Tuple unpacking
x, y = coordinates
name, age, job = person

# Single element tuple
single = (42,)  # Note the comma
```

### Dictionaries

```python
# Creating dictionaries
person = {
    "name": "John Doe",
    "age": 30,
    "city": "New York",
    "skills": ["Python", "ML", "Data Science"]
}

# Accessing values
print(person["name"])           # "John Doe"
print(person.get("age"))        # 30
print(person.get("salary", 0))  # 0 (default value)

# Modifying dictionaries
person["age"] = 31
person["salary"] = 75000
person.update({"experience": 5, "education": "MS"})

# Dictionary methods
print(person.keys())    # dict_keys(['name', 'age', 'city', 'skills', 'salary', 'experience', 'education'])
print(person.values())  # dict_values(['John Doe', 31, 'New York', ['Python', 'ML', 'Data Science'], 75000, 5, 'MS'])
print(person.items())   # dict_items([('name', 'John Doe'), ...])

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

### Sets

```python
# Creating sets
numbers = {1, 2, 3, 4, 5}
fruits = set(['apple', 'banana', 'orange'])

# Set operations
numbers.add(6)           # {1, 2, 3, 4, 5, 6}
numbers.remove(1)        # {2, 3, 4, 5, 6}
numbers.discard(10)      # No error if element doesn't exist

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
print(set1 | set2)       # Union: {1, 2, 3, 4, 5, 6}
print(set1 & set2)       # Intersection: {3, 4}
print(set1 - set2)       # Difference: {1, 2}
print(set1 ^ set2)       # Symmetric difference: {1, 2, 5, 6}
```

## Control Flow

### Conditional Statements

```python
# if-elif-else
age = 25
if age < 18:
    print("Minor")
elif age < 65:
    print("Adult")
else:
    print("Senior")

# Ternary operator
status = "adult" if age >= 18 else "minor"

# Multiple conditions
score = 85
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "F"
```

### Loops

```python
# for loop
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# Iterating over lists
fruits = ['apple', 'banana', 'orange']
for fruit in fruits:
    print(fruit)

# Iterating over dictionaries
person = {"name": "John", "age": 30}
for key, value in person.items():
    print(f"{key}: {value}")

# while loop
count = 0
while count < 5:
    print(count)
    count += 1

# Loop control
for i in range(10):
    if i == 3:
        continue  # Skip iteration
    if i == 7:
        break     # Exit loop
    print(i)
```

## Functions

### Basic Functions

```python
# Function definition
def greet(name):
    return f"Hello, {name}!"

# Function call
message = greet("Alice")
print(message)  # "Hello, Alice!"

# Function with default parameters
def greet_with_title(name, title="Mr."):
    return f"Hello, {title} {name}!"

print(greet_with_title("Smith"))           # "Hello, Mr. Smith!"
print(greet_with_title("Johnson", "Dr."))  # "Hello, Dr. Johnson!"

# Function with multiple return values
def get_name_and_age():
    return "John", 30

name, age = get_name_and_age()
```

### Advanced Functions

```python
# Variable number of arguments
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3, 4, 5))  # 15

# Keyword arguments
def create_profile(**kwargs):
    return kwargs

profile = create_profile(name="John", age=30, city="NYC")
print(profile)  # {'name': 'John', 'age': 30, 'city': 'NYC'}

# Lambda functions
square = lambda x: x**2
print(square(5))  # 25

# Using lambda with built-in functions
numbers = [1, 2, 3, 4, 5]
squares = list(map(lambda x: x**2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
```

## File I/O

### Reading Files

```python
# Reading entire file
with open('data.txt', 'r') as file:
    content = file.read()
    print(content)

# Reading line by line
with open('data.txt', 'r') as file:
    for line in file:
        print(line.strip())

# Reading all lines into a list
with open('data.txt', 'r') as file:
    lines = file.readlines()

# Reading CSV-like data
with open('data.csv', 'r') as file:
    for line in file:
        values = line.strip().split(',')
        print(values)
```

### Writing Files

```python
# Writing to a file
with open('output.txt', 'w') as file:
    file.write("Hello, World!\n")
    file.write("This is a test file.\n")

# Appending to a file
with open('output.txt', 'a') as file:
    file.write("This line is appended.\n")

# Writing multiple lines
lines = ["Line 1", "Line 2", "Line 3"]
with open('output.txt', 'w') as file:
    file.writelines(line + '\n' for line in lines)
```

## Error Handling

### Try-Except Blocks

```python
# Basic error handling
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Multiple exception types
try:
    value = int("abc")
except ValueError:
    print("Invalid integer!")
except TypeError:
    print("Wrong type!")
except Exception as e:
    print(f"An error occurred: {e}")

# Try-except with else and finally
try:
    number = int(input("Enter a number: "))
except ValueError:
    print("Invalid input!")
else:
    print(f"You entered: {number}")
finally:
    print("This always executes")
```

### Raising Exceptions

```python
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return a / b

# Custom exception
class DataValidationError(Exception):
    pass

def validate_age(age):
    if age < 0 or age > 150:
        raise DataValidationError("Invalid age!")
    return age
```

## List Comprehensions

### Basic List Comprehensions

```python
# Basic syntax: [expression for item in iterable]
squares = [x**2 for x in range(5)]  # [0, 1, 4, 9, 16]

# With condition: [expression for item in iterable if condition]
evens = [x for x in range(10) if x % 2 == 0]  # [0, 2, 4, 6, 8]

# Nested comprehensions
matrix = [[i+j for j in range(3)] for i in range(3)]
# [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
```

### Advanced List Comprehensions

```python
# Dictionary comprehension
squares_dict = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# Set comprehension
unique_squares = {x**2 for x in range(5)}
# {0, 1, 4, 9, 16}

# Generator expression (memory efficient)
squares_gen = (x**2 for x in range(5))
for square in squares_gen:
    print(square)
```

## Working with Libraries

### Importing Libraries

```python
# Standard imports
import math
import random
import datetime

# Import specific functions
from math import sqrt, pi
from random import choice, randint

# Import with alias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import all (not recommended)
from math import *
```

### Common Standard Library Modules

```python
# Math operations
import math
print(math.sqrt(16))      # 4.0
print(math.pi)           # 3.141592653589793
print(math.ceil(3.2))    # 4
print(math.floor(3.8))   # 3

# Random numbers
import random
print(random.random())    # Random float between 0 and 1
print(random.randint(1, 10))  # Random integer between 1 and 10
print(random.choice(['a', 'b', 'c']))  # Random choice from list

# Date and time
import datetime
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# File system operations
import os
import glob

# List files in directory
files = os.listdir('.')
print(files)

# Find files with pattern
csv_files = glob.glob('*.csv')
print(csv_files)
```

## Best Practices

### Code Style (PEP 8)

```python
# Use meaningful variable names
user_age = 25  # Good
ua = 25        # Bad

# Use snake_case for variables and functions
def calculate_average(numbers):
    return sum(numbers) / len(numbers)

# Use CamelCase for classes
class DataProcessor:
    def __init__(self):
        pass

# Use UPPERCASE for constants
MAX_RETRIES = 3
PI = 3.14159

# Proper indentation (4 spaces)
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
```

### Documentation

```python
def calculate_statistics(data):
    """
    Calculate basic statistics for a list of numbers.
    
    Args:
        data (list): List of numerical values
        
    Returns:
        dict: Dictionary containing mean, median, and standard deviation
        
    Raises:
        ValueError: If data is empty or contains non-numeric values
        
    Example:
        >>> calculate_statistics([1, 2, 3, 4, 5])
        {'mean': 3.0, 'median': 3.0, 'std': 1.5811388300841898}
    """
    if not data:
        raise ValueError("Data cannot be empty")
    
    # Implementation here
    pass
```

### Performance Tips

```python
# Use list comprehensions instead of loops when possible
# Good
squares = [x**2 for x in range(1000)]

# Less efficient
squares = []
for x in range(1000):
    squares.append(x**2)

# Use sets for membership testing
numbers = set(range(1000))
if 500 in numbers:  # O(1) average case
    print("Found")

# Use generators for large datasets
def generate_numbers(n):
    for i in range(n):
        yield i

# Memory efficient
for num in generate_numbers(1000000):
    process(num)
```

### Testing

```python
import unittest

class TestCalculator(unittest.TestCase):
    def test_addition(self):
        self.assertEqual(2 + 2, 4)
    
    def test_division(self):
        self.assertEqual(10 / 2, 5)
        with self.assertRaises(ZeroDivisionError):
            10 / 0

if __name__ == '__main__':
    unittest.main()
```

## Exercises

1. **Basic Operations**: Write a function that calculates the area and perimeter of a rectangle.
2. **List Manipulation**: Create a function that removes duplicates from a list while preserving order.
3. **String Processing**: Write a function that counts the frequency of each character in a string.
4. **File Processing**: Create a script that reads a CSV file and calculates basic statistics.
5. **Error Handling**: Write a function that safely converts a string to an integer with error handling.

## Next Steps

After mastering these Python basics, explore:
- [Object-Oriented Programming](oop_guide.md)
- [Functional Programming](functional_programming_guide.md)
- [Data Manipulation](data_manipulation_guide.md)
- [Working with NumPy and Pandas](../NumPy/numpy_basics_guide.md) 