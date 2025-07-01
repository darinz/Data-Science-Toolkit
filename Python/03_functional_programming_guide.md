# Python Functional Programming Guide

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Functional](https://img.shields.io/badge/Functional-Programming-green.svg)](https://docs.python.org/3/howto/functional.html)

A comprehensive guide to Functional Programming concepts in Python for data science and machine learning applications.

## Table of Contents

1. [Introduction to Functional Programming](#introduction-to-functional-programming)
2. [Pure Functions](#pure-functions)
3. [Lambda Functions](#lambda-functions)
4. [Higher-Order Functions](#higher-order-functions)
5. [Map, Filter, and Reduce](#map-filter-and-reduce)
6. [List Comprehensions](#list-comprehensions)
7. [Generator Expressions](#generator-expressions)
8. [Decorators](#decorators)
9. [Function Composition](#function-composition)
10. [Functional Data Processing](#functional-data-processing)
11. [Best Practices](#best-practices)

## Introduction to Functional Programming

Functional Programming (FP) is a programming paradigm that treats computation as the evaluation of mathematical functions and avoids changing state and mutable data.

### Key Concepts

- **Pure Functions**: Functions with no side effects
- **Immutability**: Data cannot be changed after creation
- **Higher-Order Functions**: Functions that take or return other functions
- **Function Composition**: Combining functions to create new functions
- **Recursion**: Functions calling themselves

### Why Functional Programming in Data Science?

```python
import numpy as np
import pandas as pd

# Imperative approach
def process_data_imperative(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result

# Functional approach
def process_data_functional(data):
    return list(map(lambda x: x * 2, filter(lambda x: x > 0, data)))

# Even more functional with list comprehension
def process_data_comprehension(data):
    return [x * 2 for x in data if x > 0]

# Test
data = [-2, -1, 0, 1, 2, 3]
print(process_data_imperative(data))  # [2, 4, 6]
print(process_data_functional(data))  # [2, 4, 6]
print(process_data_comprehension(data))  # [2, 4, 6]
```

## Pure Functions

### Characteristics of Pure Functions

```python
# Pure function - same input always produces same output, no side effects
def pure_square(x):
    """Pure function: always returns the same result for the same input."""
    return x ** 2

# Impure function - has side effects
total = 0
def impure_add(x):
    """Impure function: modifies global state."""
    global total
    total += x
    return total

# Test pure function
print(pure_square(5))  # 25
print(pure_square(5))  # 25 (always the same)

# Test impure function
print(impure_add(5))   # 5
print(impure_add(5))   # 10 (different result!)
```

### Benefits of Pure Functions

```python
# Pure functions are easier to test
def test_pure_function():
    assert pure_square(2) == 4
    assert pure_square(0) == 0
    assert pure_square(-3) == 9
    print("All tests passed!")

# Pure functions can be memoized (cached)
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(x):
    """Simulate expensive calculation."""
    import time
    time.sleep(0.1)  # Simulate computation
    return x ** 2 + x + 1

# First call is slow
print(expensive_calculation(5))  # Takes 0.1 seconds

# Subsequent calls are fast (cached)
print(expensive_calculation(5))  # Instant!
```

## Lambda Functions

### Basic Lambda Functions

```python
# Lambda function syntax: lambda arguments: expression
square = lambda x: x ** 2
add = lambda x, y: x + y
is_even = lambda x: x % 2 == 0

print(square(5))      # 25
print(add(3, 4))      # 7
print(is_even(6))     # True
print(is_even(7))     # False

# Lambda functions are often used inline
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25]
```

### Lambda Functions in Data Science

```python
import pandas as pd

# Lambda with pandas
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50]
})

# Apply lambda to create new column
df['C'] = df['A'].apply(lambda x: x * 2 + 1)
print(df)

# Lambda with conditional logic
df['D'] = df['A'].apply(lambda x: 'even' if x % 2 == 0 else 'odd')
print(df)

# Lambda with multiple conditions
def categorize_value(x):
    if x < 2:
        return 'low'
    elif x < 4:
        return 'medium'
    else:
        return 'high'

# Equivalent lambda
categorize_lambda = lambda x: 'low' if x < 2 else 'medium' if x < 4 else 'high'

df['E'] = df['A'].apply(categorize_lambda)
print(df)
```

## Higher-Order Functions

### Functions as Arguments

```python
def apply_operation(func, data):
    """Higher-order function that takes a function as an argument."""
    return [func(item) for item in data]

def square(x):
    return x ** 2

def cube(x):
    return x ** 3

def double(x):
    return x * 2

# Use the same higher-order function with different operations
numbers = [1, 2, 3, 4, 5]

squared_numbers = apply_operation(square, numbers)
cubed_numbers = apply_operation(cube, numbers)
doubled_numbers = apply_operation(double, numbers)

print(squared_numbers)  # [1, 4, 9, 16, 25]
print(cubed_numbers)    # [1, 8, 27, 64, 125]
print(doubled_numbers)  # [2, 4, 6, 8, 10]
```

### Functions Returning Functions

```python
def create_multiplier(factor):
    """Higher-order function that returns a function."""
    def multiplier(x):
        return x * factor
    return multiplier

# Create specialized functions
double = create_multiplier(2)
triple = create_multiplier(3)
quadruple = create_multiplier(4)

print(double(5))    # 10
print(triple(5))    # 15
print(quadruple(5)) # 20

# More complex example
def create_filter_function(threshold, operation='greater'):
    """Create a filter function based on threshold and operation."""
    if operation == 'greater':
        return lambda x: x > threshold
    elif operation == 'less':
        return lambda x: x < threshold
    elif operation == 'equal':
        return lambda x: x == threshold
    else:
        raise ValueError(f"Unknown operation: {operation}")

# Create filter functions
greater_than_5 = create_filter_function(5, 'greater')
less_than_10 = create_filter_function(10, 'less')

numbers = [1, 3, 5, 7, 9, 11, 13]
filtered_greater = list(filter(greater_than_5, numbers))
filtered_less = list(filter(less_than_10, numbers))

print(filtered_greater)  # [7, 9, 11, 13]
print(filtered_less)     # [1, 3, 5, 7, 9]
```

## Map, Filter, and Reduce

### Map Function

```python
# Map applies a function to every item in an iterable
numbers = [1, 2, 3, 4, 5]

# Using map with a regular function
def square(x):
    return x ** 2

squared = list(map(square, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# Using map with lambda
squared_lambda = list(map(lambda x: x ** 2, numbers))
print(squared_lambda)  # [1, 4, 9, 16, 25]

# Map with multiple iterables
list1 = [1, 2, 3]
list2 = [10, 20, 30]

summed = list(map(lambda x, y: x + y, list1, list2))
print(summed)  # [11, 22, 33]

# Map with pandas
import pandas as pd
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df['sum'] = list(map(lambda x, y: x + y, df['A'], df['B']))
print(df)
```

### Filter Function

```python
# Filter creates an iterator of elements for which a function returns True
numbers = range(-5, 6)

# Filter positive numbers
positive = list(filter(lambda x: x > 0, numbers))
print(positive)  # [1, 2, 3, 4, 5]

# Filter even numbers
even = list(filter(lambda x: x % 2 == 0, numbers))
print(even)  # [-4, -2, 0, 2, 4]

# Filter with custom function
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = list(filter(is_prime, range(1, 20)))
print(primes)  # [2, 3, 5, 7, 11, 13, 17, 19]

# Filter with pandas
df = pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})
filtered_df = df[df['A'].apply(lambda x: x % 2 == 0)]
print(filtered_df)
```

### Reduce Function

```python
from functools import reduce

# Reduce applies a function of two arguments cumulatively to the items of an iterable
numbers = [1, 2, 3, 4, 5]

# Sum all numbers
total = reduce(lambda x, y: x + y, numbers)
print(total)  # 15

# Find maximum
maximum = reduce(lambda x, y: x if x > y else y, numbers)
print(maximum)  # 5

# Multiply all numbers
product = reduce(lambda x, y: x * y, numbers)
print(product)  # 120

# Reduce with initial value
total_with_initial = reduce(lambda x, y: x + y, numbers, 10)
print(total_with_initial)  # 25 (10 + 1 + 2 + 3 + 4 + 5)

# Custom reduce function
def custom_reduce(func, iterable, initial=None):
    """Custom implementation of reduce."""
    iterator = iter(iterable)
    if initial is None:
        try:
            result = next(iterator)
        except StopIteration:
            raise TypeError("reduce() of empty sequence with no initial value")
    else:
        result = initial
    
    for item in iterator:
        result = func(result, item)
    
    return result

# Test custom reduce
test_numbers = [1, 2, 3, 4]
print(custom_reduce(lambda x, y: x + y, test_numbers))  # 10
print(custom_reduce(lambda x, y: x + y, test_numbers, 5))  # 15
```

## List Comprehensions

### Basic List Comprehensions

```python
# List comprehension syntax: [expression for item in iterable]
numbers = [1, 2, 3, 4, 5]

# Square all numbers
squared = [x ** 2 for x in numbers]
print(squared)  # [1, 4, 9, 16, 25]

# Filter even numbers
even = [x for x in numbers if x % 2 == 0]
print(even)  # [2, 4]

# Combine operations
squared_even = [x ** 2 for x in numbers if x % 2 == 0]
print(squared_even)  # [4, 16]

# Nested list comprehension
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [item for row in matrix for item in row]
print(flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

### Advanced List Comprehensions

```python
# List comprehension with conditional expression
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Map with conditional logic
result = [x ** 2 if x % 2 == 0 else x ** 3 for x in numbers]
print(result)  # [1, 4, 27, 16, 125, 36, 343, 64, 729, 100]

# Multiple conditions
categorized = ['even' if x % 2 == 0 else 'odd' if x % 2 == 1 else 'zero' 
               for x in numbers]
print(categorized)

# List comprehension with function calls
def process_number(x):
    if x < 5:
        return x * 2
    else:
        return x + 10

processed = [process_number(x) for x in numbers]
print(processed)  # [2, 4, 6, 8, 15, 16, 17, 18, 19, 20]

# Dictionary comprehension
squares_dict = {x: x ** 2 for x in numbers}
print(squares_dict)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25, ...}

# Set comprehension
unique_squares = {x ** 2 for x in numbers}
print(unique_squares)  # {1, 4, 9, 16, 25, 36, 49, 64, 81, 100}
```

## Generator Expressions

### Basic Generator Expressions

```python
# Generator expression syntax: (expression for item in iterable)
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Generator expression (memory efficient)
squares_gen = (x ** 2 for x in numbers)
print(squares_gen)  # <generator object <genexpr>>

# Convert to list when needed
squares_list = list(squares_gen)
print(squares_list)  # [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# Generator expressions are lazy - they don't compute until needed
large_numbers = range(1000000)
squares_large = (x ** 2 for x in large_numbers)

# This doesn't create a large list in memory
print(next(squares_large))  # 0
print(next(squares_large))  # 1
print(next(squares_large))  # 4
```

### Generator Functions

```python
def number_generator(start, end):
    """Generator function that yields numbers."""
    current = start
    while current <= end:
        yield current
        current += 1

# Use the generator
for num in number_generator(1, 5):
    print(num)  # 1, 2, 3, 4, 5

# Generator for infinite sequences
def fibonacci():
    """Generate Fibonacci numbers infinitely."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# Get first 10 Fibonacci numbers
fib_gen = fibonacci()
fibonacci_10 = [next(fib_gen) for _ in range(10)]
print(fibonacci_10)  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# Generator for data processing
def process_data_stream(data_stream):
    """Process data stream with generator."""
    for item in data_stream:
        if item > 0:  # Filter positive numbers
            yield item * 2  # Transform

# Test with sample data
data = [-2, -1, 0, 1, 2, 3, 4, 5]
processed = list(process_data_stream(data))
print(processed)  # [2, 4, 6, 8, 10]
```

## Decorators

### Basic Decorators

```python
def timer(func):
    """Decorator to measure function execution time."""
    import time
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    
    return wrapper

@timer
def slow_function():
    """Function that takes time to execute."""
    import time
    time.sleep(1)
    return "Done!"

# Use the decorated function
result = slow_function()
print(result)

# Decorator with arguments
def repeat(times):
    """Decorator that repeats a function multiple times."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")  # Prints "Hello, Alice!" three times
```

### Advanced Decorators

```python
def cache(func):
    """Simple caching decorator."""
    cache_data = {}
    
    def wrapper(*args):
        if args not in cache_data:
            cache_data[args] = func(*args)
        return cache_data[args]
    
    return wrapper

@cache
def fibonacci(n):
    """Calculate Fibonacci number (inefficient without cache)."""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Test caching
print(fibonacci(10))  # 55 (computed)
print(fibonacci(10))  # 55 (from cache)

# Decorator for validation
def validate_positive(func):
    """Decorator to validate that arguments are positive."""
    def wrapper(*args, **kwargs):
        for arg in args:
            if isinstance(arg, (int, float)) and arg <= 0:
                raise ValueError(f"Argument {arg} must be positive")
        return func(*args, **kwargs)
    return wrapper

@validate_positive
def square_root(x):
    """Calculate square root of positive number."""
    import math
    return math.sqrt(x)

# Test validation
print(square_root(4))  # 2.0
# print(square_root(-1))  # ValueError: Argument -1 must be positive
```

## Function Composition

### Basic Function Composition

```python
def compose(*functions):
    """Compose multiple functions from right to left."""
    def inner(arg):
        result = arg
        for func in reversed(functions):
            result = func(result)
        return result
    return inner

# Example functions
def add_one(x):
    return x + 1

def multiply_by_two(x):
    return x * 2

def square(x):
    return x ** 2

# Compose functions
composed = compose(square, multiply_by_two, add_one)

# This is equivalent to: square(multiply_by_two(add_one(x)))
result = composed(3)
print(result)  # 64 (add_one(3) = 4, multiply_by_two(4) = 8, square(8) = 64)

# Alternative composition using lambda
composed_lambda = lambda x: square(multiply_by_two(add_one(x)))
print(composed_lambda(3))  # 64
```

### Pipeline Pattern

```python
class Pipeline:
    """Pipeline for chaining functions together."""
    
    def __init__(self):
        self.functions = []
    
    def add(self, func):
        """Add a function to the pipeline."""
        self.functions.append(func)
        return self
    
    def execute(self, data):
        """Execute all functions in the pipeline."""
        result = data
        for func in self.functions:
            result = func(result)
        return result

# Create a data processing pipeline
pipeline = (Pipeline()
    .add(lambda x: x * 2)
    .add(lambda x: x + 1)
    .add(lambda x: x ** 2))

result = pipeline.execute(3)
print(result)  # 49 ((3 * 2 + 1) ** 2 = 49)

# Pipeline with data validation
def validate_data(data):
    """Validate that data is a list of numbers."""
    if not isinstance(data, list):
        raise ValueError("Data must be a list")
    if not all(isinstance(x, (int, float)) for x in data):
        raise ValueError("All elements must be numbers")
    return data

def filter_positive(data):
    """Filter out non-positive numbers."""
    return [x for x in data if x > 0]

def normalize_data(data):
    """Normalize data to [0, 1] range."""
    if not data:
        return []
    min_val = min(data)
    max_val = max(data)
    if max_val == min_val:
        return [0.5] * len(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

# Create data processing pipeline
data_pipeline = (Pipeline()
    .add(validate_data)
    .add(filter_positive)
    .add(normalize_data))

# Test pipeline
test_data = [1, -2, 3, 0, 5, -1, 7]
result = data_pipeline.execute(test_data)
print(result)  # [0.0, 0.333..., 0.666..., 1.0]
```

## Functional Data Processing

### Data Processing with Functional Tools

```python
import pandas as pd
import numpy as np

# Sample data
data = [
    {'name': 'Alice', 'age': 25, 'salary': 50000},
    {'name': 'Bob', 'age': 30, 'salary': 60000},
    {'name': 'Charlie', 'age': 35, 'salary': 70000},
    {'name': 'Diana', 'age': 28, 'salary': 55000},
    {'name': 'Eve', 'age': 32, 'salary': 65000}
]

# Functional data processing
def process_employee_data(employees):
    """Process employee data using functional programming."""
    
    # Filter employees over 30
    older_employees = list(filter(lambda emp: emp['age'] > 30, employees))
    
    # Map to get names of older employees
    older_names = list(map(lambda emp: emp['name'], older_employees))
    
    # Calculate average salary using reduce
    total_salary = reduce(lambda acc, emp: acc + emp['salary'], employees, 0)
    avg_salary = total_salary / len(employees)
    
    # Create salary bands
    def get_salary_band(emp):
        salary = emp['salary']
        if salary < 55000:
            return 'Low'
        elif salary < 65000:
            return 'Medium'
        else:
            return 'High'
    
    employees_with_bands = list(map(
        lambda emp: {**emp, 'salary_band': get_salary_band(emp)},
        employees
    ))
    
    return {
        'older_employees': older_names,
        'average_salary': avg_salary,
        'employees_with_bands': employees_with_bands
    }

# Process the data
result = process_employee_data(data)
print(result)
```

### Functional Data Analysis

```python
def analyze_dataset(data, operations):
    """Analyze dataset using functional operations."""
    
    def apply_operation(data, operation):
        """Apply a single operation to the data."""
        op_type = operation['type']
        
        if op_type == 'filter':
            return list(filter(operation['condition'], data))
        elif op_type == 'map':
            return list(map(operation['transform'], data))
        elif op_type == 'reduce':
            return reduce(operation['reducer'], data, operation.get('initial', 0))
        else:
            raise ValueError(f"Unknown operation type: {op_type}")
    
    # Apply operations in sequence
    result = data
    for operation in operations:
        result = apply_operation(result, operation)
    
    return result

# Example usage
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

operations = [
    {
        'type': 'filter',
        'condition': lambda x: x % 2 == 0  # Filter even numbers
    },
    {
        'type': 'map',
        'transform': lambda x: x ** 2  # Square the numbers
    },
    {
        'type': 'reduce',
        'reducer': lambda acc, x: acc + x,  # Sum the results
        'initial': 0
    }
]

result = analyze_dataset(numbers, operations)
print(result)  # 220 (sum of squares of even numbers: 4 + 16 + 36 + 64 + 100)
```

## Best Practices

### 1. Prefer Pure Functions

```python
# Good - pure function
def calculate_statistics(data):
    """Calculate statistics without side effects."""
    if not data:
        return {'mean': 0, 'std': 0, 'count': 0}
    
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std = variance ** 0.5
    
    return {
        'mean': mean,
        'std': std,
        'count': len(data)
    }

# Bad - impure function with side effects
def calculate_statistics_impure(data):
    """Calculate statistics with side effects."""
    global last_calculation_time
    import time
    
    last_calculation_time = time.time()  # Side effect
    # ... calculation logic
    return result
```

### 2. Use Immutable Data Structures

```python
from collections import namedtuple
from dataclasses import dataclass

# Good - immutable data structure
@dataclass(frozen=True)
class DataPoint:
    x: float
    y: float
    label: str

# Good - named tuple (immutable)
Point = namedtuple('Point', ['x', 'y', 'label'])

# Bad - mutable dictionary
data_point = {'x': 1.0, 'y': 2.0, 'label': 'A'}
data_point['x'] = 3.0  # Can be modified
```

### 3. Use Function Composition

```python
# Good - compose functions
def process_data(data):
    """Process data using function composition."""
    pipeline = compose(
        lambda x: filter(lambda y: y > 0, x),
        lambda x: map(lambda y: y * 2, x),
        lambda x: list(x)
    )
    return pipeline(data)

# Bad - nested function calls
def process_data_nested(data):
    """Process data with nested function calls."""
    filtered = list(filter(lambda x: x > 0, data))
    doubled = list(map(lambda x: x * 2, filtered))
    return doubled
```

### 4. Use Generators for Large Datasets

```python
# Good - generator for memory efficiency
def process_large_dataset(data_stream):
    """Process large dataset using generator."""
    for item in data_stream:
        if item > 0:
            yield item * 2

# Bad - list comprehension for large data
def process_large_dataset_bad(data_stream):
    """Process large dataset using list (memory intensive)."""
    return [item * 2 for item in data_stream if item > 0]
```

### 5. Use Type Hints

```python
from typing import List, Callable, Optional, Union

def process_numbers(
    numbers: List[float],
    operation: Callable[[float], float],
    filter_func: Optional[Callable[[float], bool]] = None
) -> List[float]:
    """Process numbers with optional filtering."""
    if filter_func:
        numbers = list(filter(filter_func, numbers))
    return list(map(operation, numbers))

# Usage
result = process_numbers(
    [1, 2, 3, 4, 5],
    lambda x: x ** 2,
    lambda x: x % 2 == 0
)
print(result)  # [4, 16]
```

## Summary

Functional Programming in Python provides powerful tools for data science:

- **Pure Functions**: Predictable, testable code
- **Higher-Order Functions**: Flexible, reusable code
- **Map, Filter, Reduce**: Efficient data processing
- **List Comprehensions**: Concise data transformations
- **Generators**: Memory-efficient data processing
- **Decorators**: Code reuse and enhancement
- **Function Composition**: Building complex operations from simple ones

Mastering functional programming concepts will help you write more maintainable, efficient, and elegant data science code.

## Next Steps

- Practice writing pure functions for your data processing tasks
- Explore functional programming libraries like `toolz` and `fn`
- Learn about monads and other advanced functional concepts
- Study how functional programming is used in data science frameworks

---

**Happy Functional Programming!** 🐍✨ 