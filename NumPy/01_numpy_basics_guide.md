# NumPy Basics: A Comprehensive Guide

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

## Table of Contents

1. [Introduction to NumPy](#introduction-to-numpy)
2. [Array Creation](#array-creation)
3. [Array Attributes and Properties](#array-attributes-and-properties)
4. [Data Types](#data-types)
5. [Basic Operations](#basic-operations)
6. [Indexing and Slicing](#indexing-and-slicing)
7. [Mathematical Functions](#mathematical-functions)
8. [Statistical Functions](#statistical-functions)
9. [Shape Manipulation](#shape-manipulation)
10. [Broadcasting](#broadcasting)
11. [Best Practices](#best-practices)
12. [Common Pitfalls](#common-pitfalls)
13. [Performance Tips](#performance-tips)

## Introduction to NumPy

NumPy (Numerical Python) is the fundamental package for scientific computing in Python. It provides:

- **Multidimensional array objects** with fast operations
- **Mathematical functions** for array operations
- **Linear algebra** capabilities
- **Random number generation**
- **Fourier transforms** and other mathematical tools

### Why NumPy?

```python
import numpy as np

# Python lists (slow for numerical operations)
python_list = [1, 2, 3, 4, 5]
result = [x * 2 for x in python_list]  # List comprehension

# NumPy arrays (fast vectorized operations)
numpy_array = np.array([1, 2, 3, 4, 5])
result = numpy_array * 2  # Vectorized operation
```

**Key Advantages:**
- **Speed**: C-optimized operations
- **Memory efficiency**: Contiguous memory layout
- **Vectorization**: Operations on entire arrays
- **Rich ecosystem**: Integration with other scientific libraries

## Array Creation

### 1. From Python Lists

```python
import numpy as np

# 1D array
arr1d = np.array([1, 2, 3, 4, 5])
print(f"1D Array: {arr1d}")
print(f"Shape: {arr1d.shape}")
print(f"Type: {arr1d.dtype}")

# 2D array (matrix)
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n2D Array:\n{arr2d}")
print(f"Shape: {arr2d.shape}")

# 3D array
arr3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"\n3D Array:\n{arr3d}")
print(f"Shape: {arr3d.shape}")
```

### 2. Using NumPy Functions

```python
# Zeros array
zeros_1d = np.zeros(5)
zeros_2d = np.zeros((3, 4))
print(f"Zeros 1D: {zeros_1d}")
print(f"Zeros 2D:\n{zeros_2d}")

# Ones array
ones_1d = np.ones(5)
ones_2d = np.ones((2, 3))
print(f"\nOnes 1D: {ones_1d}")
print(f"Ones 2D:\n{ones_2d}")

# Empty array (uninitialized memory)
empty_arr = np.empty((2, 3))
print(f"\nEmpty array:\n{empty_arr}")

# Identity matrix
identity = np.eye(3)
print(f"\nIdentity matrix:\n{identity}")

# Diagonal matrix
diagonal = np.diag([1, 2, 3, 4])
print(f"\nDiagonal matrix:\n{diagonal}")
```

### 3. Range and Sequence Arrays

```python
# arange (similar to range)
arr_range = np.arange(0, 10, 2)  # start, stop, step
print(f"arange: {arr_range}")

# linspace (linear spacing)
arr_linspace = np.linspace(0, 1, 5)  # start, stop, num_points
print(f"linspace: {arr_linspace}")

# logspace (logarithmic spacing)
arr_logspace = np.logspace(0, 2, 5)  # 10^0 to 10^2, 5 points
print(f"logspace: {arr_logspace}")

# meshgrid (for 2D coordinate grids)
x = np.linspace(-2, 2, 5)
y = np.linspace(-2, 2, 5)
X, Y = np.meshgrid(x, y)
print(f"\nX grid:\n{X}")
print(f"Y grid:\n{Y}")
```

### 4. Random Arrays

```python
# Random numbers between 0 and 1
random_uniform = np.random.random((3, 3))
print(f"Random uniform:\n{random_uniform}")

# Random integers
random_ints = np.random.randint(0, 10, (3, 3))
print(f"\nRandom integers:\n{random_ints}")

# Normal distribution
random_normal = np.random.normal(0, 1, (3, 3))
print(f"\nRandom normal:\n{random_normal}")
```

## Array Attributes and Properties

```python
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Basic attributes
print(f"Array:\n{arr}")
print(f"Shape: {arr.shape}")
print(f"Size: {arr.size}")
print(f"Number of dimensions: {arr.ndim}")
print(f"Data type: {arr.dtype}")
print(f"Item size: {arr.itemsize} bytes")
print(f"Total memory: {arr.nbytes} bytes")

# Array flags
print(f"\nFlags:")
print(f"  C-contiguous: {arr.flags.c_contiguous}")
print(f"  F-contiguous: {arr.flags.f_contiguous}")
print(f"  Writeable: {arr.flags.writeable}")
print(f"  Aligned: {arr.flags.aligned}")
```

## Data Types

NumPy provides various data types for different use cases:

```python
# Integer types
int8_arr = np.array([1, 2, 3], dtype=np.int8)
int16_arr = np.array([1, 2, 3], dtype=np.int16)
int32_arr = np.array([1, 2, 3], dtype=np.int32)
int64_arr = np.array([1, 2, 3], dtype=np.int64)

print(f"int8: {int8_arr.dtype}, size: {int8_arr.itemsize}")
print(f"int16: {int16_arr.dtype}, size: {int16_arr.itemsize}")
print(f"int32: {int32_arr.dtype}, size: {int32_arr.itemsize}")
print(f"int64: {int64_arr.dtype}, size: {int64_arr.itemsize}")

# Float types
float32_arr = np.array([1.1, 2.2, 3.3], dtype=np.float32)
float64_arr = np.array([1.1, 2.2, 3.3], dtype=np.float64)

print(f"\nfloat32: {float32_arr.dtype}, size: {float32_arr.itemsize}")
print(f"float64: {float64_arr.dtype}, size: {float64_arr.itemsize}")

# Complex types
complex64_arr = np.array([1+2j, 3+4j], dtype=np.complex64)
complex128_arr = np.array([1+2j, 3+4j], dtype=np.complex128)

print(f"\ncomplex64: {complex64_arr.dtype}, size: {complex64_arr.itemsize}")
print(f"complex128: {complex128_arr.dtype}, size: {complex128_arr.itemsize}")

# Type conversion
arr = np.array([1, 2, 3, 4])
float_arr = arr.astype(np.float32)
print(f"\nOriginal: {arr.dtype}")
print(f"Converted: {float_arr.dtype}")
```

## Basic Operations

### 1. Arithmetic Operations

```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

# Element-wise operations
print(f"a: {a}")
print(f"b: {b}")
print(f"Addition: {a + b}")
print(f"Subtraction: {a - b}")
print(f"Multiplication: {a * b}")
print(f"Division: {a / b}")
print(f"Power: {a ** 2}")
print(f"Modulo: {a % 2}")

# In-place operations
a += 1
print(f"\nAfter in-place addition: {a}")

# Broadcasting with scalars
print(f"\nArray + scalar: {a + 10}")
print(f"Array * scalar: {a * 2}")
```

### 2. Comparison Operations

```python
arr = np.array([1, 2, 3, 4, 5])

# Element-wise comparisons
print(f"Array: {arr}")
print(f"Greater than 3: {arr > 3}")
print(f"Less than or equal to 3: {arr <= 3}")
print(f"Equal to 3: {arr == 3}")
print(f"Not equal to 3: {arr != 3}")

# Boolean operations
mask1 = arr > 2
mask2 = arr < 5
print(f"\nMask1 (arr > 2): {mask1}")
print(f"Mask2 (arr < 5): {mask2}")
print(f"AND: {mask1 & mask2}")
print(f"OR: {mask1 | mask2}")
print(f"NOT: {~mask1}")
```

### 3. Logical Operations

```python
# Logical functions
arr = np.array([True, False, True, False])
print(f"Array: {arr}")
print(f"Logical AND: {np.logical_and(arr, [True, True, False, False])}")
print(f"Logical OR: {np.logical_or(arr, [True, True, False, False])}")
print(f"Logical NOT: {np.logical_not(arr)}")
print(f"Logical XOR: {np.logical_xor(arr, [True, True, False, False])}")

# Any and All
print(f"\nAny True: {np.any(arr)}")
print(f"All True: {np.all(arr)}")
```

## Indexing and Slicing

### 1. Basic Indexing

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(f"Array:\n{arr}")
print(f"Shape: {arr.shape}")

# Single element indexing
print(f"\nElement at [0, 1]: {arr[0, 1]}")
print(f"Element at [2, 3]: {arr[2, 3]}")

# Row indexing
print(f"\nFirst row: {arr[0]}")
print(f"Second row: {arr[1]}")

# Column indexing
print(f"\nFirst column: {arr[:, 0]}")
print(f"Second column: {arr[:, 1]}")
```

### 2. Slicing

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(f"Original array:\n{arr}")

# Basic slicing
print(f"\nFirst two rows:\n{arr[:2]}")
print(f"Last two rows:\n{arr[-2:]}")
print(f"First two columns:\n{arr[:, :2]}")
print(f"Last two columns:\n{arr[:, -2:]}")

# Subarray
print(f"\nSubarray [1:3, 1:3]:\n{arr[1:3, 1:3]}")

# Step slicing
print(f"\nEvery other row:\n{arr[::2]}")
print(f"Every other column:\n{arr[:, ::2]}")
print(f"Reverse array:\n{arr[::-1]}")
```

### 3. Advanced Indexing

```python
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(f"Original array:\n{arr}")

# Integer array indexing
indices = [0, 2]
print(f"\nRows 0 and 2:\n{arr[indices]}")

# Boolean indexing
mask = arr > 5
print(f"\nMask (arr > 5):\n{mask}")
print(f"Elements > 5: {arr[mask]}")

# Combined indexing
print(f"\nElements > 5 in first two rows: {arr[:2][arr[:2] > 5]}")
```

## Mathematical Functions

### 1. Basic Mathematical Functions

```python
arr = np.array([1, 2, 3, 4, 5])

print(f"Array: {arr}")
print(f"Square root: {np.sqrt(arr)}")
print(f"Square: {np.square(arr)}")
print(f"Exponential: {np.exp(arr)}")
print(f"Natural log: {np.log(arr)}")
print(f"Log base 10: {np.log10(arr)}")
print(f"Absolute value: {np.abs([-1, -2, 3, -4])}")
print(f"Sign: {np.sign([-1, -2, 0, 3, -4])}")
```

### 2. Trigonometric Functions

```python
angles = np.array([0, np.pi/4, np.pi/2, np.pi])

print(f"Angles (radians): {angles}")
print(f"Angles (degrees): {np.degrees(angles)}")
print(f"Sine: {np.sin(angles)}")
print(f"Cosine: {np.cos(angles)}")
print(f"Tangent: {np.tan(angles)}")

# Inverse trigonometric functions
values = np.array([0, 0.5, 1])
print(f"\nValues: {values}")
print(f"Arcsin: {np.arcsin(values)}")
print(f"Arccos: {np.arccos(values)}")
print(f"Arctan: {np.arctan(values)}")
```

### 3. Rounding Functions

```python
arr = np.array([1.234, 2.567, 3.891, -1.234])

print(f"Original: {arr}")
print(f"Round: {np.round(arr)}")
print(f"Floor: {np.floor(arr)}")
print(f"Ceiling: {np.ceil(arr)}")
print(f"Truncate: {np.trunc(arr)}")

# Round to specific decimal places
print(f"\nRound to 2 decimals: {np.round(arr, 2)}")
```

## Statistical Functions

### 1. Descriptive Statistics

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print(f"Array: {arr}")
print(f"Mean: {np.mean(arr)}")
print(f"Median: {np.median(arr)}")
print(f"Standard deviation: {np.std(arr)}")
print(f"Variance: {np.var(arr)}")
print(f"Minimum: {np.min(arr)}")
print(f"Maximum: {np.max(arr)}")
print(f"Sum: {np.sum(arr)}")
print(f"Product: {np.prod(arr)}")

# Percentiles
print(f"\n25th percentile: {np.percentile(arr, 25)}")
print(f"50th percentile: {np.percentile(arr, 50)}")
print(f"75th percentile: {np.percentile(arr, 75)}")
```

### 2. Multi-dimensional Statistics

```python
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(f"2D Array:\n{arr_2d}")

# Along different axes
print(f"\nMean along axis 0 (columns): {np.mean(arr_2d, axis=0)}")
print(f"Mean along axis 1 (rows): {np.mean(arr_2d, axis=1)}")
print(f"Overall mean: {np.mean(arr_2d)}")

print(f"\nSum along axis 0: {np.sum(arr_2d, axis=0)}")
print(f"Sum along axis 1: {np.sum(arr_2d, axis=1)}")
print(f"Overall sum: {np.sum(arr_2d)}")
```

### 3. Cumulative Operations

```python
arr = np.array([1, 2, 3, 4, 5])

print(f"Array: {arr}")
print(f"Cumulative sum: {np.cumsum(arr)}")
print(f"Cumulative product: {np.cumprod(arr)}")
print(f"Running mean: {np.cumsum(arr) / np.arange(1, len(arr) + 1)}")
```

## Shape Manipulation

### 1. Reshaping Arrays

```python
arr = np.arange(12)
print(f"Original array: {arr}")
print(f"Shape: {arr.shape}")

# Reshape to 2D
arr_2d = arr.reshape(3, 4)
print(f"\nReshaped to (3, 4):\n{arr_2d}")

# Reshape to 3D
arr_3d = arr.reshape(2, 2, 3)
print(f"\nReshaped to (2, 2, 3):\n{arr_3d}")

# Flatten
arr_flat = arr_2d.flatten()
print(f"\nFlattened: {arr_flat}")

# Ravel (similar to flatten)
arr_ravel = arr_2d.ravel()
print(f"Raveled: {arr_ravel}")
```

### 2. Transposition

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Original array:\n{arr}")
print(f"Shape: {arr.shape}")

# Transpose
arr_t = arr.T
print(f"\nTransposed:\n{arr_t}")
print(f"Shape: {arr_t.shape}")

# Transpose with transpose()
arr_t2 = np.transpose(arr)
print(f"\nTransposed (method):\n{arr_t2}")
```

### 3. Adding and Removing Dimensions

```python
arr = np.array([1, 2, 3, 4])
print(f"Original: {arr}, shape: {arr.shape}")

# Add dimension
arr_expanded = np.expand_dims(arr, axis=0)
print(f"Expanded axis 0: {arr_expanded}, shape: {arr_expanded.shape}")

arr_expanded2 = np.expand_dims(arr, axis=1)
print(f"Expanded axis 1: {arr_expanded2}, shape: {arr_expanded2.shape}")

# Remove dimension
arr_squeezed = np.squeeze(arr_expanded)
print(f"Squeezed: {arr_squeezed}, shape: {arr_squeezed.shape}")
```

## Broadcasting

Broadcasting allows NumPy to work with arrays of different shapes:

```python
# Broadcasting rules:
# 1. Arrays must have the same number of dimensions, or
# 2. One array must have fewer dimensions
# 3. Dimensions must be compatible (equal or one must be 1)

# Example 1: Array + scalar
arr = np.array([[1, 2, 3], [4, 5, 6]])
scalar = 10
result = arr + scalar
print(f"Array:\n{arr}")
print(f"Scalar: {scalar}")
print(f"Result:\n{result}")

# Example 2: Array + 1D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_1d = np.array([10, 20, 30])
result = arr_2d + arr_1d
print(f"\n2D Array:\n{arr_2d}")
print(f"1D Array: {arr_1d}")
print(f"Result:\n{result}")

# Example 3: Broadcasting with different shapes
arr_3x3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr_1x3 = np.array([[10, 20, 30]])
result = arr_3x3 + arr_1x3
print(f"\n3x3 Array:\n{arr_3x3}")
print(f"1x3 Array:\n{arr_1x3}")
print(f"Result:\n{result}")
```

## Best Practices

### 1. Memory Efficiency

```python
# Use appropriate data types
large_array = np.zeros(1000000, dtype=np.int32)  # 4MB
large_array_64 = np.zeros(1000000, dtype=np.int64)  # 8MB

# Avoid unnecessary copies
arr = np.array([1, 2, 3, 4, 5])
view = arr[1:4]  # Creates a view (no copy)
copy = arr[1:4].copy()  # Creates a copy

print(f"Original: {arr}")
print(f"View: {view}")
print(f"Copy: {copy}")

# Modify view affects original
view[0] = 100
print(f"\nAfter modifying view:")
print(f"Original: {arr}")
print(f"View: {view}")
print(f"Copy: {copy}")
```

### 2. Performance Optimization

```python
import time

# Slow: Python loops
def slow_sum(arr):
    result = 0
    for i in range(len(arr)):
        result += arr[i]
    return result

# Fast: NumPy operations
def fast_sum(arr):
    return np.sum(arr)

# Benchmark
large_arr = np.random.random(1000000)

start = time.time()
slow_result = slow_sum(large_arr)
slow_time = time.time() - start

start = time.time()
fast_result = fast_sum(large_arr)
fast_time = time.time() - start

print(f"Slow method: {slow_time:.6f} seconds")
print(f"Fast method: {fast_time:.6f} seconds")
print(f"Speedup: {slow_time/fast_time:.1f}x")
```

### 3. Code Organization

```python
# Good: Use descriptive variable names
data_matrix = np.random.random((100, 50))
mean_values = np.mean(data_matrix, axis=0)
normalized_data = (data_matrix - mean_values) / np.std(data_matrix, axis=0)

# Good: Use constants for magic numbers
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_ITERATIONS = 1000

# Good: Document complex operations
def normalize_features(X):
    """
    Normalize features using z-score normalization.
    
    Parameters:
    X : ndarray, shape (n_samples, n_features)
        Input data matrix
        
    Returns:
    ndarray : Normalized data matrix
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std
```

## Common Pitfalls

### 1. Mutable Views

```python
# Pitfall: Modifying views affects original
original = np.array([1, 2, 3, 4, 5])
view = original[1:4]
view[0] = 100
print(f"Original: {original}")  # [1, 100, 3, 4, 5]

# Solution: Use copy() when you need independent data
original = np.array([1, 2, 3, 4, 5])
copy = original[1:4].copy()
copy[0] = 100
print(f"Original: {original}")  # [1, 2, 3, 4, 5]
print(f"Copy: {copy}")  # [100, 3, 4]
```

### 2. Broadcasting Errors

```python
# Pitfall: Incompatible shapes
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3, 4])
# result = arr1 + arr2  # ValueError: operands could not be broadcast

# Solution: Ensure compatible shapes
arr1 = np.array([1, 2, 3])
arr2 = np.array([1, 2, 3])
result = arr1 + arr2  # Works fine
```

### 3. Data Type Issues

```python
# Pitfall: Integer division
arr = np.array([1, 2, 3, 4])
result = arr / 2
print(f"Result: {result}")  # [0.5, 1.0, 1.5, 2.0]

# If you want integer division
result_int = arr // 2
print(f"Integer division: {result_int}")  # [0, 1, 1, 2]
```

## Performance Tips

### 1. Use Vectorized Operations

```python
# Slow: Element-wise operations in loops
def slow_vectorize(arr1, arr2):
    result = np.zeros_like(arr1)
    for i in range(len(arr1)):
        result[i] = arr1[i] + arr2[i]
    return result

# Fast: NumPy vectorized operations
def fast_vectorize(arr1, arr2):
    return arr1 + arr2

# Benchmark
a = np.random.random(1000000)
b = np.random.random(1000000)

import time
start = time.time()
slow_result = slow_vectorize(a, b)
slow_time = time.time() - start

start = time.time()
fast_result = fast_vectorize(a, b)
fast_time = time.time() - start

print(f"Slow: {slow_time:.4f}s")
print(f"Fast: {fast_time:.4f}s")
print(f"Speedup: {slow_time/fast_time:.1f}x")
```

### 2. Memory Layout

```python
# C-contiguous (row-major) - default
arr_c = np.array([[1, 2, 3], [4, 5, 6]])
print(f"C-contiguous: {arr_c.flags.c_contiguous}")

# F-contiguous (column-major)
arr_f = np.asfortranarray(arr_c)
print(f"F-contiguous: {arr_f.flags.f_contiguous}")

# Performance depends on access pattern
# For row-wise access: C-contiguous is faster
# For column-wise access: F-contiguous is faster
```

### 3. Pre-allocation

```python
# Slow: Growing arrays
def slow_grow():
    result = []
    for i in range(10000):
        result.append(i)
    return np.array(result)

# Fast: Pre-allocate
def fast_preallocate():
    result = np.zeros(10000, dtype=int)
    for i in range(10000):
        result[i] = i
    return result

# Benchmark
import time
start = time.time()
slow_result = slow_grow()
slow_time = time.time() - start

start = time.time()
fast_result = fast_preallocate()
fast_time = time.time() - start

print(f"Slow: {slow_time:.4f}s")
print(f"Fast: {fast_time:.4f}s")
print(f"Speedup: {slow_time/fast_time:.1f}x")
```

## Summary

This guide covered the fundamental concepts of NumPy:

1. **Array Creation**: Various methods to create arrays
2. **Attributes and Properties**: Understanding array characteristics
3. **Data Types**: Choosing appropriate types for efficiency
4. **Basic Operations**: Arithmetic, comparison, and logical operations
5. **Indexing and Slicing**: Accessing and modifying array elements
6. **Mathematical Functions**: Vectorized mathematical operations
7. **Statistical Functions**: Descriptive statistics and analysis
8. **Shape Manipulation**: Reshaping and reorganizing arrays
9. **Broadcasting**: Working with arrays of different shapes
10. **Best Practices**: Writing efficient and maintainable code
11. **Common Pitfalls**: Avoiding typical mistakes
12. **Performance Tips**: Optimizing for speed and memory

### Next Steps

- Practice with the provided examples
- Explore the other NumPy guides in this toolkit
- Apply NumPy to real-world data science problems
- Learn about advanced indexing and array manipulation
- Study linear algebra operations with NumPy

### Additional Resources

- [NumPy Official Documentation](https://numpy.org/doc/)
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [NumPy Reference](https://numpy.org/doc/stable/reference/)
- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)

---

**Ready to explore more advanced NumPy features? Check out the other guides in this toolkit!** 