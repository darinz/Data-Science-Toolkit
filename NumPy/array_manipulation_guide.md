# NumPy Array Manipulation: A Comprehensive Guide

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

## Table of Contents

1. [Introduction to Array Manipulation](#introduction-to-array-manipulation)
2. [Array Reshaping](#array-reshaping)
3. [Array Transposition](#array-transposition)
4. [Concatenation and Stacking](#concatenation-and-stacking)
5. [Splitting and Chunking](#splitting-and-chunking)
6. [Broadcasting and Array Alignment](#broadcasting-and-array-alignment)
7. [Advanced Indexing and Manipulation](#advanced-indexing-and-manipulation)
8. [Memory Layout and Performance](#memory-layout-and-performance)
9. [Practical Applications](#practical-applications)
10. [Best Practices](#best-practices)
11. [Common Pitfalls](#common-pitfalls)
12. [Performance Tips](#performance-tips)

## Introduction to Array Manipulation

Array manipulation is a core skill in NumPy that allows you to transform, combine, and restructure arrays for various computational tasks.

### Key Manipulation Operations

- **Reshaping**: Change array dimensions and structure
- **Concatenation**: Combine arrays along different axes
- **Splitting**: Divide arrays into smaller parts
- **Transposition**: Rearrange array dimensions
- **Broadcasting**: Align arrays for operations
- **Advanced indexing**: Complex array access patterns

### Applications

✅ Data preprocessing and cleaning  
✅ Feature engineering for machine learning  
✅ Image and signal processing  
✅ Scientific computing and simulations  
✅ Data analysis and visualization  
✅ Performance optimization  

## Array Reshaping

Reshaping allows you to change the dimensions of an array while preserving the total number of elements.

### Basic Reshaping

```python
import numpy as np

# Create a 1D array
arr_1d = np.arange(12)
print(f"Original 1D array: {arr_1d}")
print(f"Shape: {arr_1d.shape}")

# Reshape to 2D
arr_2d = arr_1d.reshape(3, 4)
print(f"Reshaped to 2D (3x4):\n{arr_2d}")
print(f"Shape: {arr_2d.shape}")

# Reshape to 3D
arr_3d = arr_1d.reshape(2, 3, 2)
print(f"Reshaped to 3D (2x3x2):\n{arr_3d}")
print(f"Shape: {arr_3d.shape}")
```

### Reshape with -1

Using -1 to automatically calculate one dimension:

```python
# Reshape with automatic dimension calculation
arr_auto_1 = arr_1d.reshape(-1, 4)  # Calculate rows automatically
print(f"Reshape with -1, 4: {arr_auto_1.shape}")
print(f"Result:\n{arr_auto_1}")

arr_auto_2 = arr_1d.reshape(3, -1)  # Calculate columns automatically
print(f"Reshape with 3, -1: {arr_auto_2.shape}")
print(f"Result:\n{arr_auto_2}")

arr_auto_3 = arr_1d.reshape(-1)  # Flatten to 1D
print(f"Reshape with -1: {arr_auto_3.shape}")
print(f"Result: {arr_auto_3}")
```

### Reshape vs Resize

```python
# Reshape (creates new array)
arr_original = np.array([1, 2, 3, 4])
arr_reshaped = arr_original.reshape(2, 2)
print(f"Original: {arr_original}")
print(f"Reshaped:\n{arr_reshaped}")
print(f"Original unchanged: {arr_original}")

# Resize (modifies original array)
arr_to_resize = np.array([1, 2, 3, 4])
arr_to_resize.resize(2, 2)
print(f"After resize:\n{arr_to_resize}")
```

### Flattening Arrays

```python
# Create a 2D array
arr_2d_flat = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2D array:\n{arr_2d_flat}")

# Flatten methods
flattened_ravel = arr_2d_flat.ravel()  # Creates view if possible
flattened_flatten = arr_2d_flat.flatten()  # Always creates copy
flattened_reshape = arr_2d_flat.reshape(-1)

print(f"Ravel: {flattened_ravel}")
print(f"Flatten: {flattened_flatten}")
print(f"Reshape: {flattened_reshape}")

# Check if they're views or copies
print(f"Ravel is view: {flattened_ravel.base is arr_2d_flat}")
print(f"Flatten is view: {flattened_flatten.base is arr_2d_flat}")
```

## Array Transposition

Transposition rearranges the dimensions of an array, which is particularly useful for matrix operations and data alignment.

### Basic Transposition

```python
# Create a 2D array
arr_transpose = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Original array:\n{arr_transpose}")
print(f"Shape: {arr_transpose.shape}")

# Transpose
arr_T = arr_transpose.T
print(f"Transposed array:\n{arr_T}")
print(f"Shape: {arr_T.shape}")

# Using transpose function
arr_transposed = np.transpose(arr_transpose)
print(f"Using transpose():\n{arr_transposed}")
```

### Multi-dimensional Transposition

```python
# Create a 3D array
arr_3d_transpose = np.arange(24).reshape(2, 3, 4)
print(f"3D array shape: {arr_3d_transpose.shape}")
print(f"3D array:\n{arr_3d_transpose}")

# Transpose with custom axis order
arr_transposed_3d = np.transpose(arr_3d_transpose, (1, 0, 2))
print(f"Transposed (1, 0, 2) shape: {arr_transposed_3d.shape}")
print(f"Transposed array:\n{arr_transposed_3d}")
```

### Matrix Operations with Transposition

```python
# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")

# Matrix multiplication properties
AB = A @ B
AB_T = (A @ B).T
B_T_A_T = B.T @ A.T

print(f"(A @ B):\n{AB}")
print(f"(A @ B)^T:\n{AB_T}")
print(f"B^T @ A^T:\n{B_T_A_T}")
print(f"Transpose property holds: {np.array_equal(AB_T, B_T_A_T)}")
```

## Concatenation and Stacking

Concatenation combines arrays along specified axes, while stacking creates new dimensions.

### Basic Concatenation

```python
# 1D arrays
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

# Concatenate along axis 0 (default)
concatenated = np.concatenate([arr1, arr2])
print(f"Concatenated: {concatenated}")

# Using np.append
appended = np.append(arr1, arr2)
print(f"Appended: {appended}")
```

### 2D Array Concatenation

```python
# 2D arrays
arr_2d_1 = np.array([[1, 2], [3, 4]])
arr_2d_2 = np.array([[5, 6], [7, 8]])

# Concatenate along axis 0 (rows)
concatenated_rows = np.concatenate([arr_2d_1, arr_2d_2], axis=0)
print(f"Concatenated along rows:\n{concatenated_rows}")

# Concatenate along axis 1 (columns)
concatenated_cols = np.concatenate([arr_2d_1, arr_2d_2], axis=1)
print(f"Concatenated along columns:\n{concatenated_cols}")
```

### Stacking Functions

```python
# Stack along new axis
stacked = np.stack([arr1, arr2])
print(f"Stacked:\n{stacked}")
print(f"Shape: {stacked.shape}")

# vstack (vertical stack)
vstacked = np.vstack([arr_2d_1, arr_2d_2])
print(f"Vstacked:\n{vstacked}")

# hstack (horizontal stack)
hstacked = np.hstack([arr_2d_1, arr_2d_2])
print(f"Hstacked:\n{hstacked}")

# dstack (depth stack)
dstacked = np.dstack([arr_2d_1, arr_2d_2])
print(f"Dstacked:\n{dstacked}")
```

### Column and Row Stacking

```python
# Column stack
col_stacked = np.column_stack([arr1, arr2])
print(f"Column stacked:\n{col_stacked}")

# Row stack
row_stacked = np.row_stack([arr1, arr2])
print(f"Row stacked:\n{row_stacked}")
```

## Splitting and Chunking

Splitting divides arrays into smaller parts along specified axes.

### Basic Splitting

```python
# Create array to split
arr_to_split = np.arange(12).reshape(3, 4)
print(f"Array to split:\n{arr_to_split}")

# Split into equal parts
split_parts = np.split(arr_to_split, 3, axis=0)
print(f"Split into 3 parts:")
for i, part in enumerate(split_parts):
    print(f"Part {i}:\n{part}")
```

### Array Split

```python
# Split at specific indices
split_at_indices = np.array_split(arr_to_split, [1, 2], axis=1)
print(f"Split at indices [1, 2]:")
for i, part in enumerate(split_at_indices):
    print(f"Part {i}:\n{part}")
```

### Vertical and Horizontal Splitting

```python
# Vertical split
vsplit_parts = np.vsplit(arr_to_split, 3)
print(f"Vertical split:")
for i, part in enumerate(vsplit_parts):
    print(f"Part {i}:\n{part}")

# Horizontal split
hsplit_parts = np.hsplit(arr_to_split, 2)
print(f"Horizontal split:")
for i, part in enumerate(hsplit_parts):
    print(f"Part {i}:\n{part}")
```

### Chunking Arrays

```python
def chunk_array(arr, chunk_size):
    """Split array into chunks of specified size."""
    return [arr[i:i+chunk_size] for i in range(0, len(arr), chunk_size)]

# Example with 1D array
arr_1d_chunk = np.arange(10)
chunks = chunk_array(arr_1d_chunk, 3)
print(f"Original: {arr_1d_chunk}")
print(f"Chunks of size 3: {chunks}")
```

## Broadcasting and Array Alignment

Broadcasting allows operations between arrays of different shapes by automatically aligning their dimensions.

### Broadcasting Basics

```python
# Broadcasting with scalar
arr = np.array([[1, 2, 3], [4, 5, 6]])
result = arr + 10
print(f"Original:\n{arr}")
print(f"After broadcasting + 10:\n{result}")

# Broadcasting with 1D array
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_1d = np.array([10, 20, 30])
result = arr_2d + arr_1d
print(f"2D array:\n{arr_2d}")
print(f"1D array: {arr_1d}")
print(f"Broadcasted result:\n{result}")
```

### Broadcasting Rules

```python
# Rule 1: Arrays with different dimensions
arr_3d = np.arange(24).reshape(2, 3, 4)
arr_1d_broadcast = np.array([1, 2, 3, 4])
result = arr_3d + arr_1d_broadcast
print(f"3D array shape: {arr_3d.shape}")
print(f"1D array shape: {arr_1d_broadcast.shape}")
print(f"Result shape: {result.shape}")

# Rule 2: Arrays with compatible shapes
arr_a = np.array([[1, 2, 3], [4, 5, 6]])
arr_b = np.array([[10], [20]])
result = arr_a + arr_b
print(f"Array A shape: {arr_a.shape}")
print(f"Array B shape: {arr_b.shape}")
print(f"Result:\n{result}")
```

### Broadcasting for Outer Operations

```python
# Outer product using broadcasting
a = np.array([1, 2, 3])
b = np.array([4, 5, 6, 7])

# Reshape for broadcasting
a_reshaped = a.reshape(-1, 1)  # Column vector
b_reshaped = b.reshape(1, -1)  # Row vector

outer_product = a_reshaped * b_reshaped
print(f"Outer product:\n{outer_product}")
```

## Advanced Indexing and Manipulation

Advanced indexing provides powerful ways to access and manipulate array elements.

### Boolean Indexing

```python
# Create array
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Boolean indexing
mask = arr > 5
filtered = arr[mask]
print(f"Original: {arr}")
print(f"Mask: {mask}")
print(f"Filtered: {filtered}")

# Multiple conditions
mask_complex = (arr > 3) & (arr < 8)
filtered_complex = arr[mask_complex]
print(f"Complex mask: {mask_complex}")
print(f"Filtered: {filtered_complex}")
```

### Integer Array Indexing

```python
# Integer array indexing
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
indices = np.array([0, 2, 1])
selected = arr_2d[indices]
print(f"Original:\n{arr_2d}")
print(f"Indices: {indices}")
print(f"Selected rows:\n{selected}")
```

### Advanced Indexing Combinations

```python
# Combining different indexing methods
arr_3d = np.arange(24).reshape(2, 3, 4)
print(f"3D array:\n{arr_3d}")

# Mixed indexing
result = arr_3d[0, :, [1, 3]]  # First slice, all rows, columns 1 and 3
print(f"Mixed indexing result:\n{result}")
```

## Memory Layout and Performance

Understanding memory layout is crucial for performance optimization.

### Memory Layout Basics

```python
# Check memory layout
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"Array:\n{arr}")
print(f"C-contiguous: {arr.flags.c_contiguous}")
print(f"F-contiguous: {arr.flags.f_contiguous}")
print(f"Memory layout: {arr.flags}")
```

### Contiguous Arrays

```python
# Create contiguous array
arr_contiguous = np.array([[1, 2, 3], [4, 5, 6]], order='C')
print(f"C-contiguous:\n{arr_contiguous}")
print(f"Flags: {arr_contiguous.flags}")

# Create Fortran-contiguous array
arr_f_contiguous = np.array([[1, 2, 3], [4, 5, 6]], order='F')
print(f"F-contiguous:\n{arr_f_contiguous}")
print(f"Flags: {arr_f_contiguous.flags}")
```

### Performance Considerations

```python
import time

# Performance comparison
size = 1000
arr_large = np.random.random((size, size))

# Row-wise operations (faster for C-contiguous)
start_time = time.time()
row_sum = np.sum(arr_large, axis=1)
row_time = time.time() - start_time

# Column-wise operations (slower for C-contiguous)
start_time = time.time()
col_sum = np.sum(arr_large, axis=0)
col_time = time.time() - start_time

print(f"Row-wise sum time: {row_time:.6f}")
print(f"Column-wise sum time: {col_time:.6f}")
```

## Practical Applications

### Data Preprocessing

```python
# Reshape data for machine learning
data = np.random.random(1000)
features = data.reshape(-1, 10)  # 100 samples, 10 features
print(f"Data shape: {data.shape}")
print(f"Features shape: {features.shape}")

# Normalize features
features_normalized = (features - features.mean(axis=0)) / features.std(axis=0)
print(f"Normalized features mean: {features_normalized.mean(axis=0)}")
print(f"Normalized features std: {features_normalized.std(axis=0)}")
```

### Image Processing

```python
# Simulate image data
image = np.random.random((100, 100, 3))  # RGB image
print(f"Image shape: {image.shape}")

# Convert to grayscale
grayscale = np.mean(image, axis=2)
print(f"Grayscale shape: {grayscale.shape}")

# Reshape for processing
image_flat = image.reshape(-1, 3)  # Flatten to 2D
print(f"Flattened shape: {image_flat.shape}")
```

### Time Series Data

```python
# Create time series data
time_series = np.random.random(1000)
window_size = 10

# Create sliding windows
windows = []
for i in range(len(time_series) - window_size + 1):
    windows.append(time_series[i:i+window_size])

windows_array = np.array(windows)
print(f"Time series length: {len(time_series)}")
print(f"Windows shape: {windows_array.shape}")
```

## Best Practices

### 1. Use Appropriate Data Types

```python
# Choose appropriate data types
small_ints = np.array([1, 2, 3], dtype=np.int8)  # Save memory
large_floats = np.array([1.1, 2.2, 3.3], dtype=np.float64)  # Precision
```

### 2. Prefer Vectorized Operations

```python
# Good: Vectorized operation
arr = np.array([1, 2, 3, 4, 5])
result = arr * 2 + 1

# Avoid: Loops for simple operations
result_loop = np.array([x * 2 + 1 for x in arr])
```

### 3. Use Views When Possible

```python
# Use views to save memory
arr = np.array([1, 2, 3, 4, 5])
view = arr[1:4]  # Creates view, not copy
copy = arr[1:4].copy()  # Creates copy
```

### 4. Understand Broadcasting

```python
# Leverage broadcasting for efficiency
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
scalar = 10
result = arr_2d + scalar  # Broadcasting
```

## Common Pitfalls

### 1. Modifying Views

```python
# Be careful with views
arr = np.array([1, 2, 3, 4, 5])
view = arr[1:4]
view[0] = 100  # Modifies original array
print(f"Original: {arr}")  # [1, 100, 3, 4, 5]
```

### 2. Shape Mismatches

```python
# Check shapes before operations
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5])

try:
    result = arr1 + arr2  # Will raise error
except ValueError as e:
    print(f"Error: {e}")
```

### 3. Memory Layout Issues

```python
# Transpose can affect performance
arr = np.array([[1, 2, 3], [4, 5, 6]])
transposed = arr.T

# Check if transposed is a view
print(f"Transposed is view: {transposed.base is arr}")
```

## Performance Tips

### 1. Use Contiguous Arrays

```python
# Ensure arrays are contiguous for better performance
arr = np.array([[1, 2, 3], [4, 5, 6]])
if not arr.flags.c_contiguous:
    arr = np.ascontiguousarray(arr)
```

### 2. Avoid Unnecessary Copies

```python
# Use in-place operations when possible
arr = np.array([1, 2, 3, 4, 5])
arr *= 2  # In-place multiplication
# Instead of: arr = arr * 2
```

### 3. Use Appropriate Functions

```python
# Use specialized functions for better performance
arr = np.array([1, 2, 3, 4, 5])

# For concatenation
result = np.concatenate([arr, arr])  # Better than np.append

# For stacking
result = np.stack([arr, arr])  # Better than manual reshaping
```

### 4. Profile Your Code

```python
import time

def time_operation(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

# Example usage
arr = np.random.random(1000000)
result, elapsed = time_operation(np.sort, arr)
print(f"Sort time: {elapsed:.6f} seconds")
```

---

This guide covers the essential array manipulation techniques in NumPy. Practice these concepts with your own data to become proficient in array manipulation for data science and scientific computing applications. 