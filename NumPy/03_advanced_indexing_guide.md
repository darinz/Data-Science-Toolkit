# Advanced NumPy Indexing: A Comprehensive Guide

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

## Table of Contents

1. [Introduction to Advanced Indexing](#introduction-to-advanced-indexing)
2. [Boolean Indexing](#boolean-indexing)
3. [Fancy Indexing](#fancy-indexing)
4. [Combined Indexing](#combined-indexing)
5. [Structured Arrays](#structured-arrays)
6. [Masked Arrays](#masked-arrays)
7. [Advanced Slicing](#advanced-slicing)
8. [Performance Considerations](#performance-considerations)
9. [Real-World Applications](#real-world-applications)
10. [Best Practices](#best-practices)
11. [Common Pitfalls](#common-pitfalls)

## Introduction to Advanced Indexing

NumPy provides powerful indexing capabilities beyond basic integer indexing. Advanced indexing allows you to:

- **Select elements conditionally** using boolean masks
- **Access elements using arrays of indices** (fancy indexing)
- **Combine different indexing methods** for complex selections
- **Work with structured data** efficiently
- **Handle missing or invalid data** with masked arrays

### Indexing Methods Overview

```python
import numpy as np

arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"Original array:\n{arr}")

# Basic indexing
print(f"\nBasic indexing - arr[0, 1]: {arr[0, 1]}")

# Boolean indexing
mask = arr > 5
print(f"\nBoolean mask:\n{mask}")
print(f"Boolean indexing - arr[mask]: {arr[mask]}")

# Fancy indexing
indices = [0, 2]
print(f"\nFancy indexing - arr[indices]:\n{arr[indices]}")

# Combined indexing
print(f"\nCombined - arr[mask][:3]: {arr[mask][:3]}")
```

## Boolean Indexing

Boolean indexing uses boolean arrays to select elements conditionally.

### 1. Basic Boolean Indexing

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Simple conditions
mask = arr > 5
print(f"Array: {arr}")
print(f"Mask (arr > 5): {mask}")
print(f"Selected elements: {arr[mask]}")

# Multiple conditions
mask_even = arr % 2 == 0
mask_greater_than_3 = arr > 3
print(f"\nEven numbers: {arr[mask_even]}")
print(f"Numbers > 3: {arr[mask_greater_than_3]}")

# Combining conditions
mask_combined = (arr > 3) & (arr % 2 == 0)
print(f"Even numbers > 3: {arr[mask_combined]}")
```

### 2. Multi-dimensional Boolean Indexing

```python
arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"2D Array:\n{arr_2d}")

# Boolean indexing on 2D arrays
mask_2d = arr_2d > 5
print(f"\nBoolean mask:\n{mask_2d}")
print(f"Selected elements: {arr_2d[mask_2d]}")

# Row-wise conditions
mask_rows = np.any(arr_2d > 5, axis=1)
print(f"\nRows with any element > 5: {mask_rows}")
print(f"Selected rows:\n{arr_2d[mask_rows]}")

# Column-wise conditions
mask_cols = np.all(arr_2d > 5, axis=0)
print(f"\nColumns with all elements > 5: {mask_cols}")
print(f"Selected columns:\n{arr_2d[:, mask_cols]}")
```

### 3. Complex Boolean Conditions

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Using logical functions
mask1 = np.logical_and(arr > 3, arr < 8)
mask2 = np.logical_or(arr < 3, arr > 8)
mask3 = np.logical_not(arr % 2 == 0)

print(f"Array: {arr}")
print(f"3 < arr < 8: {arr[mask1]}")
print(f"arr < 3 OR arr > 8: {arr[mask2]}")
print(f"Odd numbers: {arr[mask3]}")

# Using where function
indices = np.where(arr > 5)
print(f"\nIndices where arr > 5: {indices}")
print(f"Values at those indices: {arr[indices]}")

# Using nonzero
nonzero_indices = (arr > 5).nonzero()
print(f"Nonzero indices: {nonzero_indices}")
print(f"Values: {arr[nonzero_indices]}")
```

### 4. Boolean Indexing with Functions

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Using mathematical functions
mask_sqrt = np.sqrt(arr) > 2
mask_log = np.log(arr) > 1
mask_sin = np.sin(arr) > 0

print(f"Array: {arr}")
print(f"sqrt(arr) > 2: {arr[mask_sqrt]}")
print(f"log(arr) > 1: {arr[mask_log]}")
print(f"sin(arr) > 0: {arr[mask_sin]}")

# Using statistical functions
mean_val = np.mean(arr)
std_val = np.std(arr)
mask_outliers = np.abs(arr - mean_val) > 2 * std_val

print(f"\nMean: {mean_val:.2f}")
print(f"Std: {std_val:.2f}")
print(f"Outliers (> 2 std): {arr[mask_outliers]}")
```

## Fancy Indexing

Fancy indexing uses arrays of indices to select elements.

### 1. Basic Fancy Indexing

```python
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Integer array indexing
indices = [0, 2, 4, 6, 8]
selected = arr[indices]
print(f"Array: {arr}")
print(f"Indices: {indices}")
print(f"Selected: {selected}")

# Negative indices
negative_indices = [-1, -3, -5]
selected_negative = arr[negative_indices]
print(f"\nNegative indices: {negative_indices}")
print(f"Selected: {selected_negative}")

# Repeated indices
repeated_indices = [0, 0, 1, 1, 2, 2]
selected_repeated = arr[repeated_indices]
print(f"\nRepeated indices: {repeated_indices}")
print(f"Selected: {selected_repeated}")
```

### 2. Multi-dimensional Fancy Indexing

```python
arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"2D Array:\n{arr_2d}")

# Row indexing
row_indices = [0, 2]
selected_rows = arr_2d[row_indices]
print(f"\nRow indices: {row_indices}")
print(f"Selected rows:\n{selected_rows}")

# Column indexing
col_indices = [1, 3]
selected_cols = arr_2d[:, col_indices]
print(f"\nColumn indices: {col_indices}")
print(f"Selected columns:\n{selected_cols}")

# Both row and column indexing
row_idx = [0, 2]
col_idx = [1, 3]
selected_both = arr_2d[row_idx, col_idx]
print(f"\nRow indices: {row_idx}")
print(f"Column indices: {col_idx}")
print(f"Selected elements: {selected_both}")
```

### 3. Advanced Fancy Indexing

```python
arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Using meshgrid for coordinate indexing
row_coords = np.array([0, 1, 2])
col_coords = np.array([1, 2, 3])
selected_coords = arr_2d[row_coords, col_coords]
print(f"Array:\n{arr_2d}")
print(f"Row coordinates: {row_coords}")
print(f"Column coordinates: {col_coords}")
print(f"Selected elements: {selected_coords}")

# Using boolean arrays for fancy indexing
bool_mask = np.array([[True, False, True, False],
                      [False, True, False, True],
                      [True, False, True, False]])
selected_bool = arr_2d[bool_mask]
print(f"\nBoolean mask:\n{bool_mask}")
print(f"Selected elements: {selected_bool}")
```

## Combined Indexing

Combining different indexing methods for complex selections.

### 1. Boolean + Fancy Indexing

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Boolean indexing followed by fancy indexing
mask = arr > 5
filtered = arr[mask]
indices = [0, 2, 3]
final_selection = filtered[indices]

print(f"Original: {arr}")
print(f"After boolean filtering: {filtered}")
print(f"Fancy indices: {indices}")
print(f"Final selection: {final_selection}")

# Alternative: using where
where_indices = np.where(arr > 5)[0]
fancy_indices = where_indices[[0, 2, 3]]
result = arr[fancy_indices]
print(f"\nUsing where: {result}")
```

### 2. Multi-dimensional Combined Indexing

```python
arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"Original array:\n{arr_2d}")

# Boolean indexing on rows, fancy indexing on columns
row_mask = np.any(arr_2d > 5, axis=1)
col_indices = [1, 3]

selected_rows = arr_2d[row_mask]
final_result = selected_rows[:, col_indices]

print(f"Row mask: {row_mask}")
print(f"Selected rows:\n{selected_rows}")
print(f"Column indices: {col_indices}")
print(f"Final result:\n{final_result}")
```

### 3. Complex Combined Operations

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Multiple conditions with fancy indexing
condition1 = arr > 3
condition2 = arr % 2 == 0
combined_mask = condition1 & condition2

filtered = arr[combined_mask]
indices = [0, 1]  # Select first two elements
result = filtered[indices]

print(f"Original: {arr}")
print(f"Condition 1 (arr > 3): {condition1}")
print(f"Condition 2 (arr % 2 == 0): {condition2}")
print(f"Combined mask: {combined_mask}")
print(f"Filtered: {filtered}")
print(f"Final result: {result}")
```

## Structured Arrays

Structured arrays allow you to work with heterogeneous data efficiently.

### 1. Creating Structured Arrays

```python
# Define data types
dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4')]

# Create structured array
data = [('Alice', 25, 1.75), ('Bob', 30, 1.80), ('Charlie', 35, 1.70)]
structured_arr = np.array(data, dtype=dtype)

print(f"Structured array:\n{structured_arr}")
print(f"Data type: {structured_arr.dtype}")
print(f"Field names: {structured_arr.dtype.names}")

# Accessing fields
print(f"\nNames: {structured_arr['name']}")
print(f"Ages: {structured_arr['age']}")
print(f"Heights: {structured_arr['height']}")
```

### 2. Indexing Structured Arrays

```python
# Boolean indexing on fields
age_mask = structured_arr['age'] > 28
print(f"Age mask: {age_mask}")
print(f"People older than 28:\n{structured_arr[age_mask]}")

# Fancy indexing
indices = [0, 2]
print(f"\nSelected people:\n{structured_arr[indices]}")

# Combined indexing
height_mask = structured_arr['height'] > 1.75
selected = structured_arr[height_mask]
names_only = selected['name']
print(f"\nTall people: {names_only}")
```

### 3. Complex Structured Arrays

```python
# Multi-dimensional structured array
dtype = [('id', 'i4'), ('scores', 'f4', (3,))]
data = [(1, [85.5, 90.2, 78.9]), 
        (2, [92.1, 88.7, 95.3]), 
        (3, [76.8, 82.4, 89.1])]
structured_2d = np.array(data, dtype=dtype)

print(f"Structured 2D array:\n{structured_2d}")
print(f"Scores for student 1: {structured_2d[0]['scores']}")

# Boolean indexing on computed values
mean_scores = np.mean(structured_2d['scores'], axis=1)
high_performers = structured_2d[mean_scores > 85]
print(f"\nHigh performers (mean > 85):\n{high_performers}")
```

## Masked Arrays

Masked arrays handle missing or invalid data efficiently.

### 1. Creating Masked Arrays

```python
import numpy.ma as ma

# Create masked array with invalid values
data = [1, 2, -999, 4, 5, -999, 7, 8]
masked_arr = ma.masked_equal(data, -999)

print(f"Original data: {data}")
print(f"Masked array: {masked_arr}")
print(f"Mask: {masked_arr.mask}")
print(f"Valid data: {masked_arr.compressed()}")

# Create masked array with conditions
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
masked_condition = ma.masked_where(arr > 5, arr)
print(f"\nMasked where arr > 5: {masked_condition}")
```

### 2. Operations on Masked Arrays

```python
# Mathematical operations
arr1 = ma.masked_equal([1, 2, -999, 4], -999)
arr2 = ma.masked_equal([5, -999, 7, 8], -999)

sum_result = arr1 + arr2
print(f"Array 1: {arr1}")
print(f"Array 2: {arr2}")
print(f"Sum: {sum_result}")

# Statistical operations
mean_val = ma.mean(arr1)
std_val = ma.std(arr1)
print(f"\nMean: {mean_val}")
print(f"Std: {std_val}")
```

### 3. Advanced Masked Array Operations

```python
# Combining masks
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mask1 = ma.masked_less(arr, 3)
mask2 = ma.masked_greater(arr, 7)
combined_mask = ma.masked_or(mask1.mask, mask2.mask)
result = ma.array(arr, mask=combined_mask)

print(f"Original: {arr}")
print(f"Masked < 3: {mask1}")
print(f"Masked > 7: {mask2}")
print(f"Combined mask: {result}")
```

## Advanced Slicing

Advanced slicing techniques for complex array operations.

### 1. Strided Slicing

```python
arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Basic strided slicing
print(f"Original: {arr}")
print(f"Every 2nd element: {arr[::2]}")
print(f"Every 3rd element: {arr[::3]}")
print(f"Reverse: {arr[::-1]}")
print(f"Reverse every 2nd: {arr[::-2]}")

# 2D strided slicing
arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"\n2D Array:\n{arr_2d}")
print(f"Every other row:\n{arr_2d[::2]}")
print(f"Every other column:\n{arr_2d[:, ::2]}")
```

### 2. Ellipsis and Newaxis

```python
arr = np.array([1, 2, 3, 4, 5])

# Using ellipsis
arr_2d = arr.reshape(1, -1)
print(f"Original: {arr}")
print(f"Reshaped:\n{arr_2d}")
print(f"Using ellipsis: {arr_2d[..., 0]}")

# Using newaxis
arr_newaxis = arr[np.newaxis, :]
print(f"\nWith newaxis:\n{arr_newaxis}")
print(f"Shape: {arr_newaxis.shape}")

# Combining ellipsis and newaxis
arr_3d = arr[np.newaxis, ..., np.newaxis]
print(f"\n3D with newaxis:\n{arr_3d}")
print(f"Shape: {arr_3d.shape}")
```

### 3. Advanced Indexing with Slicing

```python
arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

# Mixed indexing
print(f"Array:\n{arr_2d}")
print(f"Rows 0,2, columns 1:3:\n{arr_2d[[0, 2], 1:3]}")
print(f"All rows, columns 0,2:\n{arr_2d[:, [0, 2]]}")

# Boolean indexing with slicing
mask = arr_2d > 5
print(f"\nBoolean mask:\n{mask}")
print(f"Selected elements: {arr_2d[mask]}")
print(f"First 3 selected: {arr_2d[mask][:3]}")
```

## Performance Considerations

### 1. Memory Layout and Performance

```python
import time

# C-contiguous vs F-contiguous
arr_c = np.array([[1, 2, 3], [4, 5, 6]])
arr_f = np.asfortranarray(arr_c)

# Row-wise access (C-contiguous is faster)
start = time.time()
for i in range(10000):
    _ = arr_c[0, :]
c_time = time.time() - start

start = time.time()
for i in range(10000):
    _ = arr_f[0, :]
f_time = time.time() - start

print(f"C-contiguous row access: {c_time:.6f}s")
print(f"F-contiguous row access: {f_time:.6f}s")

# Column-wise access (F-contiguous is faster)
start = time.time()
for i in range(10000):
    _ = arr_c[:, 0]
c_col_time = time.time() - start

start = time.time()
for i in range(10000):
    _ = arr_f[:, 0]
f_col_time = time.time() - start

print(f"\nC-contiguous column access: {c_col_time:.6f}s")
print(f"F-contiguous column access: {f_col_time:.6f}s")
```

### 2. Indexing Performance Comparison

```python
arr = np.random.random(1000000)

# Boolean indexing vs fancy indexing
mask = arr > 0.5
indices = np.where(mask)[0]

# Boolean indexing
start = time.time()
result_bool = arr[mask]
bool_time = time.time() - start

# Fancy indexing
start = time.time()
result_fancy = arr[indices]
fancy_time = time.time() - start

print(f"Boolean indexing: {bool_time:.6f}s")
print(f"Fancy indexing: {fancy_time:.6f}s")
print(f"Speedup: {fancy_time/bool_time:.2f}x")
```

### 3. Memory Efficiency

```python
# Views vs copies
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# View (no memory copy)
view = arr[::2]
print(f"View is view: {view.base is arr}")

# Copy (memory copy)
copy = arr[::2].copy()
print(f"Copy is view: {copy.base is arr}")

# Memory usage
print(f"Original array size: {arr.nbytes} bytes")
print(f"View size: {view.nbytes} bytes")
print(f"Copy size: {copy.nbytes} bytes")
```

## Real-World Applications

### 1. Data Filtering and Selection

```python
# Simulate dataset
np.random.seed(42)
ages = np.random.randint(18, 80, 1000)
salaries = np.random.normal(50000, 15000, 1000)
departments = np.random.choice(['IT', 'HR', 'Sales', 'Marketing'], 1000)

# Filter high earners in IT
it_mask = departments == 'IT'
high_salary_mask = salaries > 60000
it_high_earners = salaries[it_mask & high_salary_mask]

print(f"Total employees: {len(ages)}")
print(f"IT employees: {np.sum(it_mask)}")
print(f"High earners in IT: {len(it_high_earners)}")
print(f"Average salary of IT high earners: ${np.mean(it_high_earners):.2f}")
```

### 2. Image Processing

```python
# Simulate image data
image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

# Extract red channel
red_channel = image[:, :, 0]
print(f"Image shape: {image.shape}")
print(f"Red channel shape: {red_channel.shape}")

# Threshold red channel
bright_red = red_channel > 200
print(f"Bright red pixels: {np.sum(bright_red)}")

# Extract bright regions
bright_regions = image[bright_red]
print(f"Bright region pixels: {len(bright_regions)}")
```

### 3. Time Series Analysis

```python
# Simulate time series data
dates = np.arange('2023-01-01', '2023-12-31', dtype='datetime64[D]')
values = np.random.normal(100, 10, len(dates))

# Filter by date range
start_date = np.datetime64('2023-06-01')
end_date = np.datetime64('2023-08-31')
summer_mask = (dates >= start_date) & (dates <= end_date)
summer_values = values[summer_mask]

print(f"Total days: {len(dates)}")
print(f"Summer days: {len(summer_values)}")
print(f"Summer average: {np.mean(summer_values):.2f}")
```

## Best Practices

### 1. Efficient Indexing Patterns

```python
# Good: Use boolean indexing for conditions
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
good_selection = arr[arr > 5]

# Avoid: Loops for simple conditions
bad_selection = []
for i in range(len(arr)):
    if arr[i] > 5:
        bad_selection.append(arr[i])
bad_selection = np.array(bad_selection)

print(f"Good method: {good_selection}")
print(f"Bad method: {bad_selection}")
```

### 2. Memory Management

```python
# Good: Use views when possible
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
view = arr[::2]  # Creates a view

# Good: Use copy when you need independent data
copy = arr[::2].copy()  # Creates a copy

# Check if it's a view
print(f"View is view: {view.base is arr}")
print(f"Copy is view: {copy.base is arr}")
```

### 3. Readable Code

```python
# Good: Use descriptive variable names
data = np.random.random(1000)
outlier_threshold = 2.0
mean_val = np.mean(data)
std_val = np.std(data)

outlier_mask = np.abs(data - mean_val) > outlier_threshold * std_val
outliers = data[outlier_mask]

print(f"Outliers found: {len(outliers)}")
print(f"Outlier percentage: {len(outliers)/len(data)*100:.1f}%")
```

## Common Pitfalls

### 1. Mutable Views

```python
# Pitfall: Modifying views affects original
original = np.array([1, 2, 3, 4, 5])
view = original[1:4]
view[0] = 100
print(f"Original: {original}")  # [1, 100, 3, 4, 5]

# Solution: Use copy when you need independent data
original = np.array([1, 2, 3, 4, 5])
copy = original[1:4].copy()
copy[0] = 100
print(f"Original: {original}")  # [1, 2, 3, 4, 5]
print(f"Copy: {copy}")  # [100, 3, 4]
```

### 2. Broadcasting Errors

```python
# Pitfall: Incompatible shapes for boolean indexing
arr = np.array([[1, 2, 3], [4, 5, 6]])
mask = np.array([True, False, True])  # Wrong shape
# result = arr[mask]  # ValueError

# Solution: Use correct shape
mask_correct = np.array([[True, False, True], [False, True, False]])
result = arr[mask_correct]
print(f"Correct result: {result}")
```

### 3. Index Out of Bounds

```python
# Pitfall: Index out of bounds
arr = np.array([1, 2, 3, 4, 5])
# value = arr[10]  # IndexError

# Solution: Check bounds or use safe indexing
if 10 < len(arr):
    value = arr[10]
else:
    value = None

# Or use try-except
try:
    value = arr[10]
except IndexError:
    value = None
```

## Summary

This guide covered advanced NumPy indexing techniques:

1. **Boolean Indexing**: Conditional element selection
2. **Fancy Indexing**: Array-based index selection
3. **Combined Indexing**: Mixing different indexing methods
4. **Structured Arrays**: Working with heterogeneous data
5. **Masked Arrays**: Handling missing/invalid data
6. **Advanced Slicing**: Complex array operations
7. **Performance Considerations**: Optimizing for speed and memory
8. **Real-World Applications**: Practical use cases
9. **Best Practices**: Writing efficient and maintainable code
10. **Common Pitfalls**: Avoiding typical mistakes

### Key Takeaways

- **Boolean indexing** is powerful for conditional selection
- **Fancy indexing** provides flexible element access
- **Views vs copies** affect memory usage and performance
- **Structured arrays** handle heterogeneous data efficiently
- **Masked arrays** manage missing data gracefully
- **Performance** depends on memory layout and access patterns

### Next Steps

- Practice with the provided examples
- Explore real-world datasets
- Learn about NumPy's linear algebra capabilities
- Study array manipulation and broadcasting
- Master performance optimization techniques

### Additional Resources

- [NumPy Advanced Indexing](https://numpy.org/doc/stable/user/basics.indexing.html)
- [NumPy Structured Arrays](https://numpy.org/doc/stable/user/basics.rec.html)
- [NumPy Masked Arrays](https://numpy.org/doc/stable/reference/maskedarray.html)
- [NumPy Performance Tips](https://numpy.org/doc/stable/user/basics.performance.html)

---

**Ready to explore more advanced NumPy features? Check out the linear algebra and array manipulation guides!** 