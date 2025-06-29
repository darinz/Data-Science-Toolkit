# TensorFlow Basics Guide

A comprehensive guide to TensorFlow fundamentals, covering tensors, operations, and core concepts.

## Table of Contents

1. [Introduction to TensorFlow](#introduction-to-tensorflow)
2. [Understanding Tensors](#understanding-tensors)
3. [Basic Operations](#basic-operations)
4. [Computational Graphs](#computational-graphs)
5. [Variables and Constants](#variables-and-constants)
6. [Data Types and Shapes](#data-types-and-shapes)
7. [Broadcasting](#broadcasting)
8. [Control Flow](#control-flow)

## Introduction to TensorFlow

TensorFlow is an open-source machine learning framework developed by Google. It provides a comprehensive ecosystem for building and deploying machine learning models.

```python
import tensorflow as tf
import numpy as np

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Check for GPU availability
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

## Understanding Tensors

Tensors are the fundamental data structure in TensorFlow. They are multi-dimensional arrays with a uniform type.

```python
# Creating tensors
# 0D tensor (scalar)
scalar = tf.constant(5)
print(f"Scalar: {scalar}, Shape: {scalar.shape}, Rank: {scalar.ndim}")

# 1D tensor (vector)
vector = tf.constant([1, 2, 3, 4, 5])
print(f"Vector: {vector}, Shape: {vector.shape}, Rank: {vector.ndim}")

# 2D tensor (matrix)
matrix = tf.constant([[1, 2, 3], [4, 5, 6]])
print(f"Matrix: {matrix}, Shape: {matrix.shape}, Rank: {matrix.ndim}")

# 3D tensor
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3D Tensor: {tensor_3d}, Shape: {tensor_3d.shape}, Rank: {tensor_3d.ndim}")

# Creating tensors from Python lists
list_tensor = tf.constant([[1, 2], [3, 4]])
print(f"From list: {list_tensor}")

# Creating tensors from NumPy arrays
numpy_array = np.array([[1, 2], [3, 4]])
numpy_tensor = tf.constant(numpy_array)
print(f"From NumPy: {numpy_tensor}")
```

## Basic Operations

TensorFlow provides a rich set of mathematical operations.

```python
# Arithmetic operations
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

# Addition
c = tf.add(a, b)
print(f"Addition: {c}")

# Subtraction
d = tf.subtract(a, b)
print(f"Subtraction: {d}")

# Multiplication (element-wise)
e = tf.multiply(a, b)
print(f"Element-wise multiplication: {e}")

# Division
f = tf.divide(a, b)
print(f"Division: {f}")

# Matrix multiplication
matrix_a = tf.constant([[1, 2], [3, 4]])
matrix_b = tf.constant([[5, 6], [7, 8]])
matrix_c = tf.matmul(matrix_a, matrix_b)
print(f"Matrix multiplication:\n{matrix_c}")

# Using operators (more Pythonic)
print(f"a + b: {a + b}")
print(f"a - b: {a - b}")
print(f"a * b: {a * b}")
print(f"a / b: {a / b}")
print(f"matrix_a @ matrix_b:\n{matrix_a @ matrix_b}")
```

## Computational Graphs

TensorFlow uses computational graphs to represent operations. In eager execution (default), operations are executed immediately.

```python
# Eager execution (default in TF 2.x)
x = tf.constant(3.0)
y = tf.constant(2.0)
z = x * y + 2
print(f"Result: {z}")

# Building a simple computational graph
def simple_function(x, y):
    return x * y + 2

result = simple_function(tf.constant(3.0), tf.constant(2.0))
print(f"Function result: {result}")

# Using tf.function for graph optimization
@tf.function
def optimized_function(x, y):
    return x * y + 2

result = optimized_function(tf.constant(3.0), tf.constant(2.0))
print(f"Optimized function result: {result}")
```

## Variables and Constants

Variables are mutable tensors that can be updated during training.

```python
# Creating variables
initial_value = tf.random.normal([3, 3])
variable = tf.Variable(initial_value)
print(f"Variable:\n{variable}")

# Updating variables
variable.assign(tf.random.normal([3, 3]))
print(f"Updated variable:\n{variable}")

# Adding to variables
variable.assign_add(tf.ones([3, 3]))
print(f"After adding ones:\n{variable}")

# Variables in computations
x = tf.Variable(2.0)
y = tf.Variable(3.0)
z = x * y
print(f"z = {z}")

# Updating x affects z
x.assign(4.0)
print(f"After updating x, z = {z}")

# Constants (immutable)
constant = tf.constant([1, 2, 3])
print(f"Constant: {constant}")
```

## Data Types and Shapes

TensorFlow supports various data types and provides shape manipulation operations.

```python
# Different data types
int_tensor = tf.constant([1, 2, 3], dtype=tf.int32)
float_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
bool_tensor = tf.constant([True, False, True], dtype=tf.bool)

print(f"Int tensor: {int_tensor}, dtype: {int_tensor.dtype}")
print(f"Float tensor: {float_tensor}, dtype: {float_tensor.dtype}")
print(f"Bool tensor: {bool_tensor}, dtype: {bool_tensor.dtype}")

# Type conversion
converted = tf.cast(int_tensor, tf.float32)
print(f"Converted to float: {converted}")

# Shape operations
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
print(f"Original shape: {tensor.shape}")
print(f"Rank: {tensor.ndim}")
print(f"Size: {tf.size(tensor)}")

# Reshaping
reshaped = tf.reshape(tensor, [6])
print(f"Reshaped to 1D: {reshaped}")

reshaped_2 = tf.reshape(tensor, [3, 2])
print(f"Reshaped to 3x2:\n{reshaped_2}")

# Expanding dimensions
expanded = tf.expand_dims(tensor, axis=0)
print(f"Expanded shape: {expanded.shape}")

# Squeezing dimensions
squeezed = tf.squeeze(expanded)
print(f"Squeezed shape: {squeezed.shape}")
```

## Broadcasting

TensorFlow automatically broadcasts tensors of different shapes during operations.

```python
# Broadcasting examples
# Adding scalar to tensor
tensor = tf.constant([[1, 2, 3], [4, 5, 6]])
scalar = tf.constant(10)
result = tensor + scalar
print(f"Tensor + scalar:\n{result}")

# Broadcasting with different shapes
tensor_2d = tf.constant([[1, 2, 3], [4, 5, 6]])
tensor_1d = tf.constant([10, 20, 30])
result = tensor_2d + tensor_1d
print(f"2D + 1D (broadcasted):\n{result}")

# Broadcasting in matrix operations
matrix = tf.constant([[1, 2], [3, 4]])
vector = tf.constant([10, 20])
result = matrix * vector
print(f"Matrix * vector (broadcasted):\n{result}")
```

## Control Flow

TensorFlow provides control flow operations for conditional and iterative computations.

```python
# Conditional operations
x = tf.constant(5.0)
y = tf.constant(3.0)

# Using tf.cond
result = tf.cond(x > y, 
                 lambda: tf.square(x), 
                 lambda: tf.square(y))
print(f"Conditional result: {result}")

# Using tf.where
condition = tf.constant([True, False, True, False])
x_values = tf.constant([1, 2, 3, 4])
y_values = tf.constant([10, 20, 30, 40])
result = tf.where(condition, x_values, y_values)
print(f"Where result: {result}")

# Iterative operations
def body(i, x):
    return i + 1, x * 2

def condition(i, x):
    return i < 5

initial_i = tf.constant(0)
initial_x = tf.constant(1.0)

final_i, final_x = tf.while_loop(condition, body, [initial_i, initial_x])
print(f"While loop result: i={final_i}, x={final_x}")

# Using tf.map_fn for element-wise operations
tensor = tf.constant([1, 2, 3, 4, 5])
squared = tf.map_fn(lambda x: x * x, tensor)
print(f"Mapped result: {squared}")
```

## Summary

- Tensors are the fundamental data structure in TensorFlow
- TensorFlow provides rich mathematical operations and automatic broadcasting
- Variables are mutable tensors used for model parameters
- Computational graphs optimize execution in TensorFlow
- Control flow operations enable conditional and iterative computations

## Next Steps

- Explore TensorFlow's Keras API for building neural networks
- Learn about data pipelines with tf.data
- Practice with real-world datasets and models 