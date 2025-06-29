# PyTorch Tensor Operations: A Comprehensive Guide

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Table of Contents

1. [Introduction to Tensors](#introduction-to-tensors)
2. [Tensor Creation](#tensor-creation)
3. [Tensor Attributes](#tensor-attributes)
4. [Basic Operations](#basic-operations)
5. [Mathematical Operations](#mathematical-operations)
6. [Indexing and Slicing](#indexing-and-slicing)
7. [Shape Manipulation](#shape-manipulation)
8. [Device Management](#device-management)
9. [Memory Management](#memory-management)
10. [Best Practices](#best-practices)

## Introduction to Tensors

Tensors are the fundamental data structure in PyTorch, similar to NumPy arrays but with additional capabilities for deep learning. They can represent scalars, vectors, matrices, and higher-dimensional data.

### Key Features of PyTorch Tensors:
- **GPU Acceleration**: Can run on CUDA-enabled GPUs for faster computation
- **Automatic Differentiation**: Support for gradient computation (when `requires_grad=True`)
- **Dynamic Computation Graphs**: Enable flexible neural network architectures
- **Rich API**: Extensive operations for mathematical and linear algebra operations

## Tensor Creation

### 1. From Python Data

```python
import torch

# From lists
data = [[1, 2, 3], [4, 5, 6]]
tensor = torch.tensor(data)
print(tensor)
# Output:
# tensor([[1, 2, 3],
#         [4, 5, 6]])

# From single values
scalar = torch.tensor(42)
print(scalar)  # tensor(42)
```

### 2. From NumPy Arrays

```python
import numpy as np

# Create NumPy array
np_array = np.array([[1, 2], [3, 4]])
print(f"NumPy array:\n{np_array}")

# Convert to PyTorch tensor
tensor = torch.from_numpy(np_array)
print(f"PyTorch tensor:\n{tensor}")

# Convert back to NumPy
tensor_np = tensor.numpy()
print(f"Back to NumPy:\n{tensor_np}")
```

### 3. Special Tensor Creation

```python
# Zeros tensor
zeros = torch.zeros(2, 3)
print(f"Zeros tensor:\n{zeros}")

# Ones tensor
ones = torch.ones(2, 3)
print(f"Ones tensor:\n{ones}")

# Random tensor
random_tensor = torch.rand(2, 3)
print(f"Random tensor:\n{random_tensor}")

# Identity matrix
identity = torch.eye(3)
print(f"Identity matrix:\n{identity}")

# Range tensor
range_tensor = torch.arange(0, 10, 2)
print(f"Range tensor: {range_tensor}")

# Linspace tensor
linspace_tensor = torch.linspace(0, 1, 5)
print(f"Linspace tensor: {linspace_tensor}")
```

### 4. From Other Tensors

```python
# Create base tensor
base_tensor = torch.tensor([[1, 2], [3, 4]])

# Create similar tensors
ones_like = torch.ones_like(base_tensor)
zeros_like = torch.zeros_like(base_tensor)
rand_like = torch.rand_like(base_tensor, dtype=torch.float)

print(f"Base tensor:\n{base_tensor}")
print(f"Ones like:\n{ones_like}")
print(f"Zeros like:\n{zeros_like}")
print(f"Random like:\n{rand_like}")
```

## Tensor Attributes

Understanding tensor attributes is crucial for effective tensor manipulation:

```python
# Create a sample tensor
tensor = torch.randn(3, 4, 5)

# Shape (dimensions)
print(f"Shape: {tensor.shape}")
print(f"Size: {tensor.size()}")

# Data type
print(f"Data type: {tensor.dtype}")

# Device (CPU/GPU)
print(f"Device: {tensor.device}")

# Number of elements
print(f"Number of elements: {tensor.numel()}")

# Number of dimensions
print(f"Number of dimensions: {tensor.ndim}")

# Requires gradient
print(f"Requires gradient: {tensor.requires_grad}")
```

## Basic Operations

### 1. Arithmetic Operations

```python
# Create tensors
a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([5, 6, 7, 8])

# Addition
print(f"a + b = {a + b}")
print(f"torch.add(a, b) = {torch.add(a, b)}")

# Subtraction
print(f"a - b = {a - b}")
print(f"torch.sub(a, b) = {torch.sub(a, b)}")

# Multiplication (element-wise)
print(f"a * b = {a * b}")
print(f"torch.mul(a, b) = {torch.mul(a, b)}")

# Division
print(f"a / b = {a / b}")
print(f"torch.div(a, b) = {torch.div(a, b)}")

# Power
print(f"a ** 2 = {a ** 2}")
print(f"torch.pow(a, 2) = {torch.pow(a, 2)}")
```

### 2. In-place Operations

```python
# Create tensor
x = torch.tensor([1, 2, 3, 4])
print(f"Original: {x}")

# In-place addition
x.add_(5)
print(f"After add_(5): {x}")

# In-place multiplication
x.mul_(2)
print(f"After mul_(2): {x}")

# In-place operations modify the original tensor
# They are more memory efficient but can cause issues with autograd
```

### 3. Comparison Operations

```python
# Create tensors
a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([2, 2, 2, 2])

# Element-wise comparison
print(f"a > b: {a > b}")
print(f"a >= b: {a >= b}")
print(f"a == b: {a == b}")
print(f"a != b: {a != b}")

# Boolean operations
print(f"torch.all(a > 0): {torch.all(a > 0)}")
print(f"torch.any(a > 3): {torch.any(a > 3)}")
```

## Mathematical Operations

### 1. Statistical Operations

```python
# Create tensor
tensor = torch.randn(3, 4)
print(f"Tensor:\n{tensor}")

# Mean
print(f"Mean: {tensor.mean()}")
print(f"Mean along dim 0: {tensor.mean(dim=0)}")
print(f"Mean along dim 1: {tensor.mean(dim=1)}")

# Sum
print(f"Sum: {tensor.sum()}")
print(f"Sum along dim 0: {tensor.sum(dim=0)}")

# Standard deviation
print(f"Std: {tensor.std()}")
print(f"Std along dim 0: {tensor.std(dim=0)}")

# Min and Max
print(f"Min: {tensor.min()}")
print(f"Max: {tensor.max()}")
print(f"Min values: {tensor.min(dim=0)}")
print(f"Max values: {tensor.max(dim=0)}")
```

### 2. Linear Algebra Operations

```python
# Matrix multiplication
a = torch.randn(2, 3)
b = torch.randn(3, 2)
print(f"Matrix A:\n{a}")
print(f"Matrix B:\n{b}")

# Matrix multiplication
c = torch.matmul(a, b)
print(f"A @ B:\n{c}")

# Alternative syntax
c_alt = a @ b
print(f"A @ B (alternative):\n{c_alt}")

# Transpose
print(f"Transpose of A:\n{a.T}")

# Determinant (for square matrices)
square_matrix = torch.randn(3, 3)
det = torch.det(square_matrix)
print(f"Determinant: {det}")

# Eigenvalues and eigenvectors
eigenvals, eigenvecs = torch.linalg.eig(square_matrix)
print(f"Eigenvalues: {eigenvals}")
print(f"Eigenvectors:\n{eigenvecs}")
```

### 3. Trigonometric and Other Functions

```python
# Create tensor
x = torch.tensor([0, 0.5, 1.0, 1.5])

# Trigonometric functions
print(f"sin(x): {torch.sin(x)}")
print(f"cos(x): {torch.cos(x)}")
print(f"tan(x): {torch.tan(x)}")

# Exponential and logarithmic
print(f"exp(x): {torch.exp(x)}")
print(f"log(x + 1): {torch.log(x + 1)}")  # log(0) is undefined
print(f"log10(x + 1): {torch.log10(x + 1)}")

# Square root and absolute value
print(f"sqrt(x): {torch.sqrt(x)}")
print(f"abs(x - 1): {torch.abs(x - 1)}")
```

## Indexing and Slicing

### 1. Basic Indexing

```python
# Create 3D tensor
tensor = torch.randn(3, 4, 5)
print(f"Tensor shape: {tensor.shape}")

# Single element indexing
element = tensor[0, 1, 2]
print(f"Element at [0, 1, 2]: {element}")

# Row indexing
row = tensor[0]
print(f"First row shape: {row.shape}")

# Column indexing
column = tensor[:, 1]
print(f"Second column shape: {column.shape}")
```

### 2. Advanced Indexing

```python
# Create tensor
tensor = torch.randn(5, 5)
print(f"Original tensor:\n{tensor}")

# Boolean indexing
mask = tensor > 0
print(f"Positive elements: {tensor[mask]}")

# Integer indexing
indices = torch.tensor([0, 2, 4])
selected = tensor[indices]
print(f"Selected rows:\n{selected}")

# Advanced indexing
rows = torch.tensor([0, 1])
cols = torch.tensor([2, 3])
selected_elements = tensor[rows, cols]
print(f"Selected elements: {selected_elements}")
```

### 3. Slicing

```python
# Create tensor
tensor = torch.randn(4, 4)
print(f"Original tensor:\n{tensor}")

# Basic slicing
slice_1 = tensor[1:3, :]
print(f"Rows 1-2:\n{slice_1}")

slice_2 = tensor[:, 1:3]
print(f"Columns 1-2:\n{slice_2}")

# Step slicing
step_slice = tensor[::2, ::2]
print(f"Every other element:\n{step_slice}")

# Negative indexing
last_row = tensor[-1]
print(f"Last row: {last_row}")
```

## Shape Manipulation

### 1. Reshaping

```python
# Create tensor
tensor = torch.randn(2, 3, 4)
print(f"Original shape: {tensor.shape}")

# Reshape
reshaped = tensor.reshape(6, 4)
print(f"Reshaped to (6, 4): {reshaped.shape}")

# Flatten
flattened = tensor.flatten()
print(f"Flattened: {flattened.shape}")

# Squeeze and unsqueeze
# Remove dimensions of size 1
squeezed = tensor.squeeze()
print(f"Squeezed shape: {squeezed.shape}")

# Add dimension of size 1
unsqueezed = tensor.unsqueeze(0)
print(f"Unsqueezed shape: {unsqueezed.shape}")
```

### 2. Concatenation and Stacking

```python
# Create tensors
a = torch.randn(2, 3)
b = torch.randn(2, 3)
c = torch.randn(2, 3)

print(f"Tensor A:\n{a}")
print(f"Tensor B:\n{b}")
print(f"Tensor C:\n{c}")

# Concatenate along dimension 0 (rows)
cat_0 = torch.cat([a, b, c], dim=0)
print(f"Concatenated along dim 0:\n{cat_0}")

# Concatenate along dimension 1 (columns)
cat_1 = torch.cat([a, b, c], dim=1)
print(f"Concatenated along dim 1:\n{cat_1}")

# Stack (adds new dimension)
stacked = torch.stack([a, b, c])
print(f"Stacked:\n{stacked}")
print(f"Stacked shape: {stacked.shape}")
```

### 3. Splitting

```python
# Create tensor
tensor = torch.randn(6, 4)
print(f"Original tensor:\n{tensor}")

# Split into equal parts
parts = torch.split(tensor, 2, dim=0)
print(f"Split into 3 parts:")
for i, part in enumerate(parts):
    print(f"Part {i}:\n{part}")

# Chunk into specified number of parts
chunks = torch.chunk(tensor, 3, dim=0)
print(f"Chunked into 3 parts:")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:\n{chunk}")
```

## Device Management

### 1. Moving Tensors Between Devices

```python
# Check available devices
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Create tensor on CPU
cpu_tensor = torch.randn(3, 3)
print(f"CPU tensor device: {cpu_tensor.device}")

# Move to GPU (if available)
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.cuda()
    print(f"GPU tensor device: {gpu_tensor.device}")
    
    # Move back to CPU
    cpu_tensor_2 = gpu_tensor.cpu()
    print(f"Back to CPU device: {cpu_tensor_2.device}")

# Move to MPS (Apple Silicon)
if torch.backends.mps.is_available():
    mps_tensor = cpu_tensor.to('mps')
    print(f"MPS tensor device: {mps_tensor.device}")
```

### 2. Device-Agnostic Code

```python
# Determine the best available device
device = torch.device('cuda' if torch.cuda.is_available() 
                     else 'mps' if torch.backends.mps.is_available() 
                     else 'cpu')
print(f"Using device: {device}")

# Create tensor on the selected device
tensor = torch.randn(3, 3, device=device)
print(f"Tensor device: {tensor.device}")

# Move tensor to device
tensor_on_device = torch.randn(3, 3).to(device)
print(f"Tensor on device: {tensor_on_device.device}")
```

## Memory Management

### 1. Memory Efficiency

```python
# In-place operations (memory efficient)
x = torch.randn(1000, 1000)
print(f"Memory before: {x.element_size() * x.nelement()} bytes")

# In-place operation
x.add_(1.0)
print(f"Memory after in-place add: {x.element_size() * x.nelement()} bytes")

# Out-of-place operation (creates new tensor)
y = x + 1.0
print(f"Memory after out-of-place add: {y.element_size() * y.nelement()} bytes")
```

### 2. Gradient Memory

```python
# Create tensor with gradients
x = torch.randn(3, 3, requires_grad=True)
print(f"Requires grad: {x.requires_grad}")

# Perform operations
y = x ** 2
z = y.sum()

# Compute gradients
z.backward()
print(f"Gradients:\n{x.grad}")

# Clear gradients
x.grad.zero_()
print(f"Gradients after zero_: {x.grad}")
```

### 3. Memory Cleanup

```python
import gc

# Create large tensor
large_tensor = torch.randn(10000, 10000)

# Delete reference
del large_tensor

# Force garbage collection
gc.collect()

# Clear CUDA cache (if using GPU)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

## Best Practices

### 1. Performance Tips

```python
# Use appropriate data types
# Float32 for most operations
tensor_float32 = torch.randn(1000, 1000, dtype=torch.float32)

# Float16 for memory efficiency (if supported)
tensor_float16 = torch.randn(1000, 1000, dtype=torch.float16)

# Use in-place operations when possible
# Good
x.add_(1.0)

# Less efficient
x = x + 1.0
```

### 2. Common Patterns

```python
# Broadcasting
a = torch.randn(3, 1)
b = torch.randn(1, 4)
c = a + b  # Broadcasting happens automatically
print(f"Broadcasted result shape: {c.shape}")

# Vectorization
# Good - vectorized
result = torch.sum(tensor, dim=0)

# Avoid - loops
result_loop = torch.zeros(tensor.shape[1])
for i in range(tensor.shape[0]):
    result_loop += tensor[i]
```

### 3. Error Handling

```python
# Check tensor shapes before operations
def safe_matmul(a, b):
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return torch.matmul(a, b)

# Check device compatibility
def safe_add(a, b):
    if a.device != b.device:
        b = b.to(a.device)
    return a + b
```

## Summary

This guide covers the essential tensor operations in PyTorch:

1. **Creation**: Multiple ways to create tensors from different data sources
2. **Attributes**: Understanding tensor properties like shape, dtype, and device
3. **Operations**: Mathematical, logical, and linear algebra operations
4. **Indexing**: Accessing and modifying tensor elements
5. **Shape Manipulation**: Reshaping, concatenating, and splitting tensors
6. **Device Management**: Moving tensors between CPU and GPU
7. **Memory Management**: Efficient memory usage and cleanup
8. **Best Practices**: Performance optimization and common patterns

Mastering these tensor operations is fundamental for building effective deep learning models with PyTorch. Practice with different tensor shapes and operations to build intuition for working with multi-dimensional data.

## Next Steps

- Explore the [Autograd Guide](../Autograd/) to learn about automatic differentiation
- Check out the [Neural Networks Guide](../Neural-Networks/) for building models
- Practice with the [Image Classification Tutorial](../Image-Classifier/) for real-world applications

## References

- [PyTorch Tensor Documentation](https://pytorch.org/docs/stable/tensors.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [NumPy vs PyTorch Comparison](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html) 