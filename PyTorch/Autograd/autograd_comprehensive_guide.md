# PyTorch Autograd: Automatic Differentiation Guide

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Table of Contents

1. [Introduction to Autograd](#introduction-to-autograd)
2. [Basic Gradient Computation](#basic-gradient-computation)
3. [Gradient Accumulation](#gradient-accumulation)
4. [Gradient Flow Control](#gradient-flow-control)
5. [Custom Gradients](#custom-gradients)
6. [Gradient Clipping](#gradient-clipping)
7. [Memory Management](#memory-management)
8. [Advanced Autograd Features](#advanced-autograd-features)
9. [Debugging Gradients](#debugging-gradients)
10. [Best Practices](#best-practices)

## Introduction to Autograd

Autograd is PyTorch's automatic differentiation engine that powers neural network training. It automatically computes gradients of tensors with respect to their inputs, enabling efficient backpropagation through complex computational graphs.

### Key Concepts:

- **Computational Graph**: A directed acyclic graph representing the sequence of operations
- **Gradients**: Partial derivatives of the output with respect to each input
- **Backpropagation**: Algorithm for efficiently computing gradients using the chain rule
- **requires_grad**: Flag indicating whether gradients should be computed for a tensor

### Why Autograd Matters:

```python
import torch

# Without autograd (manual differentiation)
def manual_gradient(x):
    # Forward pass
    y = x ** 2 + 3 * x + 1
    
    # Manual gradient computation
    dy_dx = 2 * x + 3
    return y, dy_dx

# With autograd (automatic differentiation)
def autograd_gradient(x):
    x.requires_grad_(True)  # Enable gradient computation
    
    # Forward pass
    y = x ** 2 + 3 * x + 1
    
    # Automatic gradient computation
    y.backward()
    return y, x.grad

# Compare results
x_manual = torch.tensor(2.0)
x_autograd = torch.tensor(2.0)

y_manual, grad_manual = manual_gradient(x_manual)
y_autograd, grad_autograd = autograd_gradient(x_autograd)

print(f"Manual gradient: {grad_manual}")
print(f"Autograd gradient: {grad_autograd}")
```

## Basic Gradient Computation

### 1. Simple Scalar Functions

```python
import torch

# Create tensor with gradients enabled
x = torch.tensor(2.0, requires_grad=True)
print(f"x: {x}")
print(f"requires_grad: {x.requires_grad}")

# Define a simple function
y = x ** 2 + 3 * x + 1
print(f"y = x² + 3x + 1 = {y}")

# Compute gradients
y.backward()
print(f"dy/dx = 2x + 3 = {x.grad}")

# Verify manually
expected_grad = 2 * 2.0 + 3
print(f"Expected gradient: {expected_grad}")
```

### 2. Multi-Variable Functions

```python
# Create multiple variables
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Define function of multiple variables
z = x ** 2 + y ** 2 + x * y
print(f"z = x² + y² + xy = {z}")

# Compute gradients
z.backward()
print(f"∂z/∂x = 2x + y = {x.grad}")
print(f"∂z/∂y = 2y + x = {y.grad}")

# Verify manually
expected_dz_dx = 2 * 2.0 + 3.0
expected_dz_dy = 2 * 3.0 + 2.0
print(f"Expected ∂z/∂x: {expected_dz_dx}")
print(f"Expected ∂z/∂y: {expected_dz_dy}")
```

### 3. Vector-Valued Functions

```python
# Create input tensor
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"Input x: {x}")

# Define vector-valued function
y = torch.sum(x ** 2)  # Scalar output
print(f"y = Σx² = {y}")

# Compute gradients
y.backward()
print(f"Gradients: {x.grad}")

# Verify manually
expected_grads = 2 * x.detach()
print(f"Expected gradients: {expected_grads}")
```

### 4. Matrix Operations

```python
# Create matrix
A = torch.randn(2, 2, requires_grad=True)
print(f"Matrix A:\n{A}")

# Define function
y = torch.det(A)  # Determinant
print(f"Determinant: {y}")

# Compute gradients
y.backward()
print(f"Gradients:\n{A.grad}")

# For determinant, gradients are related to adjugate matrix
adjugate = torch.tensor([[A[1, 1].item(), -A[0, 1].item()],
                        [-A[1, 0].item(), A[0, 0].item()]])
print(f"Adjugate matrix:\n{adjugate}")
```

## Gradient Accumulation

### 1. Understanding Gradient Accumulation

```python
import torch

# Create tensor
x = torch.tensor(2.0, requires_grad=True)

# First backward pass
y1 = x ** 2
y1.backward()
print(f"After first backward: {x.grad}")

# Second backward pass (gradients accumulate)
y2 = x ** 3
y2.backward()
print(f"After second backward: {x.grad}")

# Gradients accumulate: dy1/dx + dy2/dx = 2x + 3x²
expected_total = 2 * 2.0 + 3 * (2.0 ** 2)
print(f"Expected total gradient: {expected_total}")
```

### 2. Clearing Gradients

```python
# Create tensor
x = torch.tensor(2.0, requires_grad=True)

# First computation
y1 = x ** 2
y1.backward()
print(f"Gradient after y1: {x.grad}")

# Clear gradients
x.grad.zero_()
print(f"Gradient after zero_: {x.grad}")

# Second computation
y2 = x ** 3
y2.backward()
print(f"Gradient after y2: {x.grad}")
```

### 3. Gradient Accumulation in Training

```python
# Simulate mini-batch training
x = torch.tensor(2.0, requires_grad=True)
optimizer = torch.optim.SGD([x], lr=0.1)

# Process multiple batches
for batch in range(3):
    # Forward pass
    y = x ** 2
    
    # Backward pass
    y.backward()
    
    print(f"Batch {batch + 1}: gradient = {x.grad}")
    
    # Update parameters
    optimizer.step()
    
    # Clear gradients for next batch
    optimizer.zero_grad()
    
    print(f"Batch {batch + 1}: x = {x}")
```

## Gradient Flow Control

### 1. Detaching Tensors

```python
# Create tensor
x = torch.tensor(2.0, requires_grad=True)

# Normal computation
y = x ** 2
print(f"y.requires_grad: {y.requires_grad}")

# Detached computation
y_detached = x.detach() ** 2
print(f"y_detached.requires_grad: {y_detached.requires_grad}")

# Mixed computation
z = y + y_detached
print(f"z.requires_grad: {z.requires_grad}")

# Backward pass
z.backward()
print(f"Gradient: {x.grad}")
```

### 2. Using torch.no_grad()

```python
import torch

# Create tensor
x = torch.tensor(2.0, requires_grad=True)

# Normal computation
y = x ** 2
print(f"y.requires_grad: {y.requires_grad}")

# Computation without gradients
with torch.no_grad():
    y_no_grad = x ** 2
    print(f"y_no_grad.requires_grad: {y_no_grad.requires_grad}")

# Context manager prevents gradient computation
print(f"x.grad before: {x.grad}")
# y_no_grad.backward()  # This will raise an error
```

### 3. Gradient Enabling/Disabling

```python
# Create tensor
x = torch.tensor(2.0, requires_grad=True)

# Disable gradients temporarily
x.requires_grad_(False)
y1 = x ** 2
print(f"y1.requires_grad: {y1.requires_grad}")

# Re-enable gradients
x.requires_grad_(True)
y2 = x ** 2
print(f"y2.requires_grad: {y2.requires_grad}")

# Compute gradients
y2.backward()
print(f"Gradient: {x.grad}")
```

### 4. Conditional Gradient Computation

```python
def conditional_function(x, compute_grad=True):
    if compute_grad:
        x.requires_grad_(True)
    
    y = x ** 2 + 3 * x + 1
    
    if compute_grad:
        y.backward()
        return y, x.grad
    else:
        return y, None

# With gradients
x1 = torch.tensor(2.0)
y1, grad1 = conditional_function(x1, compute_grad=True)
print(f"With gradients: y={y1}, grad={grad1}")

# Without gradients
x2 = torch.tensor(2.0)
y2, grad2 = conditional_function(x2, compute_grad=False)
print(f"Without gradients: y={y2}, grad={grad2}")
```

## Custom Gradients

### 1. Using torch.autograd.Function

```python
import torch
from torch.autograd import Function

class CustomFunction(Function):
    @staticmethod
    def forward(ctx, x):
        # Store tensors for backward pass
        ctx.save_for_backward(x)
        return x ** 2
    
    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve stored tensors
        x, = ctx.saved_tensors
        # Compute gradient
        grad_input = 2 * x * grad_output
        return grad_input

# Use custom function
x = torch.tensor(2.0, requires_grad=True)
y = CustomFunction.apply(x)
y.backward()
print(f"Custom gradient: {x.grad}")
```

### 2. Custom Activation Function

```python
class CustomReLU(Function):
    @staticmethod
    def forward(ctx, x):
        # Store mask for backward pass
        mask = (x > 0).float()
        ctx.save_for_backward(mask)
        return x * mask
    
    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        return grad_output * mask

# Test custom ReLU
x = torch.tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
y = CustomReLU.apply(x)
y.backward(torch.ones_like(y))
print(f"Input: {x}")
print(f"Output: {y}")
print(f"Gradients: {x.grad}")
```

### 3. Complex Custom Function

```python
class SigmoidFunction(Function):
    @staticmethod
    def forward(ctx, x):
        # Compute sigmoid
        sigmoid = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(sigmoid)
        return sigmoid
    
    @staticmethod
    def backward(ctx, grad_output):
        sigmoid, = ctx.saved_tensors
        # Gradient of sigmoid: sigmoid * (1 - sigmoid)
        grad_input = sigmoid * (1 - sigmoid) * grad_output
        return grad_input

# Test custom sigmoid
x = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
y = SigmoidFunction.apply(x)
y.backward(torch.ones_like(y))
print(f"Input: {x}")
print(f"Output: {y}")
print(f"Gradients: {x.grad}")
```

## Gradient Clipping

### 1. Basic Gradient Clipping

```python
import torch
import torch.nn as nn

# Create model and optimizer
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Simulate training step
x = torch.randn(32, 10)
target = torch.randn(32, 1)

# Forward pass
output = model(x)
loss = nn.MSELoss()(output, target)

# Backward pass
loss.backward()

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Check gradient norms
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** (1. / 2)
print(f"Gradient norm after clipping: {total_norm}")

# Update parameters
optimizer.step()
optimizer.zero_grad()
```

### 2. Value-Based Clipping

```python
# Create tensor with large gradients
x = torch.tensor([10.0, -15.0, 20.0], requires_grad=True)
y = torch.sum(x ** 2)
y.backward()

print(f"Original gradients: {x.grad}")

# Clip gradient values
torch.nn.utils.clip_grad_value_(x, clip_value=5.0)
print(f"Clipped gradients: {x.grad}")
```

### 3. Custom Clipping Function

```python
def custom_gradient_clipping(parameters, max_norm, norm_type=2):
    """
    Custom gradient clipping function
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0
    
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    
    total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    
    return total_norm

# Test custom clipping
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = torch.sum(x ** 2)
y.backward()

print(f"Original gradients: {x.grad}")
norm_before = custom_gradient_clipping(x, max_norm=2.0)
print(f"Gradient norm: {norm_before}")
print(f"Clipped gradients: {x.grad}")
```

## Memory Management

### 1. Gradient Memory Cleanup

```python
import torch
import gc

# Create tensor with gradients
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()

print(f"Gradient: {x.grad}")

# Clear gradients
x.grad.zero_()
print(f"After zero_: {x.grad}")

# Delete references
del x, y
gc.collect()

# Clear CUDA cache if using GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 2. Memory-Efficient Training

```python
# Memory-efficient approach
def memory_efficient_training():
    model = torch.nn.Linear(1000, 1000)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(10):
        for batch in range(100):
            # Generate data
            x = torch.randn(32, 1000)
            target = torch.randn(32, 1000)
            
            # Forward pass
            output = model(x)
            loss = torch.nn.MSELoss()(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            
            # Clear intermediate variables
            del x, target, output, loss
            
            # Force garbage collection periodically
            if batch % 10 == 0:
                gc.collect()
```

### 3. Gradient Checkpointing

```python
import torch
import torch.utils.checkpoint as checkpoint

class LargeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(1000, 1000) for _ in range(10)
        ])
    
    def forward(self, x, use_checkpointing=False):
        if use_checkpointing:
            # Use gradient checkpointing to save memory
            for layer in self.layers:
                x = checkpoint.checkpoint(layer, x)
        else:
            # Normal forward pass
            for layer in self.layers:
                x = layer(x)
        return x

# Compare memory usage
model = LargeModel()
x = torch.randn(32, 1000, requires_grad=True)

# Without checkpointing
output1 = model(x, use_checkpointing=False)
loss1 = output1.sum()
loss1.backward()

# With checkpointing
output2 = model(x, use_checkpointing=True)
loss2 = output2.sum()
loss2.backward()
```

## Advanced Autograd Features

### 1. Higher-Order Gradients

```python
import torch

# Create tensor
x = torch.tensor(2.0, requires_grad=True)

# First derivative
y = x ** 3
y.backward(create_graph=True)  # Enable higher-order gradients
first_derivative = x.grad.clone()
print(f"First derivative: {first_derivative}")

# Clear gradients
x.grad.zero_()

# Second derivative
first_derivative.backward()
second_derivative = x.grad
print(f"Second derivative: {second_derivative}")

# Verify manually
expected_first = 3 * (2.0 ** 2)
expected_second = 6 * 2.0
print(f"Expected first derivative: {expected_first}")
print(f"Expected second derivative: {expected_second}")
```

### 2. Jacobian and Hessian

```python
def compute_jacobian(func, x):
    """Compute Jacobian matrix of func at x"""
    x.requires_grad_(True)
    y = func(x)
    
    jacobian = torch.zeros(y.shape[0], x.shape[0])
    for i in range(y.shape[0]):
        x.grad.zero_()
        y[i].backward(retain_graph=True)
        jacobian[i] = x.grad.clone()
    
    return jacobian

def compute_hessian(func, x):
    """Compute Hessian matrix of func at x"""
    x.requires_grad_(True)
    y = func(x)
    
    # Compute gradients
    grad = torch.autograd.grad(y, x, create_graph=True)[0]
    
    # Compute Hessian
    hessian = torch.zeros(x.shape[0], x.shape[0])
    for i in range(x.shape[0]):
        x.grad.zero_()
        grad[i].backward(retain_graph=True)
        hessian[i] = x.grad.clone()
    
    return hessian

# Test functions
def quadratic_function(x):
    return x[0]**2 + x[1]**2 + x[0]*x[1]

x = torch.tensor([1.0, 2.0])
jacobian = compute_jacobian(quadratic_function, x)
hessian = compute_hessian(quadratic_function, x)

print(f"Jacobian:\n{jacobian}")
print(f"Hessian:\n{hessian}")
```

### 3. Vector-Jacobian Products

```python
def vector_jacobian_product(func, x, v):
    """Compute vector-Jacobian product v^T * J"""
    x.requires_grad_(True)
    y = func(x)
    
    # Compute gradients with respect to v
    vjp = torch.autograd.grad(y, x, grad_outputs=v, retain_graph=True)[0]
    return vjp

# Test
def test_function(x):
    return torch.stack([x[0]**2, x[1]**2])

x = torch.tensor([2.0, 3.0])
v = torch.tensor([1.0, 1.0])

vjp = vector_jacobian_product(test_function, x, v)
print(f"Vector-Jacobian product: {vjp}")
```

## Debugging Gradients

### 1. Gradient Checking

```python
def gradient_check(func, x, epsilon=1e-7):
    """Numerical gradient checking"""
    x.requires_grad_(True)
    y = func(x)
    y.backward()
    analytical_grad = x.grad.clone()
    
    # Numerical gradient
    numerical_grad = torch.zeros_like(x)
    for i in range(x.numel()):
        x_plus = x.clone()
        x_minus = x.clone()
        x_plus.flatten()[i] += epsilon
        x_minus.flatten()[i] -= epsilon
        
        y_plus = func(x_plus)
        y_minus = func(x_minus)
        
        numerical_grad.flatten()[i] = (y_plus - y_minus) / (2 * epsilon)
    
    # Compare gradients
    diff = torch.abs(analytical_grad - numerical_grad).max()
    print(f"Maximum difference: {diff}")
    print(f"Gradients match: {diff < 1e-5}")
    
    return analytical_grad, numerical_grad

# Test gradient checking
def test_func(x):
    return torch.sum(x ** 2)

x = torch.tensor([1.0, 2.0, 3.0])
analytical, numerical = gradient_check(test_func, x)
print(f"Analytical gradient: {analytical}")
print(f"Numerical gradient: {numerical}")
```

### 2. Gradient Visualization

```python
import matplotlib.pyplot as plt

def visualize_gradients(model, loss_fn, data_loader):
    """Visualize gradient norms during training"""
    gradient_norms = []
    
    for batch_idx, (data, target) in enumerate(data_loader):
        # Forward pass
        output = model(data)
        loss = loss_fn(output, target)
        
        # Backward pass
        loss.backward()
        
        # Compute gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        gradient_norms.append(total_norm)
        
        # Clear gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
    
    # Plot gradient norms
    plt.figure(figsize=(10, 6))
    plt.plot(gradient_norms)
    plt.title('Gradient Norms During Training')
    plt.xlabel('Batch')
    plt.ylabel('Gradient Norm')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    return gradient_norms
```

### 3. Gradient Flow Analysis

```python
def analyze_gradient_flow(model):
    """Analyze gradient flow through the model"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm(2).item()
            param_norm = param.norm(2).item()
            relative_grad = grad_norm / (param_norm + 1e-8)
            
            print(f"{name}:")
            print(f"  Gradient norm: {grad_norm:.6f}")
            print(f"  Parameter norm: {param_norm:.6f}")
            print(f"  Relative gradient: {relative_grad:.6f}")
            print()

# Example usage
model = torch.nn.Sequential(
    torch.nn.Linear(10, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 1)
)

# Simulate training step
x = torch.randn(32, 10)
target = torch.randn(32, 1)
output = model(x)
loss = torch.nn.MSELoss()(output, target)
loss.backward()

analyze_gradient_flow(model)
```

## Best Practices

### 1. Efficient Gradient Computation

```python
# Good: Use in-place operations when possible
def efficient_gradient_computation():
    x = torch.randn(1000, 1000, requires_grad=True)
    
    # Efficient: in-place operations
    y = x.clone()
    y.add_(1.0)
    y.mul_(2.0)
    
    # Less efficient: creates new tensors
    # y = (x + 1.0) * 2.0
    
    return y

# Good: Clear gradients properly
def proper_gradient_clearing():
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(10):
        for batch in range(100):
            # Forward pass
            x = torch.randn(32, 10)
            target = torch.randn(32, 1)
            output = model(x)
            loss = torch.nn.MSELoss()(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()  # Clear gradients
```

### 2. Memory Management

```python
# Good: Use gradient checkpointing for large models
def memory_efficient_large_model():
    model = torch.nn.Sequential(
        *[torch.nn.Linear(1000, 1000) for _ in range(20)]
    )
    
    # Use checkpointing to save memory
    def forward_with_checkpointing(x):
        return torch.utils.checkpoint.checkpoint_sequential(
            model, 5, x
        )
    
    return forward_with_checkpointing

# Good: Proper cleanup
def proper_cleanup():
    # Create tensors
    x = torch.randn(1000, 1000, requires_grad=True)
    y = x ** 2
    y.backward()
    
    # Clean up
    del x, y
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
```

### 3. Debugging and Monitoring

```python
# Good: Monitor gradient norms
def monitor_gradients(model, loss_fn, data_loader):
    gradient_norms = []
    
    for batch_idx, (data, target) in enumerate(data_loader):
        # Forward pass
        output = model(data)
        loss = loss_fn(output, target)
        
        # Backward pass
        loss.backward()
        
        # Monitor gradients
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        gradient_norms.append(total_norm)
        
        # Check for exploding gradients
        if total_norm > 10.0:
            print(f"Warning: Large gradient norm detected: {total_norm}")
        
        # Clear gradients
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
    
    return gradient_norms

# Good: Gradient clipping
def safe_training(model, optimizer, data_loader, max_norm=1.0):
    for batch_idx, (data, target) in enumerate(data_loader):
        # Forward pass
        output = model(data)
        loss = torch.nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Update parameters
        optimizer.step()
        optimizer.zero_grad()
```

## Summary

This comprehensive guide covers PyTorch's autograd system:

1. **Basic Concepts**: Understanding automatic differentiation and computational graphs
2. **Gradient Computation**: Computing gradients for various types of functions
3. **Gradient Control**: Managing gradient flow and accumulation
4. **Custom Gradients**: Implementing custom gradient functions
5. **Gradient Clipping**: Preventing gradient explosion
6. **Memory Management**: Efficient memory usage during training
7. **Advanced Features**: Higher-order gradients and vector-Jacobian products
8. **Debugging**: Tools for gradient checking and visualization
9. **Best Practices**: Efficient and robust gradient computation

Mastering autograd is essential for effective deep learning with PyTorch. Understanding how gradients flow through your models will help you debug issues, optimize performance, and implement custom training procedures.

## Next Steps

- Explore the [Neural Networks Guide](../Neural-Networks/) to build complete models
- Check out the [Image Classification Tutorial](../Image-Classifier/) for real-world applications
- Practice implementing custom loss functions and training loops

## References

- [PyTorch Autograd Documentation](https://pytorch.org/docs/stable/autograd.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Automatic Differentiation in Machine Learning](https://arxiv.org/abs/1502.05767) 