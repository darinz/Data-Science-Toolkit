# NumPy Linear Algebra: A Comprehensive Guide

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

## Table of Contents

1. [Introduction to Linear Algebra with NumPy](#introduction-to-linear-algebra-with-numpy)
2. [Matrix Creation and Basic Operations](#matrix-creation-and-basic-operations)
3. [Matrix Multiplication and Properties](#matrix-multiplication-and-properties)
4. [Linear System Solving](#linear-system-solving)
5. [Matrix Decompositions](#matrix-decompositions)
6. [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
7. [Vector Operations](#vector-operations)
8. [Matrix Norms and Condition Numbers](#matrix-norms-and-condition-numbers)
9. [Special Matrices](#special-matrices)
10. [Applications in Data Science](#applications-in-data-science)
11. [Performance Considerations](#performance-considerations)
12. [Best Practices](#best-practices)
13. [Common Pitfalls](#common-pitfalls)

## Introduction to Linear Algebra with NumPy

NumPy provides comprehensive linear algebra capabilities through the `numpy.linalg` module. This guide covers:

- **Matrix operations** and properties
- **Linear system solving** (Ax = b)
- **Matrix decompositions** (LU, QR, SVD, Cholesky)
- **Eigenvalue problems** and diagonalization
- **Vector operations** and norms
- **Applications** in data science and machine learning

### Importing Linear Algebra Functions

```python
import numpy as np
import numpy.linalg as la

# Common imports for linear algebra
from numpy.linalg import inv, det, eig, svd, qr, cholesky, solve, norm
```

## Matrix Creation and Basic Operations

### 1. Creating Matrices

```python
# From lists
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Matrix A:\n{A}")

# Using NumPy functions
zeros_matrix = np.zeros((3, 3))
ones_matrix = np.ones((2, 4))
identity_matrix = np.eye(3)
random_matrix = np.random.random((3, 3))

print(f"\nZeros matrix:\n{zeros_matrix}")
print(f"Ones matrix:\n{ones_matrix}")
print(f"Identity matrix:\n{identity_matrix}")
print(f"Random matrix:\n{random_matrix}")

# Special matrices
diagonal_matrix = np.diag([1, 2, 3, 4])
print(f"\nDiagonal matrix:\n{diagonal_matrix}")
```

### 2. Matrix Properties

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Basic properties
print(f"Matrix A:\n{A}")
print(f"Shape: {A.shape}")
print(f"Size: {A.size}")
print(f"Number of dimensions: {A.ndim}")
print(f"Data type: {A.dtype}")

# Matrix characteristics
print(f"\nIs square: {A.shape[0] == A.shape[1]}")
print(f"Is symmetric: {np.allclose(A, A.T)}")
print(f"Is diagonal: {np.allclose(A, np.diag(np.diag(A)))}")
```

### 3. Basic Matrix Operations

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")

# Element-wise operations
print(f"\nA + B:\n{A + B}")
print(f"A - B:\n{A - B}")
print(f"A * B (element-wise):\n{A * B}")
print(f"A / B (element-wise):\n{A / B}")

# Scalar operations
scalar = 2
print(f"\nA * {scalar}:\n{A * scalar}")
print(f"A + {scalar}:\n{A + scalar}")
```

## Matrix Multiplication and Properties

### 1. Matrix Multiplication

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication using @ operator
C = A @ B
print(f"Matrix A:\n{A}")
print(f"Matrix B:\n{B}")
print(f"A @ B:\n{C}")

# Matrix multiplication using np.matmul
C_alt = np.matmul(A, B)
print(f"\nUsing np.matmul:\n{C_alt}")

# Matrix multiplication using np.dot
C_dot = np.dot(A, B)
print(f"\nUsing np.dot:\n{C_dot}")

# Verify all methods give same result
print(f"\nAll methods equal: {np.allclose(C, C_alt) and np.allclose(C, C_dot)}")
```

### 2. Matrix Properties and Theorems

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.array([[9, 10], [11, 12]])

# Associativity: (AB)C = A(BC)
left = (A @ B) @ C
right = A @ (B @ C)
print(f"Associativity test: {np.allclose(left, right)}")

# Distributivity: A(B + C) = AB + AC
left_dist = A @ (B + C)
right_dist = A @ B + A @ C
print(f"Distributivity test: {np.allclose(left_dist, right_dist)}")

# Transpose properties
print(f"\n(A @ B)^T = B^T @ A^T: {np.allclose((A @ B).T, B.T @ A.T)}")
```

### 3. Matrix Powers

```python
A = np.array([[1, 1], [0, 1]])

# Matrix powers
A_squared = A @ A
A_cubed = A @ A @ A
A_power_4 = la.matrix_power(A, 4)

print(f"Matrix A:\n{A}")
print(f"A^2:\n{A_squared}")
print(f"A^3:\n{A_cubed}")
print(f"A^4:\n{A_power_4}")

# Using matrix_power function
A_power_5 = la.matrix_power(A, 5)
print(f"\nA^5 using matrix_power:\n{A_power_5}")
```

## Linear System Solving

### 1. Basic Linear System Solving

```python
# System: 2x + y = 5, x + 3y = 6
A = np.array([[2, 1], [1, 3]])
b = np.array([5, 6])

print(f"Coefficient matrix A:\n{A}")
print(f"Right-hand side b: {b}")

# Solve using np.linalg.solve
x = la.solve(A, b)
print(f"Solution x: {x}")

# Verify solution
residual = A @ x - b
print(f"Residual (should be close to zero): {residual}")
print(f"Solution is correct: {np.allclose(residual, 0)}")
```

### 2. Multiple Right-Hand Sides

```python
A = np.array([[2, 1], [1, 3]])
B = np.array([[5, 1], [6, 2]])

print(f"Coefficient matrix A:\n{A}")
print(f"Multiple right-hand sides B:\n{B}")

# Solve for multiple right-hand sides
X = la.solve(A, B)
print(f"Solution matrix X:\n{X}")

# Verify solutions
residuals = A @ X - B
print(f"Residuals:\n{residuals}")
print(f"All solutions correct: {np.allclose(residuals, 0)}")
```

### 3. Underdetermined and Overdetermined Systems

```python
# Underdetermined system (more variables than equations)
A_under = np.array([[1, 2, 3], [4, 5, 6]])
b_under = np.array([7, 8])

print(f"Underdetermined system:")
print(f"A:\n{A_under}")
print(f"b: {b_under}")

# Use least squares for underdetermined system
x_under = la.lstsq(A_under, b_under, rcond=None)[0]
print(f"Least squares solution: {x_under}")

# Overdetermined system (more equations than variables)
A_over = np.array([[1, 2], [3, 4], [5, 6]])
b_over = np.array([7, 8, 9])

print(f"\nOverdetermined system:")
print(f"A:\n{A_over}")
print(f"b: {b_over}")

# Use least squares for overdetermined system
x_over = la.lstsq(A_over, b_over, rcond=None)[0]
print(f"Least squares solution: {x_over}")
```

## Matrix Decompositions

### 1. LU Decomposition

```python
A = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])

print(f"Matrix A:\n{A}")

# LU decomposition
P, L, U = la.lu(A)
print(f"\nPermutation matrix P:\n{P}")
print(f"Lower triangular L:\n{L}")
print(f"Upper triangular U:\n{U}")

# Verify decomposition
reconstructed = P @ L @ U
print(f"\nReconstructed A:\n{reconstructed}")
print(f"Decomposition is correct: {np.allclose(A, reconstructed)}")

# Solve system using LU decomposition
b = np.array([5, -2, 9])
x_lu = la.solve(A, b)
print(f"\nSolution using LU: {x_lu}")
```

### 2. QR Decomposition

```python
A = np.array([[1, 2], [3, 4], [5, 6]])

print(f"Matrix A:\n{A}")

# QR decomposition
Q, R = la.qr(A)
print(f"\nOrthogonal matrix Q:\n{Q}")
print(f"Upper triangular R:\n{R}")

# Verify decomposition
reconstructed = Q @ R
print(f"\nReconstructed A:\n{reconstructed}")
print(f"Decomposition is correct: {np.allclose(A, reconstructed)}")

# Verify Q is orthogonal
Q_orthogonal = np.allclose(Q.T @ Q, np.eye(Q.shape[1]))
print(f"Q is orthogonal: {Q_orthogonal}")
```

### 3. Singular Value Decomposition (SVD)

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(f"Matrix A:\n{A}")

# SVD decomposition
U, s, Vt = la.svd(A)
print(f"\nLeft singular vectors U:\n{U}")
print(f"Singular values s: {s}")
print(f"Right singular vectors V^T:\n{Vt}")

# Reconstruct matrix
S = np.zeros_like(A, dtype=float)
S[:len(s), :len(s)] = np.diag(s)
reconstructed = U @ S @ Vt

print(f"\nReconstructed A:\n{reconstructed}")
print(f"Decomposition is correct: {np.allclose(A, reconstructed)}")

# Rank of matrix
rank = np.sum(s > 1e-10)
print(f"Rank of A: {rank}")
```

### 4. Cholesky Decomposition

```python
# Create positive definite matrix
A = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])

print(f"Matrix A:\n{A}")

# Check if positive definite
eigenvals = la.eigvals(A)
is_positive_definite = np.all(eigenvals > 0)
print(f"Eigenvalues: {eigenvals}")
print(f"Is positive definite: {is_positive_definite}")

if is_positive_definite:
    # Cholesky decomposition
    L = la.cholesky(A)
    print(f"\nLower triangular L:\n{L}")
    
    # Verify decomposition
    reconstructed = L @ L.T
    print(f"\nReconstructed A:\n{reconstructed}")
    print(f"Decomposition is correct: {np.allclose(A, reconstructed)}")
```

## Eigenvalues and Eigenvectors

### 1. Basic Eigenvalue Computation

```python
A = np.array([[4, -2], [-2, 4]])

print(f"Matrix A:\n{A}")

# Compute eigenvalues and eigenvectors
eigenvals, eigenvecs = la.eig(A)
print(f"Eigenvalues: {eigenvals}")
print(f"Eigenvectors:\n{eigenvecs}")

# Verify eigenvalues and eigenvectors
for i, (val, vec) in enumerate(zip(eigenvals, eigenvecs.T)):
    result = A @ vec
    expected = val * vec
    print(f"\nEigenvalue {i+1}: {val}")
    print(f"Eigenvector {i+1}: {vec}")
    print(f"A @ v = λv: {np.allclose(result, expected)}")
```

### 2. Diagonalization

```python
A = np.array([[2, 1], [1, 2]])

print(f"Matrix A:\n{A}")

# Eigenvalue decomposition
eigenvals, eigenvecs = la.eig(A)
print(f"Eigenvalues: {eigenvals}")
print(f"Eigenvectors:\n{eigenvecs}")

# Diagonalization: A = P D P^(-1)
P = eigenvecs
D = np.diag(eigenvals)
P_inv = la.inv(P)

reconstructed = P @ D @ P_inv
print(f"\nReconstructed A:\n{reconstructed}")
print(f"Diagonalization is correct: {np.allclose(A, reconstructed)}")

# Powers using diagonalization
A_power_3 = P @ (D**3) @ P_inv
print(f"\nA^3 using diagonalization:\n{A_power_3}")
print(f"A^3 using matrix_power:\n{la.matrix_power(A, 3)}")
```

### 3. Complex Eigenvalues

```python
A = np.array([[0, -1], [1, 0]])

print(f"Matrix A:\n{A}")

# Compute eigenvalues and eigenvectors
eigenvals, eigenvecs = la.eig(A)
print(f"Eigenvalues: {eigenvals}")
print(f"Eigenvectors:\n{eigenvecs}")

# Note: Complex eigenvalues come in conjugate pairs
print(f"\nEigenvalue 1: {eigenvals[0]}")
print(f"Eigenvalue 2: {eigenvals[1]}")
print(f"Are they conjugates? {np.allclose(eigenvals[0], np.conj(eigenvals[1]))}")
```

## Vector Operations

### 1. Vector Norms

```python
v = np.array([3, 4, 5])

print(f"Vector v: {v}")

# Different norms
l1_norm = la.norm(v, ord=1)
l2_norm = la.norm(v, ord=2)
l_inf_norm = la.norm(v, ord=np.inf)

print(f"L1 norm (Manhattan): {l1_norm}")
print(f"L2 norm (Euclidean): {l2_norm}")
print(f"L∞ norm (Chebyshev): {l_inf_norm}")

# Verify L2 norm manually
l2_manual = np.sqrt(np.sum(v**2))
print(f"L2 norm (manual): {l2_manual}")
print(f"L2 norms equal: {np.allclose(l2_norm, l2_manual)}")
```

### 2. Vector Products

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"Vector a: {a}")
print(f"Vector b: {b}")

# Dot product
dot_product = np.dot(a, b)
dot_product_alt = a @ b
print(f"Dot product: {dot_product}")
print(f"Dot product (alternative): {dot_product_alt}")

# Cross product (3D only)
if len(a) == 3:
    cross_product = np.cross(a, b)
    print(f"Cross product: {cross_product}")
    
    # Verify orthogonality
    ortho_a = np.dot(a, cross_product)
    ortho_b = np.dot(b, cross_product)
    print(f"a ⊥ (a × b): {np.allclose(ortho_a, 0)}")
    print(f"b ⊥ (a × b): {np.allclose(ortho_b, 0)}")
```

### 3. Vector Projections

```python
a = np.array([1, 2, 3])
b = np.array([4, 0, 0])

print(f"Vector a: {a}")
print(f"Vector b: {b}")

# Projection of a onto b
proj_a_on_b = (np.dot(a, b) / np.dot(b, b)) * b
print(f"Projection of a onto b: {proj_a_on_b}")

# Verify projection
# The projection should be parallel to b
parallel = np.allclose(np.cross(proj_a_on_b, b), 0)
print(f"Projection is parallel to b: {parallel}")

# The difference should be perpendicular to b
perpendicular = a - proj_a_on_b
orthogonal = np.allclose(np.dot(perpendicular, b), 0)
print(f"Remainder is perpendicular to b: {orthogonal}")
```

## Matrix Norms and Condition Numbers

### 1. Matrix Norms

```python
A = np.array([[1, 2], [3, 4]])

print(f"Matrix A:\n{A}")

# Different matrix norms
frobenius_norm = la.norm(A, ord='fro')
l1_norm = la.norm(A, ord=1)
l2_norm = la.norm(A, ord=2)
l_inf_norm = la.norm(A, ord=np.inf)

print(f"Frobenius norm: {frobenius_norm}")
print(f"L1 norm (max column sum): {l1_norm}")
print(f"L2 norm (spectral norm): {l2_norm}")
print(f"L∞ norm (max row sum): {l_inf_norm}")

# Verify Frobenius norm manually
frobenius_manual = np.sqrt(np.sum(A**2))
print(f"Frobenius norm (manual): {frobenius_manual}")
print(f"Frobenius norms equal: {np.allclose(frobenius_norm, frobenius_manual)}")
```

### 2. Condition Numbers

```python
A = np.array([[1, 2], [3, 4]])

print(f"Matrix A:\n{A}")

# Condition number
cond_number = la.cond(A)
print(f"Condition number: {cond_number}")

# Condition number with different norms
cond_fro = la.cond(A, p='fro')
cond_1 = la.cond(A, p=1)
cond_2 = la.cond(A, p=2)
cond_inf = la.cond(A, p=np.inf)

print(f"Condition number (Frobenius): {cond_fro}")
print(f"Condition number (L1): {cond_1}")
print(f"Condition number (L2): {cond_2}")
print(f"Condition number (L∞): {cond_inf}")

# Well-conditioned vs ill-conditioned
print(f"\nMatrix is well-conditioned: {cond_number < 100}")
```

### 3. Matrix Rank and Null Space

```python
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

print(f"Matrix A:\n{A}")

# Rank
rank = la.matrix_rank(A)
print(f"Rank: {rank}")

# Singular values
singular_vals = la.svd(A, compute_uv=False)
print(f"Singular values: {singular_vals}")

# Null space (kernel)
# For a matrix A, null space consists of vectors x such that Ax = 0
# We can find it using SVD
U, s, Vt = la.svd(A)
# Columns of Vt corresponding to zero singular values form null space
null_space_dim = len(s) - np.sum(s > 1e-10)
print(f"Null space dimension: {null_space_dim}")
```

## Special Matrices

### 1. Creating Special Matrices

```python
# Identity matrix
I = np.eye(3)
print(f"Identity matrix:\n{I}")

# Diagonal matrix
D = np.diag([1, 2, 3, 4])
print(f"\nDiagonal matrix:\n{D}")

# Triangular matrices
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
L = np.tril(A)  # Lower triangular
U = np.triu(A)  # Upper triangular
print(f"\nOriginal matrix:\n{A}")
print(f"Lower triangular:\n{L}")
print(f"Upper triangular:\n{U}")

# Symmetric matrix
S = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
print(f"\nSymmetric matrix:\n{S}")
print(f"Is symmetric: {np.allclose(S, S.T)}")
```

### 2. Matrix Properties

```python
A = np.array([[1, 2], [3, 4]])

# Determinant
det_A = la.det(A)
print(f"Matrix A:\n{A}")
print(f"Determinant: {det_A}")

# Trace
trace_A = np.trace(A)
print(f"Trace: {trace_A}")

# Inverse
inv_A = la.inv(A)
print(f"\nInverse:\n{inv_A}")

# Verify inverse
product = A @ inv_A
print(f"A @ A^(-1):\n{product}")
print(f"Is identity: {np.allclose(product, np.eye(2))}")

# Pseudo-inverse (for non-square matrices)
A_rect = np.array([[1, 2, 3], [4, 5, 6]])
pinv_A = la.pinv(A_rect)
print(f"\nRectangular matrix:\n{A_rect}")
print(f"Pseudo-inverse:\n{pinv_A}")
```

## Applications in Data Science

### 1. Principal Component Analysis (PCA)

```python
# Generate sample data
np.random.seed(42)
data = np.random.multivariate_normal([0, 0], [[4, 2], [2, 3]], 100)

print(f"Data shape: {data.shape}")

# Center the data
data_centered = data - np.mean(data, axis=0)

# Compute covariance matrix
cov_matrix = np.cov(data_centered.T)
print(f"Covariance matrix:\n{cov_matrix}")

# Eigenvalue decomposition
eigenvals, eigenvecs = la.eig(cov_matrix)
print(f"Eigenvalues: {eigenvals}")
print(f"Eigenvectors:\n{eigenvecs}")

# Sort by eigenvalues
idx = eigenvals.argsort()[::-1]
eigenvals = eigenvals[idx]
eigenvecs = eigenvecs[:, idx]

print(f"\nSorted eigenvalues: {eigenvals}")
print(f"Principal components:\n{eigenvecs}")

# Project data onto principal components
projected = data_centered @ eigenvecs
print(f"Projected data shape: {projected.shape}")
```

### 2. Linear Regression

```python
# Generate sample data
np.random.seed(42)
X = np.random.random((100, 3))
true_coeffs = np.array([2.5, -1.0, 0.8])
noise = np.random.normal(0, 0.1, 100)
y = X @ true_coeffs + noise

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Normal equation: β = (X^T X)^(-1) X^T y
X_T = X.T
X_T_X = X_T @ X
X_T_y = X_T @ y

# Solve using matrix operations
coeffs = la.solve(X_T_X, X_T_y)
print(f"True coefficients: {true_coeffs}")
print(f"Estimated coefficients: {coeffs}")

# Predictions
y_pred = X @ coeffs
mse = np.mean((y - y_pred)**2)
print(f"Mean squared error: {mse}")
```

### 3. Image Compression with SVD

```python
# Create a simple image (matrix)
image = np.array([[255, 100, 50], [200, 150, 75], [100, 200, 125]])

print(f"Original image:\n{image}")

# SVD decomposition
U, s, Vt = la.svd(image)

# Compress by keeping only first k singular values
k = 1
U_k = U[:, :k]
s_k = s[:k]
Vt_k = Vt[:k, :]

# Reconstruct compressed image
compressed = U_k @ np.diag(s_k) @ Vt_k
print(f"\nCompressed image (k={k}):\n{compressed}")

# Compression ratio
original_size = image.size
compressed_size = U_k.size + s_k.size + Vt_k.size
compression_ratio = original_size / compressed_size
print(f"Compression ratio: {compression_ratio:.2f}x")
```

## Performance Considerations

### 1. Memory Layout

```python
import time

# C-contiguous vs F-contiguous
A_c = np.array([[1, 2, 3], [4, 5, 6]])
A_f = np.asfortranarray(A_c)

# Row-wise access (C-contiguous is faster)
start = time.time()
for i in range(10000):
    _ = A_c[0, :]
c_time = time.time() - start

start = time.time()
for i in range(10000):
    _ = A_f[0, :]
f_time = time.time() - start

print(f"C-contiguous row access: {c_time:.6f}s")
print(f"F-contiguous row access: {f_time:.6f}s")

# Column-wise access (F-contiguous is faster)
start = time.time()
for i in range(10000):
    _ = A_c[:, 0]
c_col_time = time.time() - start

start = time.time()
for i in range(10000):
    _ = A_f[:, 0]
f_col_time = time.time() - start

print(f"\nC-contiguous column access: {c_col_time:.6f}s")
print(f"F-contiguous column access: {f_col_time:.6f}s")
```

### 2. Algorithm Choice

```python
# Large matrix operations
n = 1000
A = np.random.random((n, n))
b = np.random.random(n)

# Method 1: Direct solve
start = time.time()
x1 = la.solve(A, b)
time1 = time.time() - start

# Method 2: LU decomposition then solve
start = time.time()
P, L, U = la.lu(A)
x2 = la.solve_triangular(U, la.solve_triangular(L, P @ b, lower=True), lower=False)
time2 = time.time() - start

print(f"Direct solve: {time1:.4f}s")
print(f"LU decomposition + solve: {time2:.4f}s")
print(f"Speedup: {time1/time2:.2f}x")
```

### 3. Numerical Stability

```python
# Ill-conditioned system
A = np.array([[1, 1], [1, 1.0001]])
b = np.array([2, 2.0001])

print(f"Matrix A:\n{A}")
print(f"Condition number: {la.cond(A)}")

# Solve using different methods
x_solve = la.solve(A, b)
x_lstsq = la.lstsq(A, b, rcond=None)[0]

print(f"Solution (solve): {x_solve}")
print(f"Solution (lstsq): {x_lstsq}")

# Check residuals
residual_solve = la.norm(A @ x_solve - b)
residual_lstsq = la.norm(A @ x_lstsq - b)

print(f"Residual (solve): {residual_solve}")
print(f"Residual (lstsq): {residual_lstsq}")
```

## Best Practices

### 1. Error Handling

```python
def safe_solve(A, b):
    """Safely solve linear system with error handling."""
    try:
        # Check if matrix is square
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix must be square for solve")
        
        # Check condition number
        cond = la.cond(A)
        if cond > 1e15:
            print(f"Warning: Matrix is ill-conditioned (condition number: {cond})")
        
        # Solve system
        x = la.solve(A, b)
        
        # Check residual
        residual = la.norm(A @ x - b)
        if residual > 1e-10:
            print(f"Warning: Large residual ({residual})")
        
        return x
        
    except la.LinAlgError as e:
        print(f"Linear algebra error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Test the function
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])
x = safe_solve(A, b)
print(f"Solution: {x}")
```

### 2. Memory Efficiency

```python
# Efficient matrix operations
n = 1000
A = np.random.random((n, n))
B = np.random.random((n, n))

# Avoid unnecessary copies
# Good: In-place operations
A += B  # Modifies A in-place

# Good: Use views when possible
A_view = A[:n//2, :n//2]  # Creates a view

# Good: Pre-allocate for large operations
result = np.zeros_like(A)
np.add(A, B, out=result)  # Uses pre-allocated memory
```

### 3. Numerical Accuracy

```python
# Use appropriate data types
# For high precision calculations
A_float64 = np.array([[1, 2], [3, 4]], dtype=np.float64)
A_float32 = np.array([[1, 2], [3, 4]], dtype=np.float32)

# Compare precision
det_64 = la.det(A_float64)
det_32 = la.det(A_float32)
print(f"Determinant (float64): {det_64}")
print(f"Determinant (float32): {det_32}")
print(f"Precision difference: {abs(det_64 - det_32)}")

# Use relative tolerance for comparisons
def is_close_relative(a, b, rtol=1e-10):
    return abs(a - b) <= rtol * max(abs(a), abs(b))
```

## Common Pitfalls

### 1. Matrix vs Array Operations

```python
# Pitfall: Confusing matrix and array operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise multiplication (Hadamard product)
element_wise = A * B
print(f"Element-wise multiplication:\n{element_wise}")

# Matrix multiplication
matrix_mult = A @ B
print(f"Matrix multiplication:\n{matrix_mult}")

# Don't confuse them!
print(f"Are they equal? {np.allclose(element_wise, matrix_mult)}")
```

### 2. Singular Matrices

```python
# Pitfall: Trying to invert singular matrix
A = np.array([[1, 1], [1, 1]])  # Singular matrix

try:
    inv_A = la.inv(A)
except la.LinAlgError as e:
    print(f"Error: {e}")

# Solution: Use pseudo-inverse
pinv_A = la.pinv(A)
print(f"Pseudo-inverse:\n{pinv_A}")

# Or use least squares
b = np.array([2, 2])
x = la.lstsq(A, b, rcond=None)[0]
print(f"Least squares solution: {x}")
```

### 3. Numerical Instability

```python
# Pitfall: Subtracting nearly equal numbers
a = 1.0000000001
b = 1.0000000000
result = a - b
print(f"a - b = {result}")

# Pitfall: Dividing by very small numbers
small_number = 1e-15
result = 1 / small_number
print(f"1 / {small_number} = {result}")

# Solution: Use relative comparisons
def safe_divide(a, b, threshold=1e-15):
    if abs(b) < threshold:
        raise ValueError("Division by very small number")
    return a / b
```

## Summary

This guide covered comprehensive linear algebra with NumPy:

1. **Matrix Operations**: Creation, properties, and basic operations
2. **Matrix Multiplication**: Different methods and properties
3. **Linear System Solving**: Direct methods and least squares
4. **Matrix Decompositions**: LU, QR, SVD, and Cholesky
5. **Eigenvalue Problems**: Computation and diagonalization
6. **Vector Operations**: Norms, products, and projections
7. **Matrix Norms**: Different types and condition numbers
8. **Special Matrices**: Identity, diagonal, triangular, symmetric
9. **Applications**: PCA, linear regression, image compression
10. **Performance**: Memory layout and algorithm choice
11. **Best Practices**: Error handling and numerical accuracy
12. **Common Pitfalls**: Matrix vs array operations, singular matrices

### Key Takeaways

- **NumPy.linalg** provides comprehensive linear algebra capabilities
- **Matrix decompositions** are fundamental for many algorithms
- **Numerical stability** is crucial for reliable computations
- **Performance** depends on memory layout and algorithm choice
- **Applications** span data science, machine learning, and scientific computing

### Next Steps

- Practice with real-world datasets
- Explore advanced decompositions and algorithms
- Learn about sparse matrix operations
- Study optimization and numerical methods
- Apply linear algebra to machine learning problems

### Additional Resources

- [NumPy Linear Algebra](https://numpy.org/doc/stable/reference/routines.linalg.html)
- [SciPy Linear Algebra](https://docs.scipy.org/doc/scipy/reference/linalg.html)
- [Linear Algebra Textbook](https://www.math.ucdavis.edu/~linear/)
- [Matrix Computations](https://www.cs.cornell.edu/cv/GVL4/gvl4.html)

---

**Ready to explore more advanced linear algebra? Check out the array manipulation and random generation guides!** 