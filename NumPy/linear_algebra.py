#!/usr/bin/env python3
"""
NumPy Linear Algebra: Matrix Operations and Decompositions

Welcome to the NumPy linear algebra tutorial! This tutorial covers 
essential linear algebra operations using NumPy, including matrix 
operations, decompositions, and solving linear systems.

This script covers:
- Matrix creation and basic operations
- Matrix multiplication and properties
- Linear system solving
- Matrix decompositions (LU, QR, SVD, Cholesky)
- Eigenvalues and eigenvectors
- Vector operations and norms
- Applications in data science and machine learning

Prerequisites:
- Python 3.8 or higher
- Basic understanding of NumPy (covered in numpy_basics.py)
- Basic linear algebra concepts
- NumPy installed (pip install numpy)
"""

import numpy as np
import sys

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_subsection_header(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")

def main():
    """Main function to run all tutorial sections."""
    
    print("NumPy Linear Algebra Tutorial")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print("Linear algebra tutorial started successfully!")

    # Section 1: Introduction to Linear Algebra with NumPy
    print_section_header("1. Introduction to Linear Algebra with NumPy")
    
    print("""
Linear algebra is fundamental to many areas of data science, machine learning, 
and scientific computing. NumPy provides a comprehensive set of linear algebra 
functions through the numpy.linalg module.

Key Linear Algebra Operations:
- Matrix creation and manipulation
- Matrix multiplication and properties
- Solving linear systems Ax = b
- Matrix decompositions (LU, QR, SVD, Cholesky)
- Eigenvalue and eigenvector computation
- Vector operations and norms
- Determinants and matrix inverses

Applications:
✅ Machine Learning: Feature transformations, dimensionality reduction
✅ Data Science: Principal Component Analysis (PCA), regression
✅ Computer Graphics: Transformations, rotations, scaling
✅ Signal Processing: Filtering, Fourier transforms
✅ Optimization: Constraint handling, quadratic programming
✅ Statistics: Multivariate analysis, correlation matrices
""")

    # Section 2: Matrix Creation and Basic Operations
    print_section_header("2. Matrix Creation and Basic Operations")
    
    print("""
Matrices in NumPy are 2D arrays. Let's explore different ways to create 
and manipulate matrices.
""")

    print_subsection_header("Creating Matrices")
    
    print("1. Creating matrices from lists:")
    print("```python")
    print("A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])")
    print("```")
    
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"Matrix A:\n{A}")
    print(f"Shape: {A.shape}")
    print(f"Data type: {A.dtype}")

    print("\n2. Creating special matrices:")
    print("```python")
    print("identity = np.eye(3)  # Identity matrix")
    print("zeros = np.zeros((3, 3))  # Zero matrix")
    print("ones = np.ones((3, 3))  # Matrix of ones")
    print("```")
    
    identity = np.eye(3)
    zeros = np.zeros((3, 3))
    ones = np.ones((3, 3))
    
    print(f"Identity matrix:\n{identity}")
    print(f"Zero matrix:\n{zeros}")
    print(f"Ones matrix:\n{ones}")

    print("\n3. Creating random matrices:")
    print("```python")
    print("random_matrix = np.random.rand(3, 3)")
    print("normal_matrix = np.random.normal(0, 1, (3, 3))")
    print("```")
    
    np.random.seed(42)  # For reproducible results
    random_matrix = np.random.rand(3, 3)
    normal_matrix = np.random.normal(0, 1, (3, 3))
    
    print(f"Random matrix (uniform):\n{random_matrix}")
    print(f"Normal matrix:\n{normal_matrix}")

    print_subsection_header("Matrix Properties")
    
    print("Matrix properties and attributes:")
    print(f"Matrix A:\n{A}")
    print(f"  - Shape: {A.shape}")
    print(f"  - Dimensions: {A.ndim}")
    print(f"  - Size: {A.size}")
    print(f"  - Data type: {A.dtype}")
    print(f"  - Transpose:\n{A.T}")

    # Section 3: Matrix Operations
    print_section_header("3. Matrix Operations")
    
    print("""
NumPy provides various matrix operations including addition, multiplication, 
and element-wise operations.
""")

    print_subsection_header("Basic Matrix Operations")
    
    B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    
    print(f"\nElement-wise operations:")
    print(f"  - Addition:\n{A + B}")
    print(f"  - Subtraction:\n{A - B}")
    print(f"  - Element-wise multiplication:\n{A * B}")
    print(f"  - Element-wise division:\n{A / B}")

    print_subsection_header("Matrix Multiplication")
    
    print("Matrix multiplication (dot product):")
    print("```python")
    print("C = np.dot(A, B)  # or C = A @ B")
    print("```")
    
    C = np.dot(A, B)
    C_alt = A @ B  # Alternative syntax
    
    print(f"Matrix multiplication result:\n{C}")
    print(f"Using @ operator:\n{C_alt}")
    print(f"Results are equal: {np.array_equal(C, C_alt)}")

    print("\nMatrix-vector multiplication:")
    v = np.array([1, 2, 3])
    print(f"Vector v: {v}")
    result = A @ v
    print(f"A @ v = {result}")

    print_subsection_header("Matrix Properties")
    
    print("Matrix properties:")
    print(f"  - Determinant of A: {np.linalg.det(A):.2f}")
    print(f"  - Trace of A: {np.trace(A)}")
    print(f"  - Rank of A: {np.linalg.matrix_rank(A)}")
    
    # Check if matrix is invertible
    det_A = np.linalg.det(A)
    if abs(det_A) > 1e-10:
        print(f"  - Matrix A is invertible (det ≠ 0)")
        A_inv = np.linalg.inv(A)
        print(f"  - Inverse of A:\n{A_inv}")
    else:
        print(f"  - Matrix A is not invertible (det = 0)")

    # Section 4: Linear System Solving
    print_section_header("4. Linear System Solving")
    
    print("""
Solving linear systems Ax = b is one of the most common applications 
of linear algebra. NumPy provides efficient methods for this.
""")

    print_subsection_header("Basic Linear System Solving")
    
    # Create a system Ax = b
    A_system = np.array([[2, 1, 1], [1, 3, 2], [1, 0, 0]])
    b = np.array([4, 5, 6])
    
    print(f"System Ax = b:")
    print(f"Matrix A:\n{A_system}")
    print(f"Vector b: {b}")
    
    # Solve using numpy.linalg.solve
    x = np.linalg.solve(A_system, b)
    print(f"Solution x: {x}")
    
    # Verify solution
    verification = A_system @ x
    print(f"Verification (A @ x): {verification}")
    print(f"Original b: {b}")
    print(f"Solution is correct: {np.allclose(verification, b)}")

    print_subsection_header("Overdetermined and Underdetermined Systems")
    
    # Overdetermined system (more equations than unknowns)
    A_over = np.array([[1, 2], [3, 4], [5, 6]])
    b_over = np.array([1, 2, 3])
    
    print(f"Overdetermined system:")
    print(f"Matrix A:\n{A_over}")
    print(f"Vector b: {b_over}")
    
    # Use least squares solution
    x_lstsq = np.linalg.lstsq(A_over, b_over, rcond=None)[0]
    print(f"Least squares solution: {x_lstsq}")
    
    # Check residual
    residual = A_over @ x_lstsq - b_over
    print(f"Residual: {residual}")
    print(f"Residual norm: {np.linalg.norm(residual):.6f}")

    # Section 5: Matrix Decompositions
    print_section_header("5. Matrix Decompositions")
    
    print("""
Matrix decompositions are fundamental tools in linear algebra that break 
down matrices into simpler, more manageable forms.
""")

    print_subsection_header("LU Decomposition")
    
    print("LU decomposition: A = LU, where L is lower triangular and U is upper triangular")
    
    # Create a matrix for LU decomposition
    A_lu = np.array([[2, 1, 1], [4, -6, 0], [-2, 7, 2]])
    print(f"Matrix A:\n{A_lu}")
    
    # Perform LU decomposition
    P, L, U = scipy.linalg.lu(A_lu)
    print(f"Permutation matrix P:\n{P}")
    print(f"Lower triangular L:\n{L}")
    print(f"Upper triangular U:\n{U}")
    
    # Verify decomposition
    reconstructed = P @ L @ U
    print(f"Reconstructed A (P @ L @ U):\n{reconstructed}")
    print(f"Decomposition is correct: {np.allclose(A_lu, reconstructed)}")

    print_subsection_header("QR Decomposition")
    
    print("QR decomposition: A = QR, where Q is orthogonal and R is upper triangular")
    
    # Create a matrix for QR decomposition
    A_qr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"Matrix A:\n{A_qr}")
    
    # Perform QR decomposition
    Q, R = np.linalg.qr(A_qr)
    print(f"Orthogonal matrix Q:\n{Q}")
    print(f"Upper triangular R:\n{R}")
    
    # Verify decomposition
    reconstructed_qr = Q @ R
    print(f"Reconstructed A (Q @ R):\n{reconstructed_qr}")
    print(f"Decomposition is correct: {np.allclose(A_qr, reconstructed_qr)}")
    
    # Check orthogonality of Q
    Q_orthogonal = Q.T @ Q
    print(f"Q^T @ Q (should be identity):\n{Q_orthogonal}")

    print_subsection_header("Singular Value Decomposition (SVD)")
    
    print("SVD decomposition: A = UΣV^T, where U and V are orthogonal, Σ is diagonal")
    
    # Create a matrix for SVD
    A_svd = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"Matrix A:\n{A_svd}")
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(A_svd)
    print(f"U matrix:\n{U}")
    print(f"Singular values S: {S}")
    print(f"V^T matrix:\n{Vt}")
    
    # Reconstruct matrix
    S_matrix = np.zeros_like(A_svd, dtype=float)
    S_matrix[:len(S), :len(S)] = np.diag(S)
    reconstructed_svd = U @ S_matrix @ Vt
    print(f"Reconstructed A (U @ Σ @ V^T):\n{reconstructed_svd}")
    print(f"Decomposition is correct: {np.allclose(A_svd, reconstructed_svd)}")

    print_subsection_header("Cholesky Decomposition")
    
    print("Cholesky decomposition: A = LL^T, where L is lower triangular (for positive definite matrices)")
    
    # Create a positive definite matrix
    A_chol = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
    print(f"Matrix A:\n{A_chol}")
    
    # Check if positive definite
    eigenvals = np.linalg.eigvals(A_chol)
    print(f"Eigenvalues: {eigenvals}")
    print(f"Matrix is positive definite: {np.all(eigenvals > 0)}")
    
    # Perform Cholesky decomposition
    L = np.linalg.cholesky(A_chol)
    print(f"Lower triangular L:\n{L}")
    
    # Verify decomposition
    reconstructed_chol = L @ L.T
    print(f"Reconstructed A (L @ L^T):\n{reconstructed_chol}")
    print(f"Decomposition is correct: {np.allclose(A_chol, reconstructed_chol)}")

    # Section 6: Eigenvalues and Eigenvectors
    print_section_header("6. Eigenvalues and Eigenvectors")
    
    print("""
Eigenvalues and eigenvectors are fundamental concepts in linear algebra 
with applications in many fields including machine learning and physics.
""")

    print_subsection_header("Computing Eigenvalues and Eigenvectors")
    
    # Create a symmetric matrix
    A_eig = np.array([[4, -2], [-2, 4]])
    print(f"Matrix A:\n{A_eig}")
    
    # Compute eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(A_eig)
    print(f"Eigenvalues: {eigenvals}")
    print(f"Eigenvectors:\n{eigenvecs}")
    
    # Verify eigenvector property: Av = λv
    for i in range(len(eigenvals)):
        lambda_val = eigenvals[i]
        v = eigenvecs[:, i]
        Av = A_eig @ v
        lambda_v = lambda_val * v
        print(f"Eigenvalue {i+1}: λ = {lambda_val:.3f}")
        print(f"  Av = {Av}")
        print(f"  λv = {lambda_v}")
        print(f"  Av = λv: {np.allclose(Av, lambda_v)}")

    print_subsection_header("Eigendecomposition")
    
    print("Eigendecomposition: A = VΛV^(-1), where V contains eigenvectors and Λ is diagonal")
    
    # Reconstruct matrix from eigendecomposition
    V = eigenvecs
    Lambda = np.diag(eigenvals)
    V_inv = np.linalg.inv(V)
    
    reconstructed_eig = V @ Lambda @ V_inv
    print(f"Reconstructed A (V @ Λ @ V^(-1)):\n{reconstructed_eig}")
    print(f"Decomposition is correct: {np.allclose(A_eig, reconstructed_eig)}")

    print_subsection_header("Power Iteration Method")
    
    print("Power iteration method for finding the dominant eigenvalue and eigenvector")
    
    def power_iteration(A, max_iter=100, tol=1e-6):
        """Power iteration method for finding dominant eigenvalue and eigenvector."""
        n = A.shape[0]
        v = np.random.rand(n)
        v = v / np.linalg.norm(v)
        
        for i in range(max_iter):
            v_old = v.copy()
            v_new = A @ v
            v = v_new / np.linalg.norm(v_new)
            
            if np.linalg.norm(v - v_old) < tol:
                break
        
        # Compute eigenvalue
        eigenvalue = (v.T @ A @ v) / (v.T @ v)
        return eigenvalue, v
    
    # Test power iteration
    dominant_eigenval, dominant_eigenvec = power_iteration(A_eig)
    print(f"Dominant eigenvalue (power iteration): {dominant_eigenval:.6f}")
    print(f"Exact dominant eigenvalue: {np.max(eigenvals):.6f}")
    print(f"Error: {abs(dominant_eigenval - np.max(eigenvals)):.2e}")

    # Section 7: Vector Operations and Norms
    print_section_header("7. Vector Operations and Norms")
    
    print("""
Vector operations and norms are essential for understanding distances, 
similarities, and geometric properties in linear algebra.
""")

    print_subsection_header("Vector Operations")
    
    # Create vectors
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    print(f"Vector v1: {v1}")
    print(f"Vector v2: {v2}")
    
    print(f"\nVector operations:")
    print(f"  - Addition: {v1 + v2}")
    print(f"  - Subtraction: {v1 - v2}")
    print(f"  - Element-wise multiplication: {v1 * v2}")
    print(f"  - Dot product: {np.dot(v1, v2)}")
    print(f"  - Cross product: {np.cross(v1, v2)}")

    print_subsection_header("Vector Norms")
    
    print("Different types of vector norms:")
    
    # L1 norm (Manhattan distance)
    l1_norm = np.linalg.norm(v1, ord=1)
    print(f"  - L1 norm (Manhattan): {l1_norm}")
    
    # L2 norm (Euclidean distance)
    l2_norm = np.linalg.norm(v1, ord=2)
    print(f"  - L2 norm (Euclidean): {l2_norm}")
    
    # L-infinity norm (maximum absolute value)
    linf_norm = np.linalg.norm(v1, ord=np.inf)
    print(f"  - L-infinity norm: {linf_norm}")
    
    # Normalized vector
    v1_normalized = v1 / l2_norm
    print(f"  - Normalized v1: {v1_normalized}")
    print(f"  - Norm of normalized vector: {np.linalg.norm(v1_normalized):.6f}")

    print_subsection_header("Distance and Similarity")
    
    print("Distance and similarity measures between vectors:")
    
    # Euclidean distance
    euclidean_dist = np.linalg.norm(v1 - v2)
    print(f"  - Euclidean distance: {euclidean_dist}")
    
    # Cosine similarity
    cos_similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    print(f"  - Cosine similarity: {cos_similarity}")
    
    # Cosine distance
    cos_distance = 1 - cos_similarity
    print(f"  - Cosine distance: {cos_distance}")

    # Section 8: Applications in Data Science
    print_section_header("8. Applications in Data Science")
    
    print("""
Linear algebra is fundamental to many data science and machine learning 
techniques. Let's explore some practical applications.
""")

    print_subsection_header("Principal Component Analysis (PCA)")
    
    print("PCA is a dimensionality reduction technique based on eigendecomposition")
    
    # Create sample data
    np.random.seed(42)
    data = np.random.multivariate_normal([0, 0], [[4, 2], [2, 3]], 100)
    print(f"Data shape: {data.shape}")
    print(f"First 5 data points:\n{data[:5]}")
    
    # Center the data
    data_centered = data - np.mean(data, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(data_centered.T)
    print(f"Covariance matrix:\n{cov_matrix}")
    
    # Compute eigenvalues and eigenvectors
    eigenvals_pca, eigenvecs_pca = np.linalg.eigh(cov_matrix)
    
    # Sort in descending order
    idx = eigenvals_pca.argsort()[::-1]
    eigenvals_pca = eigenvals_pca[idx]
    eigenvecs_pca = eigenvecs_pca[:, idx]
    
    print(f"Eigenvalues: {eigenvals_pca}")
    print(f"Eigenvectors:\n{eigenvecs_pca}")
    
    # Project data onto principal components
    projected_data = data_centered @ eigenvecs_pca
    print(f"Projected data shape: {projected_data.shape}")
    print(f"First 5 projected points:\n{projected_data[:5]}")

    print_subsection_header("Linear Regression")
    
    print("Linear regression using normal equations")
    
    # Create synthetic data
    X = np.random.rand(50, 3)  # Features
    true_weights = np.array([2, -1, 3])
    y = X @ true_weights + np.random.normal(0, 0.1, 50)  # Target with noise
    
    print(f"Feature matrix X shape: {X.shape}")
    print(f"Target vector y shape: {y.shape}")
    print(f"True weights: {true_weights}")
    
    # Solve using normal equations: w = (X^T X)^(-1) X^T y
    X_T = X.T
    X_T_X = X_T @ X
    X_T_y = X_T @ y
    
    # Check if X^T X is invertible
    if np.linalg.det(X_T_X) > 1e-10:
        weights = np.linalg.solve(X_T_X, X_T_y)
        print(f"Estimated weights: {weights}")
        print(f"True weights: {true_weights}")
        print(f"Mean squared error: {np.mean((weights - true_weights)**2):.6f}")
    else:
        print("X^T X is not invertible, using pseudoinverse")
        weights = np.linalg.pinv(X) @ y
        print(f"Estimated weights (pseudoinverse): {weights}")

    print_subsection_header("Matrix Factorization")
    
    print("Simple matrix factorization example")
    
    # Create a matrix to factorize
    M = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"Original matrix M:\n{M}")
    
    # Use SVD for factorization
    U, S, Vt = np.linalg.svd(M)
    
    # Keep only first 2 singular values (rank-2 approximation)
    U_approx = U[:, :2]
    S_approx = S[:2]
    Vt_approx = Vt[:2, :]
    
    # Reconstruct approximation
    M_approx = U_approx @ np.diag(S_approx) @ Vt_approx
    print(f"Rank-2 approximation:\n{M_approx}")
    
    # Compute reconstruction error
    error = np.linalg.norm(M - M_approx, 'fro')
    print(f"Reconstruction error: {error:.6f}")

    # Section 9: Performance and Numerical Considerations
    print_section_header("9. Performance and Numerical Considerations")
    
    print("""
Understanding numerical stability and performance considerations 
is crucial for reliable linear algebra computations.
""")

    print_subsection_header("Condition Number")
    
    print("Condition number measures how sensitive a matrix is to perturbations")
    
    # Create matrices with different condition numbers
    A_well_conditioned = np.array([[1, 0], [0, 1]])
    A_ill_conditioned = np.array([[1, 1], [1, 1.0001]])
    
    cond_well = np.linalg.cond(A_well_conditioned)
    cond_ill = np.linalg.cond(A_ill_conditioned)
    
    print(f"Well-conditioned matrix:\n{A_well_conditioned}")
    print(f"Condition number: {cond_well:.2f}")
    
    print(f"Ill-conditioned matrix:\n{A_ill_conditioned}")
    print(f"Condition number: {cond_ill:.2e}")

    print_subsection_header("Numerical Stability")
    
    print("Demonstrating numerical stability issues")
    
    # Create a nearly singular matrix
    A_near_singular = np.array([[1, 1], [1, 1 + 1e-15]])
    print(f"Nearly singular matrix:\n{A_near_singular}")
    
    # Try to compute inverse
    try:
        A_inv = np.linalg.inv(A_near_singular)
        print(f"Inverse:\n{A_inv}")
    except np.linalg.LinAlgError:
        print("Matrix is singular or nearly singular")
    
    # Use pseudoinverse instead
    A_pinv = np.linalg.pinv(A_near_singular)
    print(f"Pseudoinverse:\n{A_pinv}")

    print_subsection_header("Performance Comparison")
    
    print("Comparing different methods for solving linear systems")
    
    # Create large system
    n = 100
    A_large = np.random.rand(n, n)
    b_large = np.random.rand(n)
    
    import time
    
    # Method 1: Direct solve
    start_time = time.time()
    x1 = np.linalg.solve(A_large, b_large)
    time1 = time.time() - start_time
    
    # Method 2: Inverse then multiply
    start_time = time.time()
    A_inv = np.linalg.inv(A_large)
    x2 = A_inv @ b_large
    time2 = time.time() - start_time
    
    print(f"System size: {n}x{n}")
    print(f"Direct solve time: {time1:.6f} seconds")
    print(f"Inverse + multiply time: {time2:.6f} seconds")
    print(f"Speedup: {time2/time1:.2f}x")
    print(f"Solutions are equal: {np.allclose(x1, x2)}")

    # Section 10: Summary and Next Steps
    print_section_header("10. Summary and Next Steps")
    
    print("""
Congratulations! You've completed the NumPy linear algebra tutorial. Here's what you've learned:

Key Concepts Covered:
✅ Matrix Operations: Creation, manipulation, and properties
✅ Linear System Solving: Direct methods and least squares
✅ Matrix Decompositions: LU, QR, SVD, and Cholesky
✅ Eigenvalues and Eigenvectors: Computation and applications
✅ Vector Operations: Norms, distances, and similarities
✅ Data Science Applications: PCA, regression, factorization
✅ Numerical Considerations: Stability and performance

Next Steps:

1. Practice Matrix Operations: Work with different matrix types and sizes
2. Master Decompositions: Understand when and how to use each decomposition
3. Explore Applications: Apply linear algebra to real-world problems
4. Study Advanced Topics: Learn about sparse matrices and iterative methods
5. Performance Optimization: Profile and optimize your linear algebra code
6. Explore Related Libraries: Learn about scipy.linalg and specialized libraries

Additional Resources:
- NumPy Linear Algebra: https://numpy.org/doc/stable/reference/routines.linalg.html
- SciPy Linear Algebra: https://docs.scipy.org/doc/scipy/reference/linalg.html
- Linear Algebra Textbook: Gilbert Strang's "Introduction to Linear Algebra"
- Online Courses: MIT OpenCourseWare Linear Algebra

Practice Exercises:
1. Implement matrix operations from scratch
2. Solve various types of linear systems
3. Apply matrix decompositions to data analysis
4. Build a simple PCA implementation
5. Compare performance of different algorithms
6. Work with real-world datasets

Happy Linear Algebra Computing! 🚀
""")

if __name__ == "__main__":
    # Import scipy for some advanced decompositions
    try:
        import scipy.linalg
    except ImportError:
        print("Warning: scipy not available. Some examples may not work.")
        print("Install with: pip install scipy")
    
    # Run the tutorial
    main()
    
    print("\n" + "="*60)
    print(" Tutorial completed successfully!")
    print(" Master NumPy linear algebra!")
    print("="*60) 