#!/usr/bin/env python3
"""
NumPy Basics: Essential Array Operations

Welcome to the NumPy basics tutorial! NumPy is the fundamental package for 
scientific computing in Python, providing powerful array objects and tools 
for working with these arrays.

This script covers:
- Introduction to NumPy arrays
- Array creation methods
- Basic array operations
- Array attributes and properties
- Mathematical operations
- Broadcasting concepts

Prerequisites:
- Python 3.8 or higher
- Basic understanding of Python programming
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
    
    print("NumPy Basics Tutorial")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print("NumPy basics tutorial started successfully!")

    # Section 1: Introduction to NumPy
    print_section_header("1. Introduction to NumPy")
    
    print("""
NumPy (Numerical Python) is the fundamental package for scientific computing 
in Python. It provides:

Key Features:
- Multidimensional array objects
- Mathematical functions for arrays
- Linear algebra operations
- Random number generation
- Fourier transforms
- Tools for integrating C/C++ and Fortran code

Why NumPy?
✅ Fast array operations (vectorized)
✅ Memory efficient
✅ Broadcasting capabilities
✅ Rich mathematical functions
✅ Foundation for other scientific libraries
✅ Industry standard for numerical computing
""")

    # Section 2: Importing NumPy
    print_section_header("2. Importing NumPy")
    
    print("""
The standard way to import NumPy is with the alias 'np':
""")
    
    print("```python")
    print("import numpy as np")
    print("```")
    
    print(f"\nNumPy version: {np.__version__}")
    print(f"NumPy configuration:")
    print(f"  - BLAS: {np.__config__.show()}")

    # Section 3: Creating NumPy Arrays
    print_section_header("3. Creating NumPy Arrays")
    
    print("""
NumPy arrays are the core data structure. They can be created from:
- Python lists and tuples
- Built-in NumPy functions
- File I/O operations
- Mathematical sequences
""")

    # Demonstrate array creation
    print_subsection_header("Array Creation Examples")
    
    print("1. Creating arrays from Python lists:")
    print("```python")
    print("# 1D array from list")
    print("arr1d = np.array([1, 2, 3, 4, 5])")
    print("print(arr1d)")
    print("```")
    
    arr1d = np.array([1, 2, 3, 4, 5])
    print(f"Result: {arr1d}")
    print(f"Type: {type(arr1d)}")
    print(f"Shape: {arr1d.shape}")
    print(f"Data type: {arr1d.dtype}")

    print("\n2. Creating 2D arrays:")
    print("```python")
    print("# 2D array from nested lists")
    print("arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])")
    print("print(arr2d)")
    print("```")
    
    arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"Result:\n{arr2d}")
    print(f"Shape: {arr2d.shape}")
    print(f"Dimensions: {arr2d.ndim}")

    print("\n3. Creating arrays with specific data types:")
    print("```python")
    print("# Float array")
    print("float_arr = np.array([1, 2, 3], dtype=np.float64)")
    print("print(float_arr.dtype)")
    print("```")
    
    float_arr = np.array([1, 2, 3], dtype=np.float64)
    print(f"Result: {float_arr}")
    print(f"Data type: {float_arr.dtype}")

    # Section 4: Array Creation Functions
    print_section_header("4. Array Creation Functions")
    
    print("""
NumPy provides many functions to create arrays with specific patterns:
""")

    print_subsection_header("Zeros and Ones")
    
    print("```python")
    print("# Create array of zeros")
    print("zeros_1d = np.zeros(5)")
    print("zeros_2d = np.zeros((3, 4))")
    print("```")
    
    zeros_1d = np.zeros(5)
    zeros_2d = np.zeros((3, 4))
    print(f"1D zeros: {zeros_1d}")
    print(f"2D zeros:\n{zeros_2d}")

    print("\n```python")
    print("# Create array of ones")
    print("ones_1d = np.ones(5)")
    print("ones_2d = np.ones((2, 3))")
    print("```")
    
    ones_1d = np.ones(5)
    ones_2d = np.ones((2, 3))
    print(f"1D ones: {ones_1d}")
    print(f"2D ones:\n{ones_2d}")

    print_subsection_header("Sequential Arrays")
    
    print("```python")
    print("# Create sequence with arange")
    print("seq1 = np.arange(0, 10, 2)  # start, stop, step")
    print("seq2 = np.arange(5, 12)     # start, stop (step=1)")
    print("```")
    
    seq1 = np.arange(0, 10, 2)
    seq2 = np.arange(5, 12)
    print(f"Sequence 1 (0 to 10, step 2): {seq1}")
    print(f"Sequence 2 (5 to 12): {seq2}")

    print("\n```python")
    print("# Create sequence with linspace")
    print("lin_seq = np.linspace(0, 1, 5)  # start, stop, num_points")
    print("```")
    
    lin_seq = np.linspace(0, 1, 5)
    print(f"Linear space (0 to 1, 5 points): {lin_seq}")

    print_subsection_header("Random Arrays")
    
    print("```python")
    print("# Random integers")
    print("rand_ints = np.random.randint(1, 100, size=6)")
    print("```")
    
    np.random.seed(42)  # For reproducible results
    rand_ints = np.random.randint(1, 100, size=6)
    print(f"Random integers: {rand_ints}")

    print("\n```python")
    print("# Random floats between 0 and 1")
    print("rand_floats = np.random.random(6)")
    print("```")
    
    rand_floats = np.random.random(6)
    print(f"Random floats: {rand_floats}")

    print("\n```python")
    print("# Random floats from normal distribution")
    print("normal_floats = np.random.normal(0, 1, 6)")
    print("```")
    
    normal_floats = np.random.normal(0, 1, 6)
    print(f"Normal distribution: {normal_floats}")

    # Section 5: Array Attributes
    print_section_header("5. Array Attributes")
    
    print("""
NumPy arrays have several important attributes that describe their properties:
""")

    print_subsection_header("Basic Attributes")
    
    # Create a sample array
    sample_arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    print(f"Sample array:\n{sample_arr}")
    print(f"\nArray attributes:")
    print(f"  - Shape: {sample_arr.shape}")
    print(f"  - Dimensions: {sample_arr.ndim}")
    print(f"  - Data type: {sample_arr.dtype}")
    print(f"  - Size (total elements): {sample_arr.size}")
    print(f"  - Item size (bytes): {sample_arr.itemsize}")
    print(f"  - Total memory (bytes): {sample_arr.nbytes}")
    print(f"  - Strides: {sample_arr.strides}")

    print_subsection_header("Data Type Examples")
    
    print("```python")
    print("# Different data types")
    print("int_arr = np.array([1, 2, 3], dtype=np.int32)")
    print("float_arr = np.array([1.1, 2.2, 3.3], dtype=np.float64)")
    print("bool_arr = np.array([True, False, True], dtype=np.bool_)")
    print("```")
    
    int_arr = np.array([1, 2, 3], dtype=np.int32)
    float_arr = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    bool_arr = np.array([True, False, True], dtype=np.bool_)
    
    print(f"Integer array: {int_arr} (dtype: {int_arr.dtype})")
    print(f"Float array: {float_arr} (dtype: {float_arr.dtype})")
    print(f"Boolean array: {bool_arr} (dtype: {bool_arr.dtype})")

    # Section 6: Basic Array Operations
    print_section_header("6. Basic Array Operations")
    
    print("""
NumPy arrays support various mathematical operations that are applied 
element-wise by default.
""")

    print_subsection_header("Element-wise Operations")
    
    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7, 8])
    
    print(f"Array a: {a}")
    print(f"Array b: {b}")
    
    print(f"\nElement-wise operations:")
    print(f"  - Addition: {a + b}")
    print(f"  - Subtraction: {a - b}")
    print(f"  - Multiplication: {a * b}")
    print(f"  - Division: {a / b}")
    print(f"  - Power: {a ** 2}")
    print(f"  - Square root: {np.sqrt(a)}")

    print_subsection_header("Broadcasting")
    
    print("""
Broadcasting allows operations between arrays of different shapes:
""")
    
    arr = np.array([1, 2, 3, 4])
    scalar = 2
    
    print(f"Array: {arr}")
    print(f"Scalar: {scalar}")
    print(f"Array + scalar: {arr + scalar}")
    print(f"Array * scalar: {arr * scalar}")
    print(f"Array ** scalar: {arr ** scalar}")

    print("\nBroadcasting with different shapes:")
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    arr_1d = np.array([10, 20, 30])
    
    print(f"2D array:\n{arr_2d}")
    print(f"1D array: {arr_1d}")
    print(f"2D + 1D (broadcasted):\n{arr_2d + arr_1d}")

    # Section 7: Mathematical Functions
    print_section_header("7. Mathematical Functions")
    
    print("""
NumPy provides a comprehensive set of mathematical functions that operate 
on arrays element-wise.
""")

    print_subsection_header("Basic Mathematical Functions")
    
    arr = np.array([1, 2, 3, 4, 5])
    print(f"Original array: {arr}")
    
    print(f"\nMathematical functions:")
    print(f"  - Square: {np.square(arr)}")
    print(f"  - Square root: {np.sqrt(arr)}")
    print(f"  - Exponential: {np.exp(arr)}")
    print(f"  - Natural log: {np.log(arr)}")
    print(f"  - Sine: {np.sin(arr)}")
    print(f"  - Cosine: {np.cos(arr)}")
    print(f"  - Absolute value: {np.abs(arr)}")

    print_subsection_header("Statistical Functions")
    
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"Data: {data}")
    
    print(f"\nStatistical functions:")
    print(f"  - Mean: {np.mean(data)}")
    print(f"  - Median: {np.median(data)}")
    print(f"  - Standard deviation: {np.std(data)}")
    print(f"  - Variance: {np.var(data)}")
    print(f"  - Minimum: {np.min(data)}")
    print(f"  - Maximum: {np.max(data)}")
    print(f"  - Sum: {np.sum(data)}")
    print(f"  - Product: {np.prod(data)}")

    # Section 8: Array Indexing and Slicing
    print_section_header("8. Array Indexing and Slicing")
    
    print("""
NumPy arrays support powerful indexing and slicing operations similar to 
Python lists, but with additional capabilities.
""")

    print_subsection_header("Basic Indexing")
    
    arr = np.array([10, 20, 30, 40, 50])
    print(f"Array: {arr}")
    
    print(f"\nBasic indexing:")
    print(f"  - First element: {arr[0]}")
    print(f"  - Last element: {arr[-1]}")
    print(f"  - Third element: {arr[2]}")

    print_subsection_header("Slicing")
    
    print(f"Array: {arr}")
    print(f"\nSlicing examples:")
    print(f"  - First 3 elements: {arr[:3]}")
    print(f"  - Last 3 elements: {arr[-3:]}")
    print(f"  - Every 2nd element: {arr[::2]}")
    print(f"  - Reverse array: {arr[::-1]}")

    print_subsection_header("2D Array Indexing")
    
    arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"2D array:\n{arr_2d}")
    
    print(f"\n2D indexing:")
    print(f"  - Element at row 1, col 2: {arr_2d[1, 2]}")
    print(f"  - First row: {arr_2d[0, :]}")
    print(f"  - Second column: {arr_2d[:, 1]}")
    print(f"  - Subarray (rows 0-1, cols 1-2):\n{arr_2d[0:2, 1:3]}")

    # Section 9: Array Shape Manipulation
    print_section_header("9. Array Shape Manipulation")
    
    print("""
NumPy provides functions to change the shape and structure of arrays 
without changing their data.
""")

    print_subsection_header("Reshaping Arrays")
    
    arr = np.arange(12)
    print(f"Original array: {arr}")
    print(f"Shape: {arr.shape}")
    
    print(f"\nReshaping examples:")
    reshaped_2d = arr.reshape(3, 4)
    print(f"  - Reshaped to 3x4:\n{reshaped_2d}")
    
    reshaped_3d = arr.reshape(2, 3, 2)
    print(f"  - Reshaped to 2x3x2:\n{reshaped_3d}")
    
    flattened = reshaped_2d.flatten()
    print(f"  - Flattened: {flattened}")

    print_subsection_header("Transposing Arrays")
    
    arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"Original array:\n{arr_2d}")
    print(f"Shape: {arr_2d.shape}")
    
    transposed = arr_2d.T
    print(f"\nTransposed:\n{transposed}")
    print(f"Shape: {transposed.shape}")

    # Section 10: Practical Examples
    print_section_header("10. Practical Examples")
    
    print("""
Let's apply NumPy concepts to solve practical problems:
""")

    print_subsection_header("Example 1: Linear Dataset Generation")
    
    print("Creating a linear dataset with noise:")
    print("```python")
    print("# Generate feature values")
    print("feature = np.arange(6, 21)")
    print("")
    print("# Generate labels using linear equation: y = 3x + 4")
    print("label = 3 * feature + 4")
    print("")
    print("# Add noise")
    print("noise = np.random.normal(0, 1, len(feature))")
    print("label_with_noise = label + noise")
    print("```")
    
    # Generate the dataset
    feature = np.arange(6, 21)
    label = 3 * feature + 4
    noise = np.random.normal(0, 1, len(feature))
    label_with_noise = label + noise
    
    print(f"\nResults:")
    print(f"  Features: {feature}")
    print(f"  Labels (clean): {label}")
    print(f"  Noise: {noise}")
    print(f"  Labels (with noise): {label_with_noise}")

    print_subsection_header("Example 2: Statistical Analysis")
    
    print("Analyzing a dataset:")
    print("```python")
    print("# Generate sample data")
    print("data = np.random.normal(100, 15, 1000)")
    print("")
    print("# Calculate statistics")
    print("mean_val = np.mean(data)")
    print("std_val = np.std(data)")
    print("percentiles = np.percentile(data, [25, 50, 75])")
    print("```")
    
    # Generate and analyze data
    data = np.random.normal(100, 15, 1000)
    mean_val = np.mean(data)
    std_val = np.std(data)
    percentiles = np.percentile(data, [25, 50, 75])
    
    print(f"\nResults:")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Standard deviation: {std_val:.2f}")
    print(f"  25th percentile: {percentiles[0]:.2f}")
    print(f"  50th percentile (median): {percentiles[1]:.2f}")
    print(f"  75th percentile: {percentiles[2]:.2f}")

    print_subsection_header("Example 3: Matrix Operations")
    
    print("Performing matrix operations:")
    print("```python")
    print("# Create matrices")
    print("A = np.array([[1, 2], [3, 4]])")
    print("B = np.array([[5, 6], [7, 8]])")
    print("")
    print("# Matrix operations")
    print("C = A + B  # Element-wise addition")
    print("D = A * B  # Element-wise multiplication")
    print("E = A @ B  # Matrix multiplication")
    print("```")
    
    # Matrix operations
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    C = A + B
    D = A * B
    E = A @ B
    
    print(f"\nResults:")
    print(f"  Matrix A:\n{A}")
    print(f"  Matrix B:\n{B}")
    print(f"  A + B (element-wise):\n{C}")
    print(f"  A * B (element-wise):\n{D}")
    print(f"  A @ B (matrix multiplication):\n{E}")

    # Section 11: Summary and Next Steps
    print_section_header("11. Summary and Next Steps")
    
    print("""
Congratulations! You've completed the NumPy basics tutorial. Here's what you've learned:

Key Concepts Covered:
✅ Array Creation: Various methods to create NumPy arrays
✅ Array Attributes: Understanding array properties and metadata
✅ Basic Operations: Element-wise operations and broadcasting
✅ Mathematical Functions: Built-in mathematical and statistical functions
✅ Indexing and Slicing: Accessing and modifying array elements
✅ Shape Manipulation: Reshaping and restructuring arrays
✅ Practical Applications: Real-world examples and use cases

Next Steps:

1. Practice with Arrays: Experiment with different array creation methods
2. Explore Advanced Indexing: Learn boolean indexing and fancy indexing
3. Master Broadcasting: Understand broadcasting rules thoroughly
4. Study Linear Algebra: Learn matrix operations and decompositions
5. Work with Real Data: Apply NumPy to actual datasets
6. Explore Other Libraries: Use NumPy with pandas, matplotlib, and scikit-learn

Additional Resources:
- NumPy Official Documentation: https://numpy.org/doc/
- NumPy User Guide: https://numpy.org/doc/stable/user/index.html
- NumPy Reference: https://numpy.org/doc/stable/reference/
- NumPy Tutorial: https://numpy.org/doc/stable/user/quickstart.html

Practice Exercises:
1. Create arrays of different shapes and data types
2. Perform mathematical operations on arrays
3. Use broadcasting to work with arrays of different shapes
4. Apply statistical functions to analyze data
5. Reshape and manipulate array structures
6. Build a simple data analysis pipeline

Happy NumPy-ing! 🚀
""")

if __name__ == "__main__":
    # Run the tutorial
    main()
    
    print("\n" + "="*60)
    print(" Tutorial completed successfully!")
    print(" Practice with NumPy arrays!")
    print("="*60) 