#!/usr/bin/env python3
"""
NumPy Array Manipulation: Reshaping, Concatenation, and Advanced Operations

Welcome to the NumPy array manipulation tutorial! This tutorial covers 
comprehensive array manipulation techniques including reshaping, 
concatenation, splitting, and advanced array operations.

This script covers:
- Array reshaping and transposition
- Concatenation and stacking
- Splitting and chunking arrays
- Broadcasting and array alignment
- Advanced indexing and manipulation
- Memory layout and performance
- Practical applications and examples

Prerequisites:
- Python 3.8 or higher
- Basic understanding of NumPy (covered in numpy_basics.py)
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
    
    print("NumPy Array Manipulation Tutorial")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print("Array manipulation tutorial started successfully!")

    # Section 1: Introduction to Array Manipulation
    print_section_header("1. Introduction to Array Manipulation")
    
    print("""
Array manipulation is a core skill in NumPy that allows you to transform, 
combine, and restructure arrays for various computational tasks.

Key Manipulation Operations:
- Reshaping: Change array dimensions and structure
- Concatenation: Combine arrays along different axes
- Splitting: Divide arrays into smaller parts
- Transposition: Rearrange array dimensions
- Broadcasting: Align arrays for operations
- Advanced indexing: Complex array access patterns

Applications:
✅ Data preprocessing and cleaning
✅ Feature engineering for machine learning
✅ Image and signal processing
✅ Scientific computing and simulations
✅ Data analysis and visualization
✅ Performance optimization
""")

    # Section 2: Array Reshaping
    print_section_header("2. Array Reshaping")
    
    print("""
Reshaping allows you to change the dimensions of an array while 
preserving the total number of elements.
""")

    print_subsection_header("Basic Reshaping")
    
    # Create a 1D array
    arr_1d = np.arange(12)
    print(f"Original 1D array: {arr_1d}")
    print(f"Shape: {arr_1d.shape}")
    
    # Reshape to 2D
    arr_2d = arr_1d.reshape(3, 4)
    print(f"\nReshaped to 2D (3x4):\n{arr_2d}")
    print(f"Shape: {arr_2d.shape}")
    
    # Reshape to 3D
    arr_3d = arr_1d.reshape(2, 3, 2)
    print(f"\nReshaped to 3D (2x3x2):\n{arr_3d}")
    print(f"Shape: {arr_3d.shape}")

    print_subsection_header("Reshape with -1")
    
    print("Using -1 to automatically calculate one dimension:")
    
    # Reshape with automatic dimension calculation
    arr_auto_1 = arr_1d.reshape(-1, 4)  # Calculate rows automatically
    print(f"Reshape with -1, 4: {arr_auto_1.shape}")
    print(f"Result:\n{arr_auto_1}")
    
    arr_auto_2 = arr_1d.reshape(3, -1)  # Calculate columns automatically
    print(f"\nReshape with 3, -1: {arr_auto_2.shape}")
    print(f"Result:\n{arr_auto_2}")
    
    arr_auto_3 = arr_1d.reshape(-1)  # Flatten to 1D
    print(f"\nReshape with -1: {arr_auto_3.shape}")
    print(f"Result: {arr_auto_3}")

    print_subsection_header("Reshape vs Resize")
    
    print("Difference between reshape and resize:")
    
    # Reshape (creates new array)
    arr_original = np.array([1, 2, 3, 4])
    arr_reshaped = arr_original.reshape(2, 2)
    print(f"Original: {arr_original}")
    print(f"Reshaped:\n{arr_reshaped}")
    print(f"Original unchanged: {arr_original}")
    
    # Resize (modifies original array)
    arr_to_resize = np.array([1, 2, 3, 4])
    arr_to_resize.resize(2, 2)
    print(f"\nAfter resize:\n{arr_to_resize}")

    print_subsection_header("Flattening Arrays")
    
    print("Different ways to flatten arrays:")
    
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

    # Section 3: Array Transposition
    print_section_header("3. Array Transposition")
    
    print("""
Transposition rearranges the dimensions of an array, which is 
particularly useful for matrix operations and data alignment.
""")

    print_subsection_header("Basic Transposition")
    
    # Create a 2D array
    arr_transpose = np.array([[1, 2, 3], [4, 5, 6]])
    print(f"Original array:\n{arr_transpose}")
    print(f"Shape: {arr_transpose.shape}")
    
    # Transpose
    arr_T = arr_transpose.T
    print(f"\nTransposed array:\n{arr_T}")
    print(f"Shape: {arr_T.shape}")
    
    # Using transpose function
    arr_transposed = np.transpose(arr_transpose)
    print(f"\nUsing transpose():\n{arr_transposed}")

    print_subsection_header("Multi-dimensional Transposition")
    
    # Create a 3D array
    arr_3d_transpose = np.arange(24).reshape(2, 3, 4)
    print(f"3D array shape: {arr_3d_transpose.shape}")
    print(f"3D array:\n{arr_3d_transpose}")
    
    # Transpose with custom axis order
    arr_transposed_3d = np.transpose(arr_3d_transpose, (1, 0, 2))
    print(f"\nTransposed (1, 0, 2) shape: {arr_transposed_3d.shape}")
    print(f"Transposed array:\n{arr_transposed_3d}")

    print_subsection_header("Matrix Operations with Transposition")
    
    # Matrix operations
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    
    # Matrix multiplication properties
    AB = A @ B
    AB_T = (A @ B).T
    B_T_A_T = B.T @ A.T
    
    print(f"\n(A @ B):\n{AB}")
    print(f"(A @ B)^T:\n{AB_T}")
    print(f"B^T @ A^T:\n{B_T_A_T}")
    print(f"(AB)^T = B^T A^T: {np.array_equal(AB_T, B_T_A_T)}")

    # Section 4: Array Concatenation
    print_section_header("4. Array Concatenation")
    
    print("""
Concatenation allows you to combine arrays along specified axes 
to create larger arrays.
""")

    print_subsection_header("Basic Concatenation")
    
    # Create arrays to concatenate
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    
    print(f"Array 1: {arr1}")
    print(f"Array 2: {arr2}")
    
    # Concatenate along axis 0 (default)
    concatenated = np.concatenate([arr1, arr2])
    print(f"Concatenated: {concatenated}")
    
    # Concatenate with axis specification
    concatenated_axis0 = np.concatenate([arr1, arr2], axis=0)
    print(f"Concatenated axis=0: {concatenated_axis0}")

    print_subsection_header("2D Array Concatenation")
    
    # Create 2D arrays
    arr_2d_1 = np.array([[1, 2], [3, 4]])
    arr_2d_2 = np.array([[5, 6], [7, 8]])
    
    print(f"2D Array 1:\n{arr_2d_1}")
    print(f"2D Array 2:\n{arr_2d_2}")
    
    # Concatenate along rows (axis=0)
    concatenated_rows = np.concatenate([arr_2d_1, arr_2d_2], axis=0)
    print(f"Concatenated along rows:\n{concatenated_rows}")
    
    # Concatenate along columns (axis=1)
    concatenated_cols = np.concatenate([arr_2d_1, arr_2d_2], axis=1)
    print(f"Concatenated along columns:\n{concatenated_cols}")

    print_subsection_header("Stacking Functions")
    
    print("NumPy provides convenient stacking functions:")
    
    # Vertical stacking (equivalent to concatenate axis=0)
    vstacked = np.vstack([arr_2d_1, arr_2d_2])
    print(f"Vstacked:\n{vstacked}")
    
    # Horizontal stacking (equivalent to concatenate axis=1)
    hstacked = np.hstack([arr_2d_1, arr_2d_2])
    print(f"Hstacked:\n{hstacked}")
    
    # Column stacking (for 1D arrays)
    col_stacked = np.column_stack([arr1, arr2])
    print(f"Column stacked:\n{col_stacked}")
    
    # Row stacking (for 1D arrays)
    row_stacked = np.row_stack([arr1, arr2])
    print(f"Row stacked:\n{row_stacked}")

    print_subsection_header("3D Array Concatenation")
    
    # Create 3D arrays
    arr_3d_1 = np.arange(8).reshape(2, 2, 2)
    arr_3d_2 = np.arange(8, 16).reshape(2, 2, 2)
    
    print(f"3D Array 1 shape: {arr_3d_1.shape}")
    print(f"3D Array 1:\n{arr_3d_1}")
    print(f"3D Array 2 shape: {arr_3d_2.shape}")
    print(f"3D Array 2:\n{arr_3d_2}")
    
    # Concatenate along different axes
    concat_axis0 = np.concatenate([arr_3d_1, arr_3d_2], axis=0)
    print(f"Concatenated axis=0 shape: {concat_axis0.shape}")
    
    concat_axis1 = np.concatenate([arr_3d_1, arr_3d_2], axis=1)
    print(f"Concatenated axis=1 shape: {concat_axis1.shape}")
    
    concat_axis2 = np.concatenate([arr_3d_1, arr_3d_2], axis=2)
    print(f"Concatenated axis=2 shape: {concat_axis2.shape}")

    # Section 5: Array Splitting
    print_section_header("5. Array Splitting")
    
    print("""
Splitting divides arrays into smaller parts, which is useful for 
data processing and analysis.
""")

    print_subsection_header("Basic Splitting")
    
    # Create array to split
    arr_to_split = np.arange(12)
    print(f"Array to split: {arr_to_split}")
    
    # Split into equal parts
    split_2 = np.split(arr_to_split, 2)
    print(f"Split into 2 parts: {split_2}")
    
    # Split into 3 parts
    split_3 = np.split(arr_to_split, 3)
    print(f"Split into 3 parts: {split_3}")
    
    # Split at specific indices
    split_indices = np.split(arr_to_split, [3, 7])
    print(f"Split at indices [3, 7]: {split_indices}")

    print_subsection_header("2D Array Splitting")
    
    # Create 2D array
    arr_2d_split = np.arange(16).reshape(4, 4)
    print(f"2D array to split:\n{arr_2d_split}")
    
    # Split along rows (axis=0)
    split_rows = np.split(arr_2d_split, 2, axis=0)
    print(f"Split along rows:\n{split_rows[0]}\n{split_rows[1]}")
    
    # Split along columns (axis=1)
    split_cols = np.split(arr_2d_split, 2, axis=1)
    print(f"Split along columns:\n{split_cols[0]}\n{split_cols[1]}")

    print_subsection_header("Alternative Splitting Functions")
    
    print("NumPy provides specialized splitting functions:")
    
    # Vertical splitting
    vsplit_result = np.vsplit(arr_2d_split, 2)
    print(f"Vsplit result:\n{vsplit_result[0]}\n{vsplit_result[1]}")
    
    # Horizontal splitting
    hsplit_result = np.hsplit(arr_2d_split, 2)
    print(f"Hsplit result:\n{hsplit_result[0]}\n{hsplit_result[1]}")
    
    # Array splitting with unequal parts
    arr_unequal = np.arange(10)
    split_unequal = np.array_split(arr_unequal, 3)  # Handles unequal division
    print(f"Array split into 3 (unequal): {split_unequal}")

    print_subsection_header("Advanced Splitting")
    
    # Split with custom indices
    arr_custom = np.arange(20)
    custom_indices = [5, 10, 15]
    split_custom = np.split(arr_custom, custom_indices)
    print(f"Custom split at {custom_indices}:")
    for i, part in enumerate(split_custom):
        print(f"  Part {i}: {part}")

    # Section 6: Broadcasting and Array Alignment
    print_section_header("6. Broadcasting and Array Alignment")
    
    print("""
Broadcasting allows NumPy to work with arrays of different shapes 
by automatically aligning them for operations.
""")

    print_subsection_header("Basic Broadcasting")
    
    # Create arrays of different shapes
    arr_2d_broadcast = np.array([[1, 2, 3], [4, 5, 6]])
    arr_1d_broadcast = np.array([10, 20, 30])
    
    print(f"2D array:\n{arr_2d_broadcast}")
    print(f"1D array: {arr_1d_broadcast}")
    
    # Broadcasting addition
    broadcasted_sum = arr_2d_broadcast + arr_1d_broadcast
    print(f"Broadcasted sum:\n{broadcasted_sum}")
    
    # Broadcasting multiplication
    broadcasted_mult = arr_2d_broadcast * arr_1d_broadcast
    print(f"Broadcasted multiplication:\n{broadcasted_mult}")

    print_subsection_header("Broadcasting Rules")
    
    print("Broadcasting follows specific rules:")
    
    # Example 1: Scalar broadcasting
    scalar = 5
    scalar_broadcast = arr_2d_broadcast + scalar
    print(f"Scalar broadcasting:\n{scalar_broadcast}")
    
    # Example 2: Column vector broadcasting
    col_vector = np.array([[1], [2]])
    col_broadcast = arr_2d_broadcast + col_vector
    print(f"Column vector broadcasting:\n{col_broadcast}")
    
    # Example 3: Complex broadcasting
    arr_3d_broadcast = np.arange(24).reshape(2, 3, 4)
    arr_2d_broadcast_complex = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    
    print(f"3D array shape: {arr_3d_broadcast.shape}")
    print(f"2D array shape: {arr_2d_broadcast_complex.shape}")
    
    complex_broadcast = arr_3d_broadcast + arr_2d_broadcast_complex
    print(f"Complex broadcasting result shape: {complex_broadcast.shape}")

    print_subsection_header("Broadcasting for Reshaping")
    
    print("Using broadcasting to align arrays for operations:")
    
    # Create arrays that need alignment
    arr_alignment_1 = np.array([1, 2, 3, 4])
    arr_alignment_2 = np.array([10, 20])
    
    # Reshape for broadcasting
    arr_alignment_1_reshaped = arr_alignment_1.reshape(-1, 1)
    arr_alignment_2_reshaped = arr_alignment_2.reshape(1, -1)
    
    print(f"Array 1 reshaped:\n{arr_alignment_1_reshaped}")
    print(f"Array 2 reshaped:\n{arr_alignment_2_reshaped}")
    
    # Outer product using broadcasting
    outer_product = arr_alignment_1_reshaped * arr_alignment_2_reshaped
    print(f"Outer product:\n{outer_product}")

    # Section 7: Advanced Array Operations
    print_section_header("7. Advanced Array Operations")
    
    print("""
Advanced array operations combine multiple manipulation techniques 
for complex data transformations.
""")

    print_subsection_header("Array Padding")
    
    print("Padding arrays with values:")
    
    # Create array to pad
    arr_pad = np.array([[1, 2], [3, 4]])
    print(f"Original array:\n{arr_pad}")
    
    # Pad with zeros
    padded_zeros = np.pad(arr_pad, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    print(f"Padded with zeros:\n{padded_zeros}")
    
    # Pad with edge values
    padded_edge = np.pad(arr_pad, ((1, 1), (1, 1)), mode='edge')
    print(f"Padded with edge values:\n{padded_edge}")
    
    # Pad with reflection
    padded_reflect = np.pad(arr_pad, ((1, 1), (1, 1)), mode='reflect')
    print(f"Padded with reflection:\n{padded_reflect}")

    print_subsection_header("Array Tiling and Repeating")
    
    print("Repeating arrays in different patterns:")
    
    # Create base array
    arr_tile = np.array([1, 2, 3])
    print(f"Base array: {arr_tile}")
    
    # Tile array
    tiled = np.tile(arr_tile, 3)
    print(f"Tiled 3 times: {tiled}")
    
    # Tile in 2D
    tiled_2d = np.tile(arr_tile, (3, 2))
    print(f"Tiled 2D:\n{tiled_2d}")
    
    # Repeat elements
    repeated = np.repeat(arr_tile, 2)
    print(f"Repeated elements: {repeated}")

    print_subsection_header("Array Masking and Filtering")
    
    print("Using boolean masks for array manipulation:")
    
    # Create array with mask
    arr_mask = np.arange(10)
    mask = arr_mask % 2 == 0  # Even numbers
    
    print(f"Original array: {arr_mask}")
    print(f"Mask (even numbers): {mask}")
    
    # Apply mask
    even_numbers = arr_mask[mask]
    print(f"Even numbers: {even_numbers}")
    
    # Set masked values
    arr_masked = arr_mask.copy()
    arr_masked[mask] = -1
    print(f"Array with masked values: {arr_masked}")

    print_subsection_header("Array Sorting and Partitioning")
    
    print("Sorting and partitioning arrays:")
    
    # Create array to sort
    arr_sort = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    print(f"Original array: {arr_sort}")
    
    # Sort array
    sorted_arr = np.sort(arr_sort)
    print(f"Sorted array: {sorted_arr}")
    
    # Sort in place
    arr_sort_inplace = arr_sort.copy()
    arr_sort_inplace.sort()
    print(f"Sorted in place: {arr_sort_inplace}")
    
    # Partial sort (partition)
    partitioned = np.partition(arr_sort, 3)  # First 3 elements are smallest
    print(f"Partitioned (first 3 smallest): {partitioned}")

    # Section 8: Memory Layout and Performance
    print_section_header("8. Memory Layout and Performance")
    
    print("""
Understanding memory layout is crucial for optimizing array operations 
and avoiding performance pitfalls.
""")

    print_subsection_header("Array Memory Layout")
    
    print("Checking array memory layout:")
    
    # Create array
    arr_memory = np.arange(12).reshape(3, 4)
    print(f"Array:\n{arr_memory}")
    
    # Check memory layout
    print(f"Memory layout (C-contiguous): {arr_memory.flags['C_CONTIGUOUS']}")
    print(f"Memory layout (F-contiguous): {arr_memory.flags['F_CONTIGUOUS']}")
    print(f"Strides: {arr_memory.strides}")
    
    # Transpose and check layout
    arr_transposed = arr_memory.T
    print(f"Transposed array:\n{arr_transposed}")
    print(f"Transposed C-contiguous: {arr_transposed.flags['C_CONTiguous']}")
    print(f"Transposed strides: {arr_transposed.strides}")

    print_subsection_header("Performance Considerations")
    
    print("Performance implications of different operations:")
    
    # Create large array
    large_arr = np.random.rand(1000, 1000)
    
    # Time different operations
    import time
    
    # Row-wise operations (C-contiguous)
    start_time = time.time()
    row_sum = np.sum(large_arr, axis=1)
    row_time = time.time() - start_time
    
    # Column-wise operations (slower for C-contiguous)
    start_time = time.time()
    col_sum = np.sum(large_arr, axis=0)
    col_time = time.time() - start_time
    
    print(f"Large array shape: {large_arr.shape}")
    print(f"Row-wise sum time: {row_time:.6f} seconds")
    print(f"Column-wise sum time: {col_time:.6f} seconds")
    print(f"Performance ratio: {col_time/row_time:.2f}x")

    print_subsection_header("Memory-Efficient Operations")
    
    print("Memory-efficient array operations:")
    
    # Create arrays
    arr1_mem = np.random.rand(1000, 1000)
    arr2_mem = np.random.rand(1000, 1000)
    
    # Memory-efficient operations
    print("Memory-efficient operations:")
    print("  - Use in-place operations when possible")
    print("  - Avoid unnecessary copies")
    print("  - Use views instead of copies")
    
    # Example: In-place addition
    arr1_mem += arr2_mem  # In-place (memory efficient)
    # vs
    # arr1_mem = arr1_mem + arr2_mem  # Creates new array (less efficient)

    # Section 9: Practical Applications
    print_section_header("9. Practical Applications")
    
    print("""
Let's apply array manipulation techniques to real-world scenarios:
""")

    print_subsection_header("Image Processing Example")
    
    print("Array manipulation for image processing:")
    
    # Simulate image data (grayscale, 5x5)
    image = np.random.randint(0, 256, (5, 5))
    print(f"Original image:\n{image}")
    
    # Reshape for processing
    image_reshaped = image.reshape(-1)  # Flatten
    print(f"Flattened image: {image_reshaped}")
    
    # Apply transformation
    image_transformed = image_reshaped * 2  # Brighten
    image_transformed = np.clip(image_transformed, 0, 255)  # Clip values
    
    # Reshape back
    image_processed = image_transformed.reshape(5, 5)
    print(f"Processed image:\n{image_processed}")

    print_subsection_header("Data Preprocessing Example")
    
    print("Array manipulation for data preprocessing:")
    
    # Create sample dataset
    features = np.random.rand(100, 5)
    labels = np.random.randint(0, 2, 100)
    
    print(f"Features shape: {features.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Split into train/test
    split_idx = 80
    X_train = features[:split_idx]
    X_test = features[split_idx:]
    y_train = labels[:split_idx]
    y_test = labels[split_idx:]
    
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    # Normalize features
    mean_train = np.mean(X_train, axis=0)
    std_train = np.std(X_train, axis=0)
    
    X_train_normalized = (X_train - mean_train) / std_train
    X_test_normalized = (X_test - mean_train) / std_train
    
    print(f"Normalized training mean: {np.mean(X_train_normalized, axis=0)}")
    print(f"Normalized training std: {np.std(X_train_normalized, axis=0)}")

    print_subsection_header("Time Series Processing")
    
    print("Array manipulation for time series data:")
    
    # Create time series data
    time_series = np.random.randn(100)
    print(f"Time series length: {len(time_series)}")
    
    # Create sliding windows
    window_size = 10
    n_windows = len(time_series) - window_size + 1
    
    windows = np.array([time_series[i:i+window_size] for i in range(n_windows)])
    print(f"Sliding windows shape: {windows.shape}")
    
    # Compute moving average
    moving_avg = np.mean(windows, axis=1)
    print(f"Moving average length: {len(moving_avg)}")
    print(f"First few moving averages: {moving_avg[:5]}")

    # Section 10: Best Practices and Tips
    print_section_header("10. Best Practices and Tips")
    
    print("""
Follow these best practices for efficient and reliable array manipulation:
""")

    print_subsection_header("Code Organization")
    
    print("""
1. Use descriptive variable names:
   ```python
   # Good
   reshaped_features = features.reshape(-1, 1)
   concatenated_data = np.concatenate([train_data, test_data])
   
   # Avoid
   a = b.reshape(-1, 1)
   c = np.concatenate([d, e])
   ```

2. Check array shapes before operations:
   ```python
   print(f"Array shape: {array.shape}")
   print(f"Expected shape: {expected_shape}")
   assert array.shape == expected_shape, "Shape mismatch"
   ```

3. Use appropriate data types:
   ```python
   # For integers
   int_array = np.array([1, 2, 3], dtype=np.int32)
   
   # For floats
   float_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
   ```
""")

    print_subsection_header("Performance Optimization")
    
    print("""
1. Avoid unnecessary copies:
   ```python
   # Good - uses view
   arr_view = arr.reshape(-1)
   
   # Avoid - creates copy
   arr_copy = arr.copy().reshape(-1)
   ```

2. Use in-place operations when possible:
   ```python
   # Good - in-place
   arr += 1
   arr *= 2
   
   # Avoid - creates new arrays
   arr = arr + 1
   arr = arr * 2
   ```

3. Consider memory layout:
   ```python
   # For C-contiguous arrays, row operations are faster
   row_sum = np.sum(arr, axis=1)  # Fast
   col_sum = np.sum(arr, axis=0)  # Slower
   ```
""")

    print_subsection_header("Error Handling")
    
    print("""
1. Validate array shapes before operations:
   ```python
   if arr1.shape != arr2.shape:
       raise ValueError("Array shapes must match")
   ```

2. Check for valid reshape operations:
   ```python
   total_elements = np.prod(arr.shape)
   if total_elements != new_shape_elements:
       raise ValueError("Cannot reshape array")
   ```

3. Handle broadcasting errors:
   ```python
   try:
       result = arr1 + arr2
   except ValueError as e:
       print(f"Broadcasting error: {e}")
   ```
""")

    # Section 11: Summary and Next Steps
    print_section_header("11. Summary and Next Steps")
    
    print("""
Congratulations! You've completed the NumPy array manipulation tutorial. Here's what you've learned:

Key Concepts Covered:
✅ Array Reshaping: Changing dimensions and structure
✅ Array Transposition: Rearranging array dimensions
✅ Array Concatenation: Combining arrays along axes
✅ Array Splitting: Dividing arrays into parts
✅ Broadcasting: Aligning arrays for operations
✅ Advanced Operations: Padding, tiling, masking
✅ Memory Layout: Understanding performance implications
✅ Practical Applications: Real-world use cases
✅ Best Practices: Efficient and reliable code

Next Steps:

1. Practice Reshaping: Work with arrays of different dimensions
2. Master Concatenation: Combine arrays in various ways
3. Explore Broadcasting: Understand alignment rules thoroughly
4. Optimize Performance: Profile and optimize your operations
5. Apply to Real Data: Use these techniques with actual datasets
6. Study Advanced Topics: Learn about sparse arrays and specialized operations

Additional Resources:
- NumPy Array Manipulation: https://numpy.org/doc/stable/reference/routines.array-manipulation.html
- NumPy Broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html
- NumPy Memory Layout: https://numpy.org/doc/stable/reference/arrays.ndarray.html
- Performance Tips: Various NumPy optimization guides

Practice Exercises:
1. Reshape arrays between different dimensions
2. Concatenate arrays along various axes
3. Use broadcasting for efficient operations
4. Implement data preprocessing pipelines
5. Optimize array operations for performance
6. Build complex data transformation workflows

Happy Array Manipulation! 🚀
""")

if __name__ == "__main__":
    # Run the tutorial
    main()
    
    print("\n" + "="*60)
    print(" Tutorial completed successfully!")
    print(" Master NumPy array manipulation!")
    print("="*60) 