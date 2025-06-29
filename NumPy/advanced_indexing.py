#!/usr/bin/env python3
"""
NumPy Advanced Indexing: Mastering Array Access

Welcome to the NumPy advanced indexing tutorial! This tutorial covers 
advanced techniques for accessing and manipulating NumPy arrays, including 
boolean indexing, fancy indexing, and complex array operations.

This script covers:
- Boolean indexing and masking
- Fancy indexing with integer arrays
- Advanced slicing techniques
- Array masking and filtering
- Structured array indexing
- Performance considerations

Prerequisites:
- Python 3.8 or higher
- Basic understanding of NumPy (covered in numpy_basics.py)
- NumPy installed (pip install numpy)
"""

import numpy as np
import sys
import time

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
    
    print("NumPy Advanced Indexing Tutorial")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print("Advanced indexing tutorial started successfully!")

    # Section 1: Introduction to Advanced Indexing
    print_section_header("1. Introduction to Advanced Indexing")
    
    print("""
Advanced indexing in NumPy goes beyond basic integer indexing and slicing. 
It includes powerful techniques for accessing array elements based on 
conditions, patterns, and complex criteria.

Key Advanced Indexing Techniques:
- Boolean Indexing: Using boolean arrays to select elements
- Fancy Indexing: Using integer arrays to select elements
- Structured Indexing: Working with structured arrays
- Advanced Masking: Complex filtering operations
- Performance Optimization: Efficient indexing strategies

Benefits:
✅ Complex data filtering and selection
✅ Efficient data manipulation
✅ Vectorized operations
✅ Memory-efficient operations
✅ Powerful data analysis capabilities
""")

    # Section 2: Boolean Indexing
    print_section_header("2. Boolean Indexing")
    
    print("""
Boolean indexing allows you to select array elements based on conditions. 
It's one of the most powerful features of NumPy for data filtering.
""")

    print_subsection_header("Basic Boolean Indexing")
    
    # Create sample data
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"Original data: {data}")
    
    # Create boolean mask
    mask = data > 5
    print(f"Boolean mask (data > 5): {mask}")
    
    # Apply boolean indexing
    filtered_data = data[mask]
    print(f"Filtered data (data > 5): {filtered_data}")
    
    print("\nMultiple conditions:")
    mask2 = (data > 3) & (data < 8)
    print(f"Mask (3 < data < 8): {mask2}")
    print(f"Filtered data: {data[mask2]}")

    print_subsection_header("Boolean Indexing with 2D Arrays")
    
    # Create 2D array
    arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"2D array:\n{arr_2d}")
    
    # Boolean indexing on 2D array
    mask_2d = arr_2d > 5
    print(f"Boolean mask (arr_2d > 5):\n{mask_2d}")
    
    # Get elements where condition is True
    elements_gt_5 = arr_2d[mask_2d]
    print(f"Elements > 5: {elements_gt_5}")
    
    # Set elements based on condition
    arr_2d_copy = arr_2d.copy()
    arr_2d_copy[mask_2d] = 0
    print(f"Array with elements > 5 set to 0:\n{arr_2d_copy}")

    print_subsection_header("Complex Boolean Conditions")
    
    # Create more complex data
    scores = np.array([85, 92, 78, 96, 88, 75, 91, 83, 89, 94])
    print(f"Test scores: {scores}")
    
    # Multiple conditions
    high_scores = scores[(scores >= 90) & (scores <= 100)]
    print(f"High scores (90-100): {high_scores}")
    
    # Using logical functions
    import numpy as np
    good_scores = scores[np.logical_and(scores >= 80, scores < 90)]
    print(f"Good scores (80-89): {good_scores}")
    
    # Using where function
    indices = np.where(scores >= 85)
    print(f"Indices of scores >= 85: {indices[0]}")
    print(f"Scores >= 85: {scores[indices]}")

    # Section 3: Fancy Indexing
    print_section_header("3. Fancy Indexing")
    
    print("""
Fancy indexing uses integer arrays to select elements from an array. 
It's useful for selecting specific elements based on their positions.
""")

    print_subsection_header("Basic Fancy Indexing")
    
    arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    print(f"Original array: {arr}")
    
    # Select specific indices
    indices = [0, 2, 5, 7]
    selected = arr[indices]
    print(f"Selected indices {indices}: {selected}")
    
    # Negative indices work too
    neg_indices = [-1, -3, -5]
    selected_neg = arr[neg_indices]
    print(f"Selected negative indices {neg_indices}: {selected_neg}")

    print_subsection_header("Fancy Indexing with 2D Arrays")
    
    matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(f"Matrix:\n{matrix}")
    
    # Select specific rows
    row_indices = [0, 2]
    selected_rows = matrix[row_indices]
    print(f"Selected rows {row_indices}:\n{selected_rows}")
    
    # Select specific elements
    row_idx = [0, 1, 2]
    col_idx = [1, 2, 3]
    selected_elements = matrix[row_idx, col_idx]
    print(f"Selected elements at positions (0,1), (1,2), (2,3): {selected_elements}")

    print_subsection_header("Advanced Fancy Indexing")
    
    # Create a larger array
    large_arr = np.arange(25).reshape(5, 5)
    print(f"5x5 array:\n{large_arr}")
    
    # Select a subarray using fancy indexing
    row_indices = [0, 2, 4]
    col_indices = [1, 3]
    subarray = large_arr[row_indices][:, col_indices]
    print(f"Subarray (rows {row_indices}, cols {col_indices}):\n{subarray}")
    
    # Using meshgrid for complex indexing
    rows, cols = np.meshgrid([0, 2], [1, 3])
    print(f"Row indices:\n{rows}")
    print(f"Column indices:\n{cols}")
    selected = large_arr[rows, cols]
    print(f"Selected using meshgrid:\n{selected}")

    # Section 4: Advanced Masking Techniques
    print_section_header("4. Advanced Masking Techniques")
    
    print("""
Advanced masking combines boolean indexing with other NumPy operations 
to create complex data filtering and manipulation workflows.
""")

    print_subsection_header("Conditional Masking")
    
    # Create sample dataset
    temperatures = np.random.normal(20, 5, 100)
    print(f"Temperature data (first 10 values): {temperatures[:10]}")
    
    # Create masks for different temperature ranges
    cold_mask = temperatures < 15
    warm_mask = (temperatures >= 15) & (temperatures < 25)
    hot_mask = temperatures >= 25
    
    print(f"Cold temperatures (< 15°C): {temperatures[cold_mask][:10]}")
    print(f"Warm temperatures (15-25°C): {temperatures[warm_mask][:10]}")
    print(f"Hot temperatures (≥ 25°C): {temperatures[hot_mask][:10]}")

    print_subsection_header("Masked Arrays")
    
    # Create masked array
    data_with_missing = np.array([1, 2, -999, 4, 5, -999, 7, 8])
    print(f"Data with missing values (-999): {data_with_missing}")
    
    # Create mask for missing values
    missing_mask = data_with_missing == -999
    print(f"Missing value mask: {missing_mask}")
    
    # Create masked array
    masked_data = np.ma.masked_array(data_with_missing, missing_mask)
    print(f"Masked array: {masked_data}")
    print(f"Valid data: {masked_data.compressed()}")
    
    # Calculate statistics ignoring missing values
    print(f"Mean (ignoring missing): {np.ma.mean(masked_data)}")
    print(f"Standard deviation (ignoring missing): {np.ma.std(masked_data)}")

    print_subsection_header("Complex Filtering")
    
    # Create multi-dimensional data
    students = np.array([
        ['Alice', 85, 'A'],
        ['Bob', 92, 'A'],
        ['Charlie', 78, 'C'],
        ['Diana', 96, 'A'],
        ['Eve', 88, 'B'],
        ['Frank', 75, 'C']
    ])
    
    print("Student data:")
    for student in students:
        print(f"  {student[0]}: {student[1]} ({student[2]})")
    
    # Filter by grade
    a_students = students[students[:, 2] == 'A']
    print(f"\nA students:\n{a_students}")
    
    # Filter by score range
    high_achievers = students[students[:, 1].astype(int) >= 90]
    print(f"\nHigh achievers (≥90):\n{high_achievers}")

    # Section 5: Structured Array Indexing
    print_section_header("5. Structured Array Indexing")
    
    print("""
Structured arrays allow you to work with heterogeneous data types 
in a single array, similar to a table or database.
""")

    print_subsection_header("Creating Structured Arrays")
    
    # Define data types
    dtype = [('name', 'U10'), ('age', 'i4'), ('height', 'f4'), ('grade', 'U1')]
    
    # Create structured array
    students_structured = np.array([
        ('Alice', 20, 165.5, 'A'),
        ('Bob', 22, 180.2, 'B'),
        ('Charlie', 19, 172.0, 'A'),
        ('Diana', 21, 168.8, 'C')
    ], dtype=dtype)
    
    print(f"Structured array:\n{students_structured}")
    print(f"Data type: {students_structured.dtype}")

    print_subsection_header("Accessing Structured Array Fields")
    
    # Access by field name
    names = students_structured['name']
    ages = students_structured['age']
    heights = students_structured['height']
    
    print(f"Names: {names}")
    print(f"Ages: {ages}")
    print(f"Heights: {heights}")
    
    # Access specific elements
    alice_data = students_structured[0]
    print(f"Alice's data: {alice_data}")
    print(f"Alice's age: {alice_data['age']}")

    print_subsection_header("Filtering Structured Arrays")
    
    # Filter by age
    young_students = students_structured[students_structured['age'] < 21]
    print(f"Young students (< 21):\n{young_students}")
    
    # Filter by grade
    a_students_structured = students_structured[students_structured['grade'] == 'A']
    print(f"A students:\n{a_students_structured}")
    
    # Complex filtering
    tall_a_students = students_structured[
        (students_structured['height'] > 170) & 
        (students_structured['grade'] == 'A')
    ]
    print(f"Tall A students:\n{tall_a_students}")

    # Section 6: Performance Considerations
    print_section_header("6. Performance Considerations")
    
    print("""
Understanding performance implications of different indexing methods 
is crucial for efficient data processing.
""")

    print_subsection_header("Indexing Performance Comparison")
    
    # Create large array for testing
    large_array = np.random.rand(10000, 1000)
    print(f"Large array shape: {large_array.shape}")
    
    # Test boolean indexing performance
    start_time = time.time()
    mask = large_array > 0.5
    boolean_result = large_array[mask]
    boolean_time = time.time() - start_time
    print(f"Boolean indexing time: {boolean_time:.4f} seconds")
    print(f"Selected elements: {len(boolean_result)}")
    
    # Test fancy indexing performance
    start_time = time.time()
    indices = np.random.choice(10000, 1000, replace=False)
    fancy_result = large_array[indices]
    fancy_time = time.time() - start_time
    print(f"Fancy indexing time: {fancy_time:.4f} seconds")
    print(f"Selected elements: {len(fancy_result)}")

    print_subsection_header("Memory Efficiency")
    
    # Demonstrate memory-efficient operations
    original = np.arange(1000)
    print(f"Original array size: {original.nbytes} bytes")
    
    # Boolean indexing creates a view (memory efficient)
    mask = original % 2 == 0
    even_numbers = original[mask]
    print(f"Even numbers array size: {even_numbers.nbytes} bytes")
    print(f"Memory efficient: {even_numbers.nbytes < original.nbytes}")

    # Section 7: Practical Applications
    print_section_header("7. Practical Applications")
    
    print("""
Let's apply advanced indexing techniques to real-world scenarios:
""")

    print_subsection_header("Data Analysis Example")
    
    # Simulate sales data
    np.random.seed(42)
    sales_data = np.array([
        np.random.randint(100, 1000, 100),  # Product A sales
        np.random.randint(50, 500, 100),    # Product B sales
        np.random.randint(200, 800, 100),   # Product C sales
        np.random.randint(75, 600, 100)     # Product D sales
    ]).T
    
    print("Sales data shape:", sales_data.shape)
    print("First 5 rows:")
    print(sales_data[:5])
    
    # Find best performing days (total sales > 2000)
    total_sales = np.sum(sales_data, axis=1)
    best_days = sales_data[total_sales > 2000]
    print(f"\nBest performing days (total sales > 2000): {len(best_days)} days")
    print("Sample of best days:")
    print(best_days[:3])
    
    # Find days where Product A outperformed others
    product_a_best = sales_data[sales_data[:, 0] > sales_data[:, 1:].max(axis=1)]
    print(f"\nDays where Product A was best: {len(product_a_best)} days")

    print_subsection_header("Image Processing Example")
    
    # Simulate image data (grayscale, 100x100)
    image = np.random.randint(0, 256, (100, 100))
    print(f"Image shape: {image.shape}")
    print(f"Image value range: {image.min()} to {image.max()}")
    
    # Apply threshold (binary image)
    threshold = 128
    binary_mask = image > threshold
    binary_image = image.copy()
    binary_image[binary_mask] = 255
    binary_image[~binary_mask] = 0
    
    print(f"Pixels above threshold: {np.sum(binary_mask)}")
    print(f"Pixels below threshold: {np.sum(~binary_mask)}")
    
    # Extract bright regions
    bright_regions = image[image > 200]
    print(f"Bright pixels (>200): {len(bright_regions)}")
    print(f"Average brightness: {np.mean(bright_regions):.2f}")

    print_subsection_header("Time Series Analysis")
    
    # Simulate time series data
    time_points = np.arange(1000)
    signal = np.sin(time_points * 0.1) + np.random.normal(0, 0.1, 1000)
    
    print(f"Time series length: {len(signal)}")
    print(f"Signal range: {signal.min():.3f} to {signal.max():.3f}")
    
    # Find peaks (local maxima)
    peaks = (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])
    peak_indices = np.where(peaks)[0] + 1
    peak_values = signal[peak_indices]
    
    print(f"Number of peaks found: {len(peak_indices)}")
    print(f"Peak values: {peak_values[:10]}")  # Show first 10 peaks
    
    # Find high-amplitude events
    high_amplitude = np.abs(signal) > 1.0
    high_amp_indices = np.where(high_amplitude)[0]
    print(f"High amplitude events: {len(high_amp_indices)}")

    # Section 8: Advanced Techniques
    print_section_header("8. Advanced Techniques")
    
    print("""
Advanced indexing techniques for complex data manipulation:
""")

    print_subsection_header("Multi-dimensional Boolean Indexing")
    
    # Create 3D array
    cube = np.random.rand(5, 5, 5)
    print(f"3D cube shape: {cube.shape}")
    
    # Apply 3D boolean indexing
    high_values = cube > 0.8
    selected_values = cube[high_values]
    print(f"Values > 0.8: {len(selected_values)} elements")
    print(f"Selected values: {selected_values[:10]}")  # Show first 10

    print_subsection_header("Conditional Array Modification")
    
    # Create array for modification
    data_mod = np.random.rand(10, 10)
    print(f"Original data (first 3x3):\n{data_mod[:3, :3]}")
    
    # Apply conditional modifications
    data_mod[data_mod < 0.3] = 0  # Set low values to 0
    data_mod[data_mod > 0.7] = 1  # Set high values to 1
    
    print(f"Modified data (first 3x3):\n{data_mod[:3, :3]}")

    print_subsection_header("Indexing with Functions")
    
    # Create data
    values = np.random.rand(100)
    print(f"Data range: {values.min():.3f} to {values.max():.3f}")
    
    # Use functions to create indices
    def get_extreme_indices(arr, n=5):
        """Get indices of n most extreme values."""
        sorted_indices = np.argsort(np.abs(arr))
        return sorted_indices[-n:]
    
    extreme_indices = get_extreme_indices(values, 10)
    extreme_values = values[extreme_indices]
    print(f"10 most extreme values: {extreme_values}")

    # Section 9: Best Practices
    print_section_header("9. Best Practices")
    
    print("""
Follow these best practices for efficient and readable advanced indexing:
""")

    print_subsection_header("Code Organization")
    
    print("""
1. Use descriptive variable names for masks:
   ```python
   # Good
   high_scores_mask = scores >= 90
   high_scores = scores[high_scores_mask]
   
   # Avoid
   mask = scores >= 90
   result = scores[mask]
   ```

2. Break complex conditions into steps:
   ```python
   # Good
   age_mask = ages >= 18
   score_mask = scores >= 80
   eligible_mask = age_mask & score_mask
   eligible_students = students[eligible_mask]
   
   # Avoid
   eligible_students = students[(ages >= 18) & (scores >= 80)]
   ```

3. Use parentheses for clarity:
   ```python
   # Good
   mask = (data > 0) & (data < 100)
   
   # Avoid
   mask = data > 0 & data < 100  # Can be ambiguous
   ```
""")

    print_subsection_header("Performance Tips")
    
    print("""
1. Avoid repeated boolean operations:
   ```python
   # Good - compute once, use multiple times
   mask = data > threshold
   filtered_data = data[mask]
   count = np.sum(mask)
   
   # Avoid - compute multiple times
   filtered_data = data[data > threshold]
   count = np.sum(data > threshold)
   ```

2. Use in-place operations when possible:
   ```python
   # Good
   data[data < 0] = 0
   
   # Avoid
   data = np.where(data < 0, 0, data)
   ```

3. Consider memory usage for large arrays:
   ```python
   # For very large arrays, consider chunked processing
   chunk_size = 10000
   for i in range(0, len(large_array), chunk_size):
       chunk = large_array[i:i+chunk_size]
       # Process chunk
   ```
""")

    # Section 10: Summary and Next Steps
    print_section_header("10. Summary and Next Steps")
    
    print("""
Congratulations! You've completed the NumPy advanced indexing tutorial. Here's what you've learned:

Key Concepts Covered:
✅ Boolean Indexing: Using conditions to filter arrays
✅ Fancy Indexing: Using integer arrays for selection
✅ Advanced Masking: Complex filtering operations
✅ Structured Arrays: Working with heterogeneous data
✅ Performance Optimization: Efficient indexing strategies
✅ Practical Applications: Real-world use cases
✅ Best Practices: Writing efficient and readable code

Next Steps:

1. Practice Boolean Indexing: Work with different conditions and datasets
2. Master Fancy Indexing: Experiment with complex selection patterns
3. Explore Structured Arrays: Work with tabular and heterogeneous data
4. Optimize Performance: Profile and optimize your indexing operations
5. Apply to Real Data: Use these techniques with actual datasets
6. Learn Related Libraries: Explore pandas for more advanced data manipulation

Additional Resources:
- NumPy Advanced Indexing: https://numpy.org/doc/stable/user/basics.indexing.html
- NumPy Structured Arrays: https://numpy.org/doc/stable/user/basics.rec.html
- NumPy Masked Arrays: https://numpy.org/doc/stable/reference/maskedarray.html
- NumPy Performance Tips: https://numpy.org/doc/stable/user/quickstart.html

Practice Exercises:
1. Create complex boolean masks for data filtering
2. Use fancy indexing to select specific data patterns
3. Work with structured arrays for tabular data
4. Optimize indexing operations for large datasets
5. Apply advanced indexing to real-world problems
6. Build data analysis pipelines using these techniques

Happy Advanced Indexing! 🚀
""")

if __name__ == "__main__":
    # Run the tutorial
    main()
    
    print("\n" + "="*60)
    print(" Tutorial completed successfully!")
    print(" Master advanced NumPy indexing!")
    print("="*60) 