# NumPy Tutorials for Scientific Computing

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

NumPy is the fundamental package for scientific computing in Python. It provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.

This module contains comprehensive Python script tutorials designed to help you master NumPy for Machine Learning and Data Science applications.

## What You'll Learn

- **Array Creation & Manipulation** - Create and modify NumPy arrays efficiently
- **Advanced Indexing** - Boolean indexing, fancy indexing, and complex array access
- **Linear Algebra Operations** - Matrix operations, decompositions, and solving systems
- **Random Generation** - Probability distributions and statistical sampling
- **Array Manipulation** - Reshaping, concatenation, splitting, and broadcasting
- **Mathematical Functions** - Fast vectorized computations and statistical operations
- **Performance Optimization** - Memory layout and efficient array operations

## Prerequisites

Before starting these tutorials, ensure you have:

- **Python 3.8+** installed on your system
- Basic understanding of [Python programming](https://www.python.org/doc/)
- [NumPy](https://numpy.org/install/) installed

## Getting Started

### Option 1: Local Environment Setup

#### Using Conda (Recommended)

```bash
# Create a new conda environment
conda env create -f environment.yml

# Activate the environment
conda activate ml

# Run tutorials
python numpy_basics.py
```

#### Using pip

```bash
# Install dependencies
pip install -r requirements.txt

# Run tutorials
python numpy_basics.py
```

### Option 2: Google Colab

1. Upload any of the Python script files to [Google Colab](https://colab.research.google.com/)
2. Run the cells to learn NumPy interactively

## File Structure

```
NumPy/
├── numpy_basics.py              # Essential array operations and concepts
├── advanced_indexing.py         # Boolean indexing, fancy indexing, masking
├── linear_algebra.py            # Matrix operations, decompositions, eigenvalues
├── random_generation.py         # Probability distributions, sampling, Monte Carlo
├── array_manipulation.py        # Reshaping, concatenation, splitting, broadcasting
├── numpy_ultraquick_tutorial.ipynb  # Original notebook (for reference)
├── environment.yml              # Conda environment configuration
├── requirements.txt             # pip dependencies
└── README.md                    # This file
```

## Tutorial Scripts

### 1. `numpy_basics.py` - Essential Array Operations
**Comprehensive coverage of fundamental NumPy concepts:**
- Introduction to NumPy arrays and their properties
- Array creation methods (from lists, functions, sequences)
- Array attributes and data types
- Basic mathematical operations and broadcasting
- Mathematical and statistical functions
- Indexing and slicing techniques
- Shape manipulation and transposition
- Practical examples and applications

**Run with:** `python numpy_basics.py`

### 2. `advanced_indexing.py` - Advanced Array Access
**Master complex array indexing and filtering:**
- Boolean indexing and masking techniques
- Fancy indexing with integer arrays
- Advanced slicing and array manipulation
- Structured array indexing
- Performance considerations and optimization
- Complex filtering and data selection
- Real-world applications and examples

**Run with:** `python advanced_indexing.py`

### 3. `linear_algebra.py` - Matrix Operations and Decompositions
**Comprehensive linear algebra with NumPy:**
- Matrix creation and basic operations
- Matrix multiplication and properties
- Linear system solving (Ax = b)
- Matrix decompositions (LU, QR, SVD, Cholesky)
- Eigenvalues and eigenvectors computation
- Vector operations and norms
- Applications in data science and machine learning
- Performance and numerical considerations

**Run with:** `python linear_algebra.py`

### 4. `random_generation.py` - Probability Distributions and Sampling
**Statistical random generation and sampling:**
- Random number generation basics and reproducibility
- Probability distributions (uniform, normal, exponential, etc.)
- Random array generation with specific shapes
- Statistical sampling methods (simple, stratified, bootstrap)
- Monte Carlo simulations and estimation
- Applications in data science and machine learning
- Best practices for reliable random generation

**Run with:** `python random_generation.py`

### 5. `array_manipulation.py` - Reshaping and Advanced Operations
**Advanced array manipulation techniques:**
- Array reshaping and transposition
- Concatenation and stacking operations
- Splitting and chunking arrays
- Broadcasting and array alignment
- Advanced indexing and manipulation
- Memory layout and performance optimization
- Practical applications and real-world examples

**Run with:** `python array_manipulation.py`

## Running the Tutorials

### Individual Tutorials
Run any tutorial script individually to focus on specific topics:

```bash
# Run basics tutorial
python numpy_basics.py

# Run advanced indexing tutorial
python advanced_indexing.py

# Run linear algebra tutorial
python linear_algebra.py

# Run random generation tutorial
python random_generation.py

# Run array manipulation tutorial
python array_manipulation.py
```

### Sequential Learning
For comprehensive learning, run the tutorials in order:

```bash
# Run all tutorials in sequence
python numpy_basics.py
python advanced_indexing.py
python linear_algebra.py
python random_generation.py
python array_manipulation.py
```

### Interactive Mode
For interactive learning, you can also run the scripts in an interactive Python session:

```python
# In Python interactive session
exec(open('numpy_basics.py').read())
```

## Environment Management

### Conda Commands

```bash
# Create new environment
conda env create -f environment.yml

# Activate environment
conda activate ml

# Update environment (when dependencies change)
conda env update -f environment.yml --prune

# Remove environment
conda remove --name ml --all

# List all environments
conda env list

# Deactivate current environment
conda deactivate
```

### Useful Commands

```bash
# Check installed packages
pip list

# Export current environment
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt

# Check NumPy version
python -c "import numpy; print(numpy.__version__)"
```

## Tutorial Content Overview

### Core Concepts Covered

1. **Array Fundamentals**
   - Creating arrays from various sources
   - Array attributes and properties
   - Data types and memory layout
   - Basic operations and broadcasting

2. **Advanced Indexing**
   - Boolean indexing and masking
   - Fancy indexing with integer arrays
   - Structured array operations
   - Performance optimization

3. **Linear Algebra**
   - Matrix operations and properties
   - Solving linear systems
   - Matrix decompositions
   - Eigenvalue problems

4. **Random Generation**
   - Probability distributions
   - Statistical sampling
   - Monte Carlo methods
   - Reproducibility and best practices

5. **Array Manipulation**
   - Reshaping and transposition
   - Concatenation and splitting
   - Broadcasting and alignment
   - Memory-efficient operations

### Practical Applications

- **Data Preprocessing**: Feature scaling, normalization, and transformation
- **Statistical Analysis**: Descriptive statistics, hypothesis testing, and inference
- **Machine Learning**: Feature engineering, data augmentation, and model validation
- **Scientific Computing**: Numerical simulations, signal processing, and optimization
- **Image Processing**: Array operations for image manipulation and analysis

## Learning Resources

### Official Documentation
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [NumPy Reference](https://numpy.org/doc/stable/reference/)
- [NumPy Tutorial](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)

### Additional Resources
- [NumPy Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)
- [NumPy GitHub Repository](https://github.com/numpy/numpy)
- [NumPy Community](https://numpy.org/community/)
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/reference/)

## Practice Exercises

After completing the tutorials, practice with these exercises:

1. **Array Creation**: Create arrays of different shapes and data types
2. **Indexing Practice**: Use boolean and fancy indexing on real datasets
3. **Linear Algebra**: Solve systems of equations and perform matrix decompositions
4. **Random Generation**: Generate samples from different distributions
5. **Array Manipulation**: Reshape and combine arrays for data analysis
6. **Performance**: Optimize array operations for large datasets

## Contributing

Found an error or have a suggestion? Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Acknowledgments

- [NumPy Development Team](https://numpy.org/about/) for creating this amazing library
- [Scientific Python Community](https://scientific-python.org/) for continuous support
- [Python Community](https://www.python.org/community/) for the excellent programming language

---

**Ready to master NumPy? Start with the basics tutorial and work your way up!**

```bash
python numpy_basics.py
```
