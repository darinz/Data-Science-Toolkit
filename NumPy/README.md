# NumPy Tutorials for Scientific Computing

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Conda](https://img.shields.io/badge/Conda-Environment-green.svg)](https://docs.conda.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

NumPy is the fundamental package for scientific computing in Python. It provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.

This module contains comprehensive markdown guides and Python script tutorials designed to help you master NumPy for Machine Learning and Data Science applications.

## What You'll Learn

- **Array Creation & Manipulation** - Create and modify NumPy arrays efficiently
- **Advanced Indexing** - Boolean indexing, fancy indexing, and complex array access
- **Linear Algebra Operations** - Matrix operations, decompositions, and solving systems
- **Random Generation** - Probability distributions and statistical sampling
- **Array Manipulation** - Reshaping, concatenation, splitting, and broadcasting
- **Mathematical Functions** - Fast vectorized computations and statistical operations
- **Performance Optimization** - Memory layout and efficient array operations

## Quick Start

### Prerequisites
- **Python 3.8+** installed on your system
- Basic understanding of [Python programming](https://www.python.org/doc/)
- [NumPy](https://numpy.org/install/) installed

### Installation Options

#### Option 1: Local Environment Setup

**Using Conda (Recommended):**
```bash
# Create a new conda environment
conda env create -f environment.yml

# Activate the environment
conda activate numpy-tutorials

# Run tutorials
python random_generation.py
```

**Using pip:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run tutorials
python random_generation.py
```

#### Option 2: Google Colab
1. Upload any of the Python script files to [Google Colab](https://colab.research.google.com/)
2. Run the cells to learn NumPy interactively

## File Structure

```
NumPy/
├── numpy_basics_guide.md         # Essential array operations and concepts
├── advanced_indexing_guide.md    # Boolean indexing, fancy indexing, masking
├── linear_algebra_guide.md       # Matrix operations, decompositions, eigenvalues
├── random_generation_guide.md    # Probability distributions, sampling, Monte Carlo
├── array_manipulation_guide.md   # Reshaping, concatenation, splitting, broadcasting
├── random_generation.py          # Python script for random generation examples
├── numpy_ultraquick_tutorial.ipynb  # Original notebook (for reference)
├── environment.yml               # Conda environment configuration
├── requirements.txt              # pip dependencies
└── README.md                     # This file
```

## Tutorial Guides

### 1. `numpy_basics_guide.md` - Essential Array Operations
**Comprehensive coverage of fundamental NumPy concepts:**
- Introduction to NumPy arrays and their properties
- Array creation methods (from lists, functions, sequences)
- Array attributes and data types
- Basic mathematical operations and broadcasting
- Mathematical and statistical functions
- Indexing and slicing techniques
- Shape manipulation and transposition
- Practical examples and applications

### 2. `advanced_indexing_guide.md` - Advanced Array Access
**Master complex array indexing and filtering:**
- Boolean indexing and masking techniques
- Fancy indexing with integer arrays
- Advanced slicing and array manipulation
- Structured array indexing
- Performance considerations and optimization
- Complex filtering and data selection
- Real-world applications and examples

### 3. `linear_algebra_guide.md` - Matrix Operations and Decompositions
**Comprehensive linear algebra with NumPy:**
- Matrix creation and basic operations
- Matrix multiplication and properties
- Linear system solving (Ax = b)
- Matrix decompositions (LU, QR, SVD, Cholesky)
- Eigenvalues and eigenvectors computation
- Vector operations and norms
- Applications in data science and machine learning
- Performance and numerical considerations

### 4. `random_generation_guide.md` - Probability Distributions and Sampling
**Statistical random generation and sampling:**
- Random number generation basics and reproducibility
- Probability distributions (uniform, normal, exponential, etc.)
- Random array generation with specific shapes
- Statistical sampling methods (simple, stratified, bootstrap)
- Monte Carlo simulations and estimation
- Applications in data science and machine learning
- Best practices for reliable random generation

### 5. `array_manipulation_guide.md` - Reshaping and Advanced Operations
**Advanced array manipulation techniques:**
- Array reshaping and transposition
- Concatenation and stacking operations
- Splitting and chunking arrays
- Broadcasting and array alignment
- Advanced indexing and manipulation
- Memory layout and performance optimization
- Practical applications and real-world examples

## Python Scripts

### `random_generation.py` - Interactive Random Generation Examples
**Executable Python script with comprehensive examples:**
- Complete random generation demonstrations
- Interactive examples and code snippets
- Practical applications and use cases
- Performance benchmarks and comparisons

**Run with:** `python random_generation.py`

## Running the Tutorials

### Reading the Guides
Open any markdown guide file in your preferred markdown viewer or text editor:

```bash
# View guides in terminal
cat numpy_basics_guide.md

# Open in text editor
code numpy_basics_guide.md
```

### Running Python Scripts
Execute the available Python script:

```bash
# Run random generation script
python random_generation.py
```

### Interactive Mode
For interactive learning, you can also run the script in an interactive Python session:

```python
# In Python interactive session
exec(open('random_generation.py').read())
```

## Environment Management

### Conda Commands

```bash
# Create new environment
conda env create -f environment.yml

# Activate environment
conda activate numpy-tutorials

# Update environment (when dependencies change)
conda env update -f environment.yml --prune

# Remove environment
conda remove --name numpy-tutorials --all

# List all environments
conda env list

# Deactivate current environment
conda deactivate
```

### pip Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Check installed packages
pip list

# Export current environment
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
```

## Learning Path

### Beginner Path
1. **Start with `numpy_basics_guide.md`** - Learn fundamental array operations
2. **Practice with `array_manipulation_guide.md`** - Master reshaping and manipulation
3. **Explore `advanced_indexing_guide.md`** - Understand complex array access

### Advanced Path
1. **Master `linear_algebra_guide.md`** - Learn matrix operations and decompositions
2. **Study `random_generation_guide.md`** - Understand statistical sampling
3. **Run `random_generation.py`** - Practice with interactive examples
4. **Combine all techniques** - Apply NumPy to real-world problems

### Data Science Path
1. **Build foundation** - Complete all basic guides
2. **Focus on linear algebra** - Essential for machine learning
3. **Master random generation** - Critical for statistical analysis
4. **Practice with real data** - Apply NumPy to pandas DataFrames

## Key Concepts Covered

### Array Fundamentals
- **ndarray** - The core NumPy array object
- **Data Types** - Understanding dtype and memory layout
- **Broadcasting** - Automatic array alignment
- **Vectorization** - Fast element-wise operations

### Advanced Operations
- **Indexing** - Basic, boolean, and fancy indexing
- **Slicing** - Efficient array subsetting
- **Masking** - Conditional array operations
- **Structured Arrays** - Complex data types

### Mathematical Operations
- **Linear Algebra** - Matrix operations and decompositions
- **Statistical Functions** - Mean, std, correlation, etc.
- **Random Generation** - Probability distributions and sampling
- **Mathematical Functions** - Trigonometric, exponential, etc.

## Performance Tips

- **Use vectorized operations** instead of loops
- **Understand broadcasting** for efficient operations
- **Choose appropriate data types** for memory efficiency
- **Use in-place operations** when possible
- **Profile your code** with NumPy's built-in tools

## Integration with Other Libraries

### pandas Integration
```python
import numpy as np
import pandas as pd

# Convert NumPy arrays to pandas
df = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'C'])

# Use NumPy functions with pandas
result = np.mean(df.values, axis=0)
```

### Matplotlib Integration
```python
import numpy as np
import matplotlib.pyplot as plt

# Create data with NumPy
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plot with Matplotlib
plt.plot(x, y)
plt.show()
```

## Additional Resources

### Official Documentation
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [NumPy Reference](https://numpy.org/doc/stable/reference/index.html)
- [NumPy Tutorial](https://numpy.org/doc/stable/user/quickstart.html)

### Learning Resources
- [NumPy Cheat Sheet](https://numpy.org/doc/stable/user/quickstart.html)
- [NumPy GitHub Repository](https://github.com/numpy/numpy)
- [NumPy Community](https://numpy.org/community/)

### Recommended Books
- "Python for Data Analysis" by Wes McKinney
- "Numerical Python" by Robert Johansson
- "Learning NumPy Array" by Ivan Idris

## Contributing

Found an error or have a suggestion? Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

**Happy Scientific Computing!**
