# NumPy Tutorials for Scientific Computing

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Conda](https://img.shields.io/badge/Conda-Environment-green.svg)](https://docs.conda.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

NumPy is the fundamental package for scientific computing in Python. It provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.

This module contains ultra-quick tutorials designed to help you master the essential aspects of NumPy needed to kickstart your journey in Machine Learning and Data Science.

## What You'll Learn

- **Array Creation & Manipulation** - Create and modify NumPy arrays efficiently
- **Mathematical Operations** - Perform fast vectorized computations
- **Indexing & Slicing** - Access and modify array elements
- **Shape Manipulation** - Reshape, transpose, and concatenate arrays
- **Statistical Functions** - Calculate means, standard deviations, and more
- **Random Number Generation** - Create random arrays for simulations
- **Linear Algebra Operations** - Matrix operations and decompositions

## Prerequisites

Before starting this tutorial, ensure you have:

- **Python 3.10+** installed on your system
- Basic understanding of [Python programming](https://www.python.org/doc/)
- [NumPy](https://numpy.org/install/) installed or access to [Google Colab](https://colab.research.google.com/)

## Getting Started

### Option 1: Google Colab (Recommended for Beginners)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the `numpy_ultraquick_tutorial.ipynb` file
3. Start learning immediately - no setup required!

**Pro Tips for Colab:**
- Use GPU runtime for faster computations
- Save your work to Google Drive
- Share notebooks easily with others

### Option 2: Local Environment Setup

#### Using Conda (Recommended)

```bash
# Create a new conda environment
conda env create -f environment.yml

# Activate the environment
conda activate ml

# Launch Jupyter Lab
jupyter lab
```

#### Using pip

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Lab
jupyter lab
```

## File Structure

```
NumPy/
├── numpy_ultraquick_tutorial.ipynb  # Main tutorial notebook
├── environment.yml                  # Conda environment configuration
├── requirements.txt                 # pip dependencies
└── README.md                       # This file
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
```

## Tutorial Content

The `numpy_ultraquick_tutorial.ipynb` notebook covers:

1. **Introduction to NumPy Arrays**
   - Creating arrays from lists, ranges, and functions
   - Array attributes and properties

2. **Array Operations**
   - Element-wise operations
   - Broadcasting rules
   - Mathematical functions

3. **Indexing and Slicing**
   - Basic indexing
   - Boolean indexing
   - Fancy indexing

4. **Array Manipulation**
   - Reshaping arrays
   - Concatenation and splitting
   - Transposing and swapping axes

5. **Statistical Operations**
   - Descriptive statistics
   - Aggregation functions
   - Percentiles and quantiles

6. **Random Number Generation**
   - Random arrays
   - Probability distributions
   - Seeding for reproducibility

## Learning Resources

### Official Documentation
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html)
- [NumPy Reference](https://numpy.org/doc/stable/reference/)
- [NumPy Tutorial](https://numpy.org/doc/stable/user/quickstart.html)

### Additional Resources
- [NumPy Cheat Sheet](https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf)
- [NumPy GitHub Repository](https://github.com/numpy/numpy)
- [NumPy Community](https://numpy.org/community/)

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
- [Jupyter Project](https://jupyter.org/) for the interactive computing platform

---

**Ready to dive into scientific computing? Start with the tutorial notebook!**
