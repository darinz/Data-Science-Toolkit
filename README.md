# Data Science Toolkit

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![Conda](https://img.shields.io/badge/Conda-Environment-green.svg)](https://docs.conda.io/)

A comprehensive collection of ultra-quick tutorials and practical examples for essential data science libraries in Python. This toolkit is designed to help you quickly get started with scientific computing, data analysis, and machine learning.

## What's Included

### NumPy Tutorials
- **Ultra-quick NumPy tutorial** for scientific computing
- Essential array operations and mathematical functions
- Perfect foundation for machine learning and data science
- Interactive Jupyter notebooks with practical examples

### pandas Tutorials  
- **Ultra-quick pandas DataFrame tutorial** for data analysis
- Data manipulation and analysis techniques
- Essential skills for data science workflows
- Real-world examples and best practices

## Quick Start

### Prerequisites
- Python 3.10 or higher
- Basic understanding of Python programming
- Conda or pip package manager

### Installation Options

#### Option 1: Google Colab (Recommended for Beginners)
- Open the tutorial notebooks directly in [Google Colab](https://colab.research.google.com/)
- No local setup required
- Free GPU access available

#### Option 2: Local Environment Setup

**Using Conda (Recommended):**
```bash
# Clone the repository
git clone https://github.com/yourusername/Data-Science-Toolkit.git
cd Data-Science-Toolkit

# Create environment for NumPy tutorials
cd NumPy
conda env create -f environment.yml
conda activate ml

# Create environment for pandas tutorials  
cd ../pandas
conda env create -f environment.yml
conda activate ml
```

**Using pip:**
```bash
# Install dependencies for NumPy
cd NumPy
pip install -r requirements.txt

# Install dependencies for pandas
cd ../pandas  
pip install -r requirements.txt
```

## Tutorial Structure

### NumPy Module
- **File:** `numpy_ultraquick_tutorial.ipynb`
- **Dependencies:** NumPy, pandas, JupyterLab
- **Focus:** Scientific computing fundamentals

### pandas Module  
- **File:** `pandas_df_ultraquick_tutorial.ipynb`
- **Dependencies:** NumPy, pandas, JupyterLab
- **Focus:** Data analysis and manipulation

## Environment Management

### Conda Commands
```bash
# Create new environment
conda env create -f environment.yml

# Activate environment
conda activate ml

# Update environment
conda env update -f environment.yml --prune

# Remove environment
conda remove --name ml --all

# List environments
conda env list
```

### Jupyter Commands
```bash
# Start Jupyter Lab
jupyter lab

# Start Jupyter Notebook
jupyter notebook
```

## Learning Path

1. **Start with NumPy** - Build your foundation in scientific computing
2. **Move to pandas** - Learn data manipulation and analysis
3. **Practice with Examples** - Work through the interactive notebooks
4. **Apply to Real Projects** - Use these skills in your own data science projects

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [NumPy](https://numpy.org/) - The fundamental package for scientific computing
- [pandas](https://pandas.pydata.org/) - Data analysis and manipulation library
- [Jupyter](https://jupyter.org/) - Interactive computing platform
- [Google Colab](https://colab.research.google.com/) - Free cloud-based Jupyter environment

## Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the individual module README files for specific guidance
- Refer to the official documentation for [NumPy](https://numpy.org/doc/) and [pandas](https://pandas.pydata.org/docs/)

---

**Happy Learning!**
