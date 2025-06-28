# pandas Tutorials for Data Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Conda](https://img.shields.io/badge/Conda-Environment-green.svg)](https://docs.conda.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

Pandas is a fast, powerful, flexible, and easy-to-use open-source data analysis and manipulation library built on top of Python. It provides data structures for efficiently storing large datasets and tools for data cleaning, transformation, and analysis.

This module contains ultra-quick tutorials designed to help you master the essential aspects of pandas DataFrames needed to kickstart your journey in Machine Learning, Deep Learning, and Data Science.

## What You'll Learn

- **DataFrame Creation & Manipulation** - Create and modify pandas DataFrames
- **Data Loading & Export** - Read from and write to various file formats
- **Data Cleaning & Preprocessing** - Handle missing values, duplicates, and data types
- **Data Filtering & Selection** - Filter and select data using various methods
- **Grouping & Aggregation** - Group data and perform statistical operations
- **Data Visualization** - Create basic plots and charts
- **Time Series Analysis** - Work with date and time data
- **Data Merging & Joining** - Combine datasets from multiple sources

## Prerequisites

Before starting this tutorial, ensure you have:

- **Python 3.10+** installed on your system
- Basic understanding of [Python programming](https://www.python.org/doc/)
- [Pandas](https://pandas.pydata.org/getting_started.html) installed or access to [Google Colab](https://colab.research.google.com/)

## Getting Started

### Option 1: Google Colab (Recommended for Beginners)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the `pandas_df_ultraquick_tutorial.ipynb` file
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
pandas/
├── pandas_df_ultraquick_tutorial.ipynb  # Main tutorial notebook
├── environment.yml                      # Conda environment configuration
├── requirements.txt                     # pip dependencies
└── README.md                           # This file
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

The `pandas_df_ultraquick_tutorial.ipynb` notebook covers:

1. **Introduction to Pandas**
   - Series and DataFrame objects
   - Basic data structures and concepts

2. **Data Loading & Export**
   - Reading CSV, Excel, JSON files
   - Writing data to various formats
   - Database connections

3. **Data Exploration**
   - Viewing data structure and information
   - Basic statistics and summaries
   - Data types and memory usage

4. **Data Selection & Indexing**
   - Column and row selection
   - Boolean indexing
   - Loc and iloc methods

5. **Data Cleaning**
   - Handling missing values
   - Removing duplicates
   - Data type conversion
   - String operations

6. **Data Manipulation**
   - Adding/removing columns
   - Sorting and ranking
   - Reshaping data (melt, pivot)

7. **Grouping & Aggregation**
   - GroupBy operations
   - Aggregation functions
   - Multi-level grouping

8. **Data Merging & Joining**
   - Concatenation
   - Merge and join operations
   - Handling different join types

9. **Time Series Analysis**
   - Date and time handling
   - Time-based indexing
   - Resampling and frequency conversion

10. **Data Visualization**
    - Basic plotting with pandas
    - Statistical plots
    - Customizing visualizations

## Learning Resources

### Official Documentation
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [Pandas API Reference](https://pandas.pydata.org/docs/reference/index.html)
- [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/index.html)

### Additional Resources
- [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
- [Pandas GitHub Repository](https://github.com/pandas-dev/pandas)
- [Pandas Community](https://pandas.pydata.org/community/)

### Recommended Books
- "Python for Data Analysis" by Wes McKinney (creator of pandas)
- "Pandas Cookbook" by Theodore Petrou

## Contributing

Found an error or have a suggestion? Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## Acknowledgments

- [Wes McKinney](https://wesmckinney.com/) for creating pandas
- [Pandas Development Team](https://pandas.pydata.org/about/) for continuous development
- [NumPy Team](https://numpy.org/) for the foundation library
- [Jupyter Project](https://jupyter.org/) for the interactive computing platform

---

**Ready to master data analysis? Start with the pandas tutorial!**
