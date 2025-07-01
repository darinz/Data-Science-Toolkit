# Jupyter Basics: Complete Guide

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![JupyterLab](https://img.shields.io/badge/JupyterLab-4.0+-blue.svg)](https://jupyterlab.readthedocs.io/)
[![IPython](https://img.shields.io/badge/IPython-8.0+-blue.svg)](https://ipython.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-blue.svg)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Getting Started](#getting-started)
4. [Cell Types](#cell-types)
5. [Basic Operations](#basic-operations)
6. [Keyboard Shortcuts](#keyboard-shortcuts)
7. [Working with Kernels](#working-with-kernels)
8. [Markdown Documentation](#markdown-documentation)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Introduction

Jupyter is an open-source web application for creating interactive documents with live code, equations, visualizations, and narrative text. It's essential for data science, machine learning, and scientific computing.

### Key Benefits
- **Interactive Computing**: Execute code and see results immediately
- **Rich Media Support**: Embed plots, images, and interactive visualizations
- **Documentation**: Combine code and explanations seamlessly
- **Reproducible Research**: Share complete analysis workflows
- **Multi-language Support**: Python, R, Julia, JavaScript, and more

## Installation

### Quick Setup
```bash
# Using conda (recommended)
conda create -n jupyter-env python=3.9
conda activate jupyter-env
conda install jupyter jupyterlab

# Using pip
pip install jupyter jupyterlab

# Using project environment
cd Jupyter
conda env create -f environment.yml
conda activate jupyter-tutorials
```

### Starting Jupyter
```bash
# Start JupyterLab (modern interface)
jupyter lab

# Start Jupyter Notebook (classic interface)
jupyter notebook

# Start with specific port
jupyter lab --port=8888 --no-browser
```

## Getting Started

### Creating Your First Notebook

1. **Launch JupyterLab**: Run `jupyter lab` in terminal
2. **Create New Notebook**: Click "+" → "Python 3"
3. **Add Content**: Mix code and markdown cells

### Basic Notebook Structure
```python
# Cell 1: Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cell 2: Data loading
df = pd.read_csv('data.csv')
print(f"Dataset shape: {df.shape}")

# Cell 3: Analysis
df.describe()

# Cell 4: Visualization
plt.figure(figsize=(10, 6))
df.hist()
plt.show()
```

## Cell Types

### 1. Code Cells
Execute Python code and display output:

```python
# Example code cell
import numpy as np
import matplotlib.pyplot as plt

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create plot
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Sine Wave')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.grid(True, alpha=0.3)
plt.show()
```

### 2. Markdown Cells
Write formatted documentation:

```markdown
# Data Analysis Report

## Introduction
This notebook demonstrates basic data analysis techniques.

### Key Findings:
- Dataset contains 1000 records
- 5 numerical features
- No missing values detected

Mathematical expression: $f(x) = \sin(x)$

[View documentation](https://jupyter.org/)
```

### 3. Raw Cells
Unprocessed text (rarely used):

```raw
This is raw text that passes through unchanged.
```

## Basic Operations

### Cell Execution
```python
# Run current cell and select next
Shift + Enter

# Run current cell and stay
Ctrl + Enter

# Run current cell and insert below
Alt + Enter
```

### Cell Management
```python
# Insert cell above
A (in command mode)

# Insert cell below
B (in command mode)

# Delete cell
D, D (in command mode)

# Merge cells
Shift + M (in command mode)
```

### Saving and Exporting
```bash
# Save notebook
Ctrl + S

# Export to different formats
jupyter nbconvert notebook.ipynb --to html
jupyter nbconvert notebook.ipynb --to pdf
jupyter nbconvert notebook.ipynb --to python
```

## Keyboard Shortcuts

### Command Mode (press Esc)
| Shortcut | Action |
|----------|--------|
| `A` | Insert cell above |
| `B` | Insert cell below |
| `D, D` | Delete cell |
| `Z` | Undo cell deletion |
| `Shift + M` | Merge cells |
| `Ctrl + S` | Save notebook |

### Edit Mode (press Enter)
| Shortcut | Action |
|----------|--------|
| `Shift + Enter` | Run cell, select below |
| `Ctrl + Enter` | Run cell, stay |
| `Alt + Enter` | Run cell, insert below |
| `Tab` | Code completion |
| `Shift + Tab` | Show documentation |

### View All Shortcuts
Press `H` in command mode to see all available shortcuts.

## Working with Kernels

### Understanding Kernels
A kernel is the computational engine that executes code. Different kernels support different programming languages.

### Kernel Operations
```python
# Check current kernel
import sys
print(f"Python: {sys.version}")
print(f"Kernel: {sys.executable}")

# Restart kernel
import IPython
IPython.get_ipython().kernel.do_shutdown(True)

# List available kernels
!jupyter kernelspec list
```

### Multi-language Support
```python
# Python cell
import numpy as np
data = np.random.randn(100)

# R cell (if R kernel installed)
%%R
library(ggplot2)
ggplot(data.frame(x = 1:100, y = rnorm(100)), aes(x, y)) + geom_point()

# JavaScript cell (if JS kernel installed)
%%javascript
console.log("Hello from JavaScript!");
```

## Markdown Documentation

### Basic Markdown
```markdown
# Main Title
## Section Title
### Subsection Title

**Bold text** and *italic text*

- Bullet point 1
- Bullet point 2

1. Numbered list item 1
2. Numbered list item 2

[Link text](https://example.com)
![Image alt text](image.png)
```

### Mathematical Equations
```markdown
# Inline math: $E = mc^2$

# Block math:
$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$

# Complex equations:
$$
\begin{align}
y &= mx + b \\
&= 2x + 3
\end{align}
$$
```

### Code and Tables
```markdown
# Inline code: `print("Hello, World!")`

# Code blocks:
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

# Tables:
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |
```

### Interactive Elements
```python
# Create interactive documentation
from IPython.display import HTML, display

html_content = """
<details>
<summary><strong>Click to expand: Advanced Configuration</strong></summary>
<div style="padding: 10px; background-color: #f9f9f9;">
    <h4>Configuration Options</h4>
    <ul>
        <li>Option 1: Enable feature X</li>
        <li>Option 2: Configure setting Y</li>
    </ul>
</div>
</details>
"""

display(HTML(html_content))
```

## Best Practices

### Notebook Organization
```python
# Recommended structure:

# 1. Title and Introduction (Markdown)
"""
# Project Title
## Overview
Brief description of the project.
"""

# 2. Setup and Imports (Code)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 3. Data Loading (Code)
df = pd.read_csv('data.csv')

# 4. Data Exploration (Code + Markdown)
# Explore data structure

# 5. Analysis (Code + Markdown)
# Perform analysis

# 6. Results (Markdown)
# Summarize findings
```

### Code Quality
```python
# Good practices:

# 1. Descriptive variable names
user_data = pd.read_csv('users.csv')  # Good
ud = pd.read_csv('users.csv')         # Bad

# 2. Add comments
# Calculate engagement score
engagement_score = (frequency * 0.6 + duration * 0.4)

# 3. Use functions
def calculate_score(freq, dur, weights=(0.6, 0.4)):
    """Calculate engagement score."""
    return freq * weights[0] + dur * weights[1]

# 4. Error handling
try:
    result = risky_operation()
except Exception as e:
    print(f"Error: {e}")
    result = None

# 5. Type hints
from typing import List, Dict, Optional

def process_data(data: List[Dict]) -> Optional[pd.DataFrame]:
    """Process data into DataFrame."""
    if not data:
        return None
    return pd.DataFrame(data)
```

### Documentation Standards
```markdown
# Documentation Best Practices

## 1. Clear Structure
- Use consistent headings
- Group related sections
- Include table of contents

## 2. Explain Process
- Document methodology
- Explain choices
- Note assumptions

## 3. Include Context
- Explain code purpose
- Provide background
- Reference sources

## 4. Make Reproducible
- Include setup instructions
- Specify versions
- Provide sample data
```

### Version Control
```bash
# Git configuration for notebooks
# .gitattributes
*.ipynb filter=strip-notebook-output

# Git filter
git config --global filter.strip-notebook-output.clean \
    "jupyter nbconvert --clear-output --stdin --stdout --log-level=ERROR"

# Pre-commit hooks
pip install pre-commit nbstripout
pre-commit install
```

## Troubleshooting

### Common Issues

#### 1. Kernel Won't Start
```bash
# Check kernel installation
jupyter kernelspec list

# Reinstall kernel
python -m ipykernel install --user --name=myenv

# Check Python path
which python
python --version
```

#### 2. Import Errors
```python
# Check installed packages
!pip list | grep package_name

# Install missing packages
!pip install package_name

# Check Python path
import sys
print(sys.path)
```

#### 3. Memory Issues
```python
# Monitor memory usage
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory: {get_memory_usage():.2f} MB")

# Clear variables
del large_variable
import gc
gc.collect()
```

#### 4. Display Issues
```python
# Fix matplotlib display
%matplotlib inline

# For interactive plots
%matplotlib notebook

# For non-interactive
import matplotlib
matplotlib.use('Agg')
```

#### 5. Performance
```python
# Profile code
%timeit your_function()

# Line profiler
%load_ext line_profiler
%lprun -f your_function your_function()

# Memory profiler
%load_ext memory_profiler
%mprun -f your_function your_function()
```

### Getting Help
```python
# Built-in help
help(pd.DataFrame)

# Object introspection
df?

# Source code
df.head??

# Search functions
pd.*read*?

# Documentation
import webbrowser
webbrowser.open('https://pandas.pydata.org/docs/')
```

### Debugging
```python
# Debugging in Jupyter
import pdb

def debug_function():
    x = 1
    pdb.set_trace()  # Breakpoint
    y = x + 1
    return y

# Interactive debugging
from IPython.core.debugger import set_trace
set_trace()

# Error debugging
try:
    error_function()
except:
    %debug
```

## Conclusion

This guide covers Jupyter fundamentals. As you progress, explore:

- Interactive widgets and visualizations
- Custom magic commands
- JupyterLab extensions
- Multi-language kernels
- Production deployment

### Resources
- [Jupyter Documentation](https://jupyter.org/)
- [JupyterLab User Guide](https://jupyterlab.readthedocs.io/)
- [IPython Documentation](https://ipython.readthedocs.io/)
- [Jupyter Widgets](https://ipywidgets.readthedocs.io/)

---

**Happy Interactive Computing!** 🚀