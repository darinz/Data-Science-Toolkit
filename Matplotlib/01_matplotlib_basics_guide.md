# Matplotlib Basics: Complete Guide

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-blue.svg)](https://matplotlib.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

## Table of Contents
1. [Introduction](#introduction)
2. [Matplotlib Architecture](#matplotlib-architecture)
3. [Basic Plot Types](#basic-plot-types)
4. [Customizing Plots](#customizing-plots)
5. [Subplots and Layouts](#subplots-and-layouts)
6. [Saving and Exporting Plots](#saving-and-exporting-plots)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Introduction

Matplotlib is the most popular Python library for creating static, animated, and interactive visualizations. It is highly customizable and integrates well with NumPy and pandas.

### Key Features
- Line, scatter, bar, histogram, and more
- Full control over plot appearance
- Publication-quality output
- Works with Jupyter, scripts, and GUIs

## Matplotlib Architecture

Matplotlib uses a hierarchical object-oriented structure:
- **Figure**: The entire drawing area
- **Axes**: The plotting area (can be multiple per figure)
- **Artists**: Everything drawn (lines, text, etc.)

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y, label='sin(x)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Simple Sine Wave')
ax.legend()
plt.show()
```

## Basic Plot Types

### Line Plot
```python
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Line Plot')
plt.legend()
plt.show()
```

### Scatter Plot
```python
x = np.linspace(0, 10, 50)
y = np.sin(x) + np.random.normal(0, 0.1, 50)
plt.scatter(x, y, alpha=0.7)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot')
plt.show()
```

### Bar Plot
```python
categories = ['A', 'B', 'C', 'D']
values = [10, 15, 7, 12]
plt.bar(categories, values, color=['red', 'blue', 'green', 'orange'])
plt.title('Bar Plot')
plt.xlabel('Category')
plt.ylabel('Value')
plt.show()
```

### Histogram
```python
data = np.random.normal(0, 1, 1000)
plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

## Customizing Plots

### Colors, Styles, and Markers
```python
x = np.linspace(0, 4*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, color='red', linestyle='-', marker='o', label='sin(x)')
plt.plot(x, y2, color='blue', linestyle='--', marker='s', label='cos(x)')
plt.xlabel('x (radians)')
plt.ylabel('y')
plt.title('Trigonometric Functions')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
```

### Labels, Titles, and Annotations
```python
plt.plot(x, y1)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Sine Function')
plt.annotate('Peak', xy=(np.pi/2, 1), xytext=(2, 1.2),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
```

### Legends and Spines
```python
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.legend(loc='upper right', frameon=True, shadow=True)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
```

## Subplots and Layouts

### Simple Subplots
```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, y1)
axes[0, 0].set_title('sin(x)')
axes[0, 1].plot(x, y2)
axes[0, 1].set_title('cos(x)')
axes[1, 0].bar(categories, values)
axes[1, 0].set_title('Bar')
axes[1, 1].hist(data, bins=20)
axes[1, 1].set_title('Histogram')
plt.tight_layout()
plt.show()
```

### GridSpec for Complex Layouts
```python
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(8, 6))
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax1.plot(x, y1)
ax2.bar(categories, values)
ax3.hist(data, bins=20)
plt.tight_layout()
plt.show()
```

## Saving and Exporting Plots

### Save to File
```python
plt.plot(x, y1)
plt.title('Save Example')
plt.savefig('my_plot.png', dpi=300, bbox_inches='tight')
plt.close()
```

### Supported Formats
- PNG, JPG, SVG, PDF, EPS
- Use `plt.savefig('filename.format')`

## Best Practices
- Use `fig, ax = plt.subplots()` for flexibility
- Always label axes and add titles
- Use legends for clarity
- Use `plt.tight_layout()` to avoid overlap
- Save plots with high DPI for publication
- Use consistent styles for multiple plots

## Troubleshooting
- **Plot not showing**: Use `plt.show()` in scripts
- **Overlapping labels**: Use `plt.tight_layout()`
- **Exported image is blank**: Call `plt.savefig()` before `plt.close()`
- **Font or style issues**: Check `matplotlibrc` and installed fonts

## Resources
- [Matplotlib Documentation](https://matplotlib.org/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)

---

**Happy Plotting!**