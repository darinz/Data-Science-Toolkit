# Matplotlib Plot Types: Complete Guide

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-blue.svg)](https://matplotlib.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

## Table of Contents
1. [Introduction](#introduction)
2. [Line Plots](#line-plots)
3. [Scatter Plots](#scatter-plots)
4. [Bar Charts](#bar-charts)
5. [Histograms](#histograms)
6. [Pie Charts](#pie-charts)
7. [Area Plots](#area-plots)
8. [Error Bars](#error-bars)
9. [Confidence Intervals](#confidence-intervals)
10. [Combining Plot Types](#combining-plot-types)

## Introduction

Matplotlib offers a wide variety of plot types for different data visualization needs. This guide covers the most commonly used plot types with practical examples.

## Line Plots

### Basic Line Plot
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Basic Line Plot')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Multiple Lines
```python
x = np.linspace(0, 4*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x + np.pi/4)

plt.plot(x, y1, label='sin(x)', linewidth=2)
plt.plot(x, y2, label='cos(x)', linewidth=2)
plt.plot(x, y3, label='sin(x + π/4)', linewidth=2)
plt.xlabel('x (radians)')
plt.ylabel('y')
plt.title('Multiple Trigonometric Functions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Line Styles and Markers
```python
x = np.linspace(0, 10, 20)
y = np.exp(-x/3)

plt.plot(x, y, 'ro-', label='Exponential decay', linewidth=2, markersize=8)
plt.plot(x, y*0.5, 'bs--', label='Half decay', linewidth=2, markersize=8)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Line Styles and Markers')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Scatter Plots

### Basic Scatter Plot
```python
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)

plt.scatter(x, y, alpha=0.6)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Basic Scatter Plot')
plt.grid(True, alpha=0.3)
plt.show()
```

### Scatter Plot with Color Mapping
```python
x = np.random.randn(100)
y = np.random.randn(100)
colors = np.random.rand(100)
sizes = 1000 * np.random.rand(100)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
plt.colorbar(label='Color value')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot with Color Mapping')
plt.show()
```

### Categorical Scatter Plot
```python
categories = ['A', 'B', 'C', 'D']
colors = ['red', 'blue', 'green', 'orange']

for i, (cat, color) in enumerate(zip(categories, colors)):
    x = np.random.normal(i, 0.3, 20)
    y = np.random.normal(0, 1, 20)
    plt.scatter(x, y, c=color, label=cat, alpha=0.7)

plt.xlabel('Category')
plt.ylabel('Value')
plt.title('Categorical Scatter Plot')
plt.legend()
plt.xticks(range(len(categories)), categories)
plt.show()
```

## Bar Charts

### Vertical Bar Chart
```python
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

plt.bar(categories, values, color='skyblue', edgecolor='black')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Vertical Bar Chart')
plt.show()
```

### Horizontal Bar Chart
```python
plt.barh(categories, values, color='lightcoral', edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Categories')
plt.title('Horizontal Bar Chart')
plt.show()
```

### Grouped Bar Chart
```python
categories = ['A', 'B', 'C', 'D']
group1 = [20, 35, 30, 35]
group2 = [25, 32, 34, 20]

x = np.arange(len(categories))
width = 0.35

plt.bar(x - width/2, group1, width, label='Group 1', color='skyblue')
plt.bar(x + width/2, group2, width, label='Group 2', color='lightcoral')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Grouped Bar Chart')
plt.xticks(x, categories)
plt.legend()
plt.show()
```

### Stacked Bar Chart
```python
categories = ['A', 'B', 'C', 'D']
bottom = [20, 35, 30, 35]
top = [25, 32, 34, 20]

plt.bar(categories, bottom, label='Bottom', color='skyblue')
plt.bar(categories, top, bottom=bottom, label='Top', color='lightcoral')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Stacked Bar Chart')
plt.legend()
plt.show()
```

## Histograms

### Basic Histogram
```python
data = np.random.normal(0, 1, 1000)
plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Basic Histogram')
plt.grid(True, alpha=0.3)
plt.show()
```

### Histogram with Multiple Datasets
```python
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(2, 1.5, 1000)

plt.hist(data1, bins=30, alpha=0.7, label='Dataset 1', color='skyblue')
plt.hist(data2, bins=30, alpha=0.7, label='Dataset 2', color='lightcoral')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram with Multiple Datasets')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Cumulative Histogram
```python
data = np.random.normal(0, 1, 1000)
plt.hist(data, bins=30, cumulative=True, alpha=0.7, color='green')
plt.xlabel('Value')
plt.ylabel('Cumulative Frequency')
plt.title('Cumulative Histogram')
plt.grid(True, alpha=0.3)
plt.show()
```

## Pie Charts

### Basic Pie Chart
```python
sizes = [30, 25, 20, 15, 10]
labels = ['A', 'B', 'C', 'D', 'E']
colors = ['lightcoral', 'lightblue', 'lightgreen', 'yellow', 'orange']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Basic Pie Chart')
plt.axis('equal')
plt.show()
```

### Pie Chart with Exploded Slices
```python
explode = (0.1, 0, 0, 0, 0)  # Explode the first slice
plt.pie(sizes, explode=explode, labels=labels, colors=colors, 
        autopct='%1.1f%%', startangle=90)
plt.title('Pie Chart with Exploded Slice')
plt.axis('equal')
plt.show()
```

### Donut Chart
```python
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax.add_artist(plt.Circle((0, 0), 0.7, fc='white'))
plt.title('Donut Chart')
plt.show()
```

## Area Plots

### Basic Area Plot
```python
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.fill_between(x, y, alpha=0.3, color='skyblue')
plt.plot(x, y, color='blue', linewidth=2)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Area Plot')
plt.grid(True, alpha=0.3)
plt.show()
```

### Stacked Area Plot
```python
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.fill_between(x, y1, alpha=0.3, label='sin(x)', color='skyblue')
plt.fill_between(x, y2, alpha=0.3, label='cos(x)', color='lightcoral')
plt.plot(x, y1, color='blue', linewidth=1)
plt.plot(x, y2, color='red', linewidth=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Stacked Area Plot')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Error Bars

### Line Plot with Error Bars
```python
x = np.linspace(0, 10, 10)
y = np.sin(x)
yerr = 0.1 * np.abs(np.cos(x))  # Error proportional to derivative

plt.errorbar(x, y, yerr=yerr, fmt='o-', capsize=5, capthick=2)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Line Plot with Error Bars')
plt.grid(True, alpha=0.3)
plt.show()
```

### Scatter Plot with Error Bars
```python
x = np.linspace(0, 10, 10)
y = np.sin(x) + np.random.normal(0, 0.1, 10)
xerr = 0.2 * np.ones_like(x)
yerr = 0.1 * np.ones_like(y)

plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', capsize=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot with Error Bars')
plt.grid(True, alpha=0.3)
plt.show()
```

## Confidence Intervals

### Confidence Interval for Mean
```python
from scipy import stats

# Generate sample data
np.random.seed(42)
data = np.random.normal(0, 1, 100)

# Calculate confidence interval
confidence = 0.95
mean = np.mean(data)
sem = stats.sem(data)
ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=sem)

# Plot
plt.hist(data, bins=20, alpha=0.7, color='skyblue', density=True)
plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.3f}')
plt.axvline(ci[0], color='green', linestyle=':', label=f'CI: [{ci[0]:.3f}, {ci[1]:.3f}]')
plt.axvline(ci[1], color='green', linestyle=':')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title(f'{confidence*100}% Confidence Interval')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Confidence Bands for Regression
```python
from scipy import stats

x = np.linspace(0, 10, 20)
y_true = 2 * x + 1
y = y_true + np.random.normal(0, 1, 20)

# Linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
y_pred = slope * x + intercept

# Calculate confidence interval
n = len(x)
x_mean = np.mean(x)
se = std_err * np.sqrt(1/n + (x - x_mean)**2 / np.sum((x - x_mean)**2))
ci = stats.t.interval(0.95, n-2, loc=y_pred, scale=se)

plt.scatter(x, y, alpha=0.7, label='Data')
plt.plot(x, y_pred, 'r-', label='Regression line')
plt.fill_between(x, ci[0], ci[1], alpha=0.3, color='red', label='95% CI')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regression with Confidence Interval')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## Combining Plot Types

### Mixed Plot Types
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left subplot: Line and scatter
x = np.linspace(0, 10, 50)
y = np.sin(x)
noise = np.random.normal(0, 0.1, 50)

ax1.plot(x, y, 'b-', label='True function', linewidth=2)
ax1.scatter(x, y + noise, c='red', alpha=0.6, label='Noisy data')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Line and Scatter')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right subplot: Bar and error bars
categories = ['A', 'B', 'C', 'D']
values = [10, 15, 7, 12]
errors = [1, 2, 1.5, 1]

ax2.bar(categories, values, yerr=errors, capsize=5, alpha=0.7)
ax2.set_xlabel('Category')
ax2.set_ylabel('Value')
ax2.set_title('Bar with Error Bars')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Complex Visualization
```python
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Top left: Line plot
x = np.linspace(0, 4*np.pi, 100)
ax1.plot(x, np.sin(x), label='sin(x)')
ax1.plot(x, np.cos(x), label='cos(x)')
ax1.set_title('Trigonometric Functions')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Top right: Scatter plot
np.random.seed(42)
x_scatter = np.random.randn(50)
y_scatter = np.random.randn(50)
colors = np.random.rand(50)
ax2.scatter(x_scatter, y_scatter, c=colors, alpha=0.6)
ax2.set_title('Scatter Plot')
ax2.grid(True, alpha=0.3)

# Bottom left: Bar chart
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]
ax3.bar(categories, values, color='skyblue')
ax3.set_title('Bar Chart')
ax3.set_ylabel('Values')

# Bottom right: Histogram
data = np.random.normal(0, 1, 1000)
ax4.hist(data, bins=30, alpha=0.7, color='lightcoral')
ax4.set_title('Histogram')
ax4.set_xlabel('Value')
ax4.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

## Best Practices

1. **Choose the right plot type** for your data and question
2. **Use appropriate colors** and ensure accessibility
3. **Label everything** clearly (axes, titles, legends)
4. **Consider your audience** when choosing complexity
5. **Use consistent styling** across related plots
6. **Test with different data** to ensure robustness

## Resources

- [Matplotlib Plot Types](https://matplotlib.org/stable/plot_types/index.html)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Choosing the Right Chart](https://www.data-to-viz.com/)

---

**Master the art of data visualization!** 📊 