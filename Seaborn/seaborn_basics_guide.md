# Seaborn Basics Guide

A comprehensive introduction to Seaborn, the statistical data visualization library that makes beautiful plots with minimal code.

## Table of Contents

1. [Introduction to Seaborn](#introduction-to-seaborn)
2. [Installation and Setup](#installation-and-setup)
3. [Basic Plotting](#basic-plotting)
4. [Figure Aesthetics](#figure-aesthetics)
5. [Color Palettes](#color-palettes)
6. [Statistical Relationships](#statistical-relationships)
7. [Categorical Data](#categorical-data)
8. [Distribution Plots](#distribution-plots)
9. [Working with Data](#working-with-data)
10. [Best Practices](#best-practices)

## Introduction to Seaborn

Seaborn is a Python data visualization library based on Matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics. It's particularly well-suited for data science and statistical analysis.

### Key Features

- **Statistical Focus**: Built specifically for statistical data visualization
- **Beautiful Defaults**: Attractive plots with minimal configuration
- **Pandas Integration**: Seamless work with pandas DataFrames
- **Statistical Annotations**: Built-in statistical testing and annotations
- **Flexible Styling**: Easy customization and theming

### When to Use Seaborn

- **Exploratory Data Analysis**: Quick insights into data distributions and relationships
- **Statistical Visualization**: Plots that show statistical relationships and patterns
- **Publication Graphics**: High-quality plots for papers and presentations
- **Data Science Workflows**: Integration with pandas and statistical libraries

## Installation and Setup

### Basic Installation

```python
# Import required libraries
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up the plotting style
sns.set_theme(style="whitegrid")

# Verify installation
print(f"Seaborn version: {sns.__version__}")
```

### Setting Up the Environment

```python
# Set the style for all plots
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)

# Alternative styles
# sns.set_style("whitegrid")  # Clean grid background
# sns.set_style("darkgrid")   # Dark grid background
# sns.set_style("white")      # Plain white background
# sns.set_style("dark")       # Dark background
# sns.set_style("ticks")      # Minimal ticks style
```

## Basic Plotting

### Simple Line Plot

```python
# Create sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a simple line plot
plt.figure(figsize=(10, 6))
sns.lineplot(x=x, y=y)
plt.title('Simple Sine Wave')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

### Scatter Plot with Regression

```python
# Generate sample data
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

# Create scatter plot with regression line
plt.figure(figsize=(10, 6))
sns.regplot(x=x, y=y, scatter_kws={'alpha': 0.6})
plt.title('Scatter Plot with Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

### Histogram with KDE

```python
# Generate sample data
data = np.random.randn(1000)

# Create histogram with kernel density estimation
plt.figure(figsize=(10, 6))
sns.histplot(data=data, kde=True, bins=30)
plt.title('Histogram with Kernel Density Estimation')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

## Figure Aesthetics

### Setting the Style

```python
# Set the overall style
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)

# Create a sample plot to see the style
data = np.random.randn(100)
plt.figure(figsize=(10, 6))
sns.histplot(data=data, kde=True)
plt.title('Styled Histogram')
plt.show()
```

### Customizing Plot Appearance

```python
# Create a more customized plot
plt.figure(figsize=(12, 8))

# Generate data
x = np.random.randn(200)
y = np.random.randn(200)

# Create scatter plot with custom styling
sns.scatterplot(x=x, y=y, alpha=0.6, s=100, color='steelblue')

# Customize the plot
plt.title('Customized Scatter Plot', fontsize=16, fontweight='bold')
plt.xlabel('X Axis', fontsize=12)
plt.ylabel('Y Axis', fontsize=12)
plt.grid(True, alpha=0.3)

# Add a trend line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x, p(x), "r--", alpha=0.8, linewidth=2)

plt.tight_layout()
plt.show()
```

### Context Settings

```python
# Different context settings for different use cases
contexts = ["paper", "notebook", "talk", "poster"]

for context in contexts:
    sns.set_context(context)
    plt.figure(figsize=(8, 6))
    
    # Create sample plot
    data = np.random.randn(100)
    sns.histplot(data=data, kde=True)
    plt.title(f'Context: {context}')
    plt.show()

# Reset to default
sns.set_context("notebook")
```

## Color Palettes

### Built-in Color Palettes

```python
# Available color palettes
palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']

# Create a figure to showcase different palettes
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for i, palette in enumerate(palettes):
    # Set the palette
    sns.set_palette(palette)
    
    # Create sample data
    data = np.random.randn(100)
    
    # Create histogram
    sns.histplot(data=data, kde=True, ax=axes[i])
    axes[i].set_title(f'Palette: {palette}')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

### Custom Color Palettes

```python
# Create custom color palettes
custom_palette = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]

# Set custom palette
sns.set_palette(custom_palette)

# Create a categorical plot to show the palette
categories = ['A', 'B', 'C', 'D', 'E']
values = np.random.rand(5)

plt.figure(figsize=(10, 6))
sns.barplot(x=categories, y=values)
plt.title('Custom Color Palette')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.show()
```

### Diverging and Sequential Palettes

```python
# Diverging palette (good for data with a meaningful center)
plt.figure(figsize=(15, 5))

# Diverging palette
plt.subplot(1, 3, 1)
data_diverging = np.random.randn(100)
sns.histplot(data=data_diverging, kde=True, color='RdBu_r')
plt.title('Diverging Palette (RdBu_r)')

# Sequential palette (good for continuous data)
plt.subplot(1, 3, 2)
data_sequential = np.random.exponential(1, 100)
sns.histplot(data=data_sequential, kde=True, color='Blues')
plt.title('Sequential Palette (Blues)')

# Qualitative palette (good for categorical data)
plt.subplot(1, 3, 3)
categories = ['A', 'B', 'C', 'D']
values = np.random.rand(4)
sns.barplot(x=categories, y=values, palette='Set3')
plt.title('Qualitative Palette (Set3)')

plt.tight_layout()
plt.show()
```

## Statistical Relationships

### Regression Plots

```python
# Generate sample data with different relationships
np.random.seed(42)

# Linear relationship
x1 = np.random.randn(100)
y1 = 2 * x1 + np.random.randn(100) * 0.5

# Non-linear relationship
x2 = np.random.randn(100)
y2 = x2**2 + np.random.randn(100) * 0.5

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Linear regression
sns.regplot(x=x1, y=y1, ax=axes[0], scatter_kws={'alpha': 0.6})
axes[0].set_title('Linear Relationship')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')

# Polynomial regression
sns.regplot(x=x2, y=y2, ax=axes[1], order=2, scatter_kws={'alpha': 0.6})
axes[1].set_title('Polynomial Relationship (Order 2)')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')

plt.tight_layout()
plt.show()
```

### Residual Plots

```python
# Create residual plot to check model assumptions
plt.figure(figsize=(10, 6))

# Generate data
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

# Create residual plot
sns.residplot(x=x, y=y, scatter_kws={'alpha': 0.6})
plt.title('Residual Plot')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)
plt.show()
```

## Categorical Data

### Bar Plots

```python
# Create sample categorical data
categories = ['A', 'B', 'C', 'D', 'E']
values = np.random.rand(5)

plt.figure(figsize=(10, 6))
sns.barplot(x=categories, y=values, palette='viridis')
plt.title('Bar Plot')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.show()
```

### Count Plots

```python
# Create sample categorical data
np.random.seed(42)
categories = np.random.choice(['Red', 'Blue', 'Green', 'Yellow'], 100)

plt.figure(figsize=(10, 6))
sns.countplot(data=pd.DataFrame({'Color': categories}), x='Color')
plt.title('Count Plot')
plt.xlabel('Color')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
```

### Box Plots

```python
# Create sample data with groups
np.random.seed(42)
group1 = np.random.normal(0, 1, 50)
group2 = np.random.normal(2, 1.5, 50)
group3 = np.random.normal(-1, 0.8, 50)

# Combine data
data = pd.DataFrame({
    'Value': np.concatenate([group1, group2, group3]),
    'Group': ['Group 1'] * 50 + ['Group 2'] * 50 + ['Group 3'] * 50
})

plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Group', y='Value')
plt.title('Box Plot')
plt.xlabel('Group')
plt.ylabel('Value')
plt.show()
```

### Violin Plots

```python
plt.figure(figsize=(10, 6))
sns.violinplot(data=data, x='Group', y='Value')
plt.title('Violin Plot')
plt.xlabel('Group')
plt.ylabel('Value')
plt.show()
```

## Distribution Plots

### Histograms

```python
# Create sample data
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(2, 1.5, 1000)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Histogram with KDE
sns.histplot(data=data1, kde=True, ax=axes[0], bins=30)
axes[0].set_title('Histogram with KDE')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Multiple histograms
sns.histplot(data=data1, alpha=0.5, label='Group 1', ax=axes[1])
sns.histplot(data=data2, alpha=0.5, label='Group 2', ax=axes[1])
axes[1].set_title('Multiple Histograms')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Kernel Density Estimation

```python
plt.figure(figsize=(10, 6))

# Create KDE plot
sns.kdeplot(data=data1, label='Group 1', linewidth=2)
sns.kdeplot(data=data2, label='Group 2', linewidth=2)

plt.title('Kernel Density Estimation')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()
```

### Joint Plots

```python
# Create sample data
x = np.random.randn(1000)
y = 0.5 * x + np.random.randn(1000) * 0.5

# Create joint plot
sns.jointplot(x=x, y=y, kind='scatter', height=8)
plt.suptitle('Joint Plot', y=1.02)
plt.show()

# Joint plot with KDE
sns.jointplot(x=x, y=y, kind='kde', height=8)
plt.suptitle('Joint Plot with KDE', y=1.02)
plt.show()
```

## Working with Data

### Pandas DataFrame Integration

```python
# Create a sample DataFrame
np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'size': np.random.rand(100) * 100
})

# Scatter plot with DataFrame
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='x', y='y', hue='category', size='size', 
                sizes=(50, 200), alpha=0.6)
plt.title('Scatter Plot with DataFrame')
plt.show()
```

### Grouped Analysis

```python
# Grouped box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='category', y='y')
plt.title('Grouped Box Plot')
plt.xlabel('Category')
plt.ylabel('Y Value')
plt.show()

# Grouped violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='category', y='y')
plt.title('Grouped Violin Plot')
plt.xlabel('Category')
plt.ylabel('Y Value')
plt.show()
```

### Statistical Annotations

```python
# Create sample data for statistical testing
group_a = np.random.normal(0, 1, 50)
group_b = np.random.normal(1, 1, 50)

# Combine into DataFrame
df_test = pd.DataFrame({
    'value': np.concatenate([group_a, group_b]),
    'group': ['A'] * 50 + ['B'] * 50
})

# Box plot with statistical annotation
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_test, x='group', y='value')
plt.title('Box Plot with Statistical Comparison')
plt.xlabel('Group')
plt.ylabel('Value')

# Add statistical annotation (you would typically use scipy.stats for actual testing)
plt.text(0.5, 3, 'p < 0.001', ha='center', va='center', 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

plt.show()
```

## Best Practices

### 1. Choose Appropriate Plot Types

```python
# Example: Different plots for different data types
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Continuous data - histogram
data_continuous = np.random.normal(0, 1, 1000)
sns.histplot(data=data_continuous, kde=True, ax=axes[0, 0])
axes[0, 0].set_title('Continuous Data - Histogram')

# Categorical data - bar plot
categories = ['A', 'B', 'C', 'D']
values = np.random.rand(4)
sns.barplot(x=categories, y=values, ax=axes[0, 1])
axes[0, 1].set_title('Categorical Data - Bar Plot')

# Relationship data - scatter plot
x = np.random.randn(100)
y = 0.5 * x + np.random.randn(100) * 0.3
sns.scatterplot(x=x, y=y, ax=axes[1, 0])
axes[1, 0].set_title('Relationship Data - Scatter Plot')

# Distribution comparison - box plot
group1 = np.random.normal(0, 1, 50)
group2 = np.random.normal(1, 1, 50)
df_compare = pd.DataFrame({
    'value': np.concatenate([group1, group2]),
    'group': ['Group 1'] * 50 + ['Group 2'] * 50
})
sns.boxplot(data=df_compare, x='group', y='value', ax=axes[1, 1])
axes[1, 1].set_title('Distribution Comparison - Box Plot')

plt.tight_layout()
plt.show()
```

### 2. Use Consistent Styling

```python
# Set consistent style
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)

# Create multiple plots with consistent styling
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1
sns.histplot(data=np.random.normal(0, 1, 1000), kde=True, ax=axes[0])
axes[0].set_title('Distribution 1')

# Plot 2
sns.scatterplot(x=np.random.randn(100), y=np.random.randn(100), ax=axes[1])
axes[1].set_title('Relationship')

# Plot 3
sns.boxplot(data=df_test, x='group', y='value', ax=axes[2])
axes[2].set_title('Comparison')

plt.tight_layout()
plt.show()
```

### 3. Add Meaningful Titles and Labels

```python
plt.figure(figsize=(10, 6))

# Create a meaningful plot
sns.scatterplot(data=df, x='x', y='y', hue='category', alpha=0.7)

# Add meaningful title and labels
plt.title('Relationship Between Variables by Category', fontsize=14, fontweight='bold')
plt.xlabel('Independent Variable (X)', fontsize=12)
plt.ylabel('Dependent Variable (Y)', fontsize=12)

# Add legend with better title
plt.legend(title='Category', title_fontsize=12)

plt.show()
```

### 4. Consider Your Audience

```python
# For technical audience - detailed plot
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='x', y='y', hue='category', size='size', 
                sizes=(50, 200), alpha=0.6)

# Add trend line
z = np.polyfit(df['x'], df['y'], 1)
p = np.poly1d(z)
plt.plot(df['x'], p(df['x']), "r--", alpha=0.8, linewidth=2)

plt.title('Detailed Analysis: Variable Relationships with Size Encoding', fontsize=14)
plt.xlabel('X Variable', fontsize=12)
plt.ylabel('Y Variable', fontsize=12)
plt.legend(title='Category', title_fontsize=12)

# Add statistical annotation
plt.text(0.02, 0.98, f'R² = {np.corrcoef(df["x"], df["y"])[0,1]**2:.3f}', 
         transform=plt.gca().transAxes, fontsize=12, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

plt.show()
```

## Summary

This guide covered the fundamentals of Seaborn:

1. **Basic Plotting**: Line plots, scatter plots, and histograms
2. **Figure Aesthetics**: Styling and customization options
3. **Color Palettes**: Built-in and custom color schemes
4. **Statistical Relationships**: Regression and residual plots
5. **Categorical Data**: Bar plots, box plots, and violin plots
6. **Distribution Plots**: Histograms, KDE, and joint plots
7. **Data Integration**: Working with pandas DataFrames
8. **Best Practices**: Choosing appropriate plots and styling

Seaborn provides a powerful and intuitive interface for creating beautiful statistical visualizations. The key is to choose the right plot type for your data and question, and to use consistent styling throughout your analysis.

## Next Steps

- Explore more advanced plot types in the other guides
- Learn about multi-plot grids and faceting
- Master correlation analysis and heatmaps
- Practice with real-world datasets
- Customize plots for publication quality

Remember: The best visualization is one that clearly communicates your data's story to your audience! 