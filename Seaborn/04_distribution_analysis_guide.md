# Distribution Analysis Guide

A comprehensive guide to analyzing data distributions with Seaborn, covering univariate and bivariate distributions, joint plots, and distribution comparisons.

## Table of Contents

1. [Univariate Distributions](#univariate-distributions)
2. [Bivariate Distributions](#bivariate-distributions)
3. [Joint Plots](#joint-plots)
4. [Pair Plots](#pair-plots)
5. [Distribution Comparisons](#distribution-comparisons)
6. [Advanced Distribution Analysis](#advanced-distribution-analysis)

## Univariate Distributions

### Histograms

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Set up the plotting style
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)

# Generate sample data
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)
exponential_data = np.random.exponential(1, 1000)
uniform_data = np.random.uniform(-2, 2, 1000)

# Create histograms for different distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Normal distribution
sns.histplot(data=normal_data, kde=True, ax=axes[0], bins=30, color='skyblue')
axes[0].set_title('Normal Distribution')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Exponential distribution
sns.histplot(data=exponential_data, kde=True, ax=axes[1], bins=30, color='lightcoral')
axes[1].set_title('Exponential Distribution')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')

# Uniform distribution
sns.histplot(data=uniform_data, kde=True, ax=axes[2], bins=30, color='lightgreen')
axes[2].set_title('Uniform Distribution')
axes[2].set_xlabel('Value')
axes[2].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

### Kernel Density Estimation (KDE)

```python
# Create KDE plots for different distributions
plt.figure(figsize=(12, 8))

# Plot KDE for each distribution
sns.kdeplot(data=normal_data, label='Normal', linewidth=2)
sns.kdeplot(data=exponential_data, label='Exponential', linewidth=2)
sns.kdeplot(data=uniform_data, label='Uniform', linewidth=2)

plt.title('Kernel Density Estimation Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Value', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Distribution', title_fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
```

### Rug Plots

```python
# Create rug plot to show individual data points
plt.figure(figsize=(12, 8))

# Create histogram with rug plot
sns.histplot(data=normal_data, kde=True, bins=30, alpha=0.7)
sns.rugplot(data=normal_data, alpha=0.5, color='red')

plt.title('Histogram with Rug Plot', fontsize=14, fontweight='bold')
plt.xlabel('Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()
```

### Distribution with Multiple Groups

```python
# Create data with multiple groups
np.random.seed(42)
group1 = np.random.normal(0, 1, 500)
group2 = np.random.normal(2, 1.5, 500)
group3 = np.random.normal(-1, 0.8, 500)

# Combine into DataFrame
df_dist = pd.DataFrame({
    'value': np.concatenate([group1, group2, group3]),
    'group': ['Group 1'] * 500 + ['Group 2'] * 500 + ['Group 3'] * 500
})

# Create overlapping histograms
plt.figure(figsize=(12, 8))

# Plot histograms for each group
for group in df_dist['group'].unique():
    data = df_dist[df_dist['group'] == group]['value']
    sns.histplot(data=data, kde=True, alpha=0.6, label=group, bins=30)

plt.title('Distribution Comparison Across Groups', fontsize=14, fontweight='bold')
plt.xlabel('Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(title='Group', title_fontsize=12)
plt.show()
```

## Bivariate Distributions

### Scatter Plots with Density

```python
# Generate correlated data
np.random.seed(42)
x = np.random.randn(1000)
y = 0.7 * x + np.random.randn(1000) * 0.5

# Create scatter plot with density
plt.figure(figsize=(12, 5))

# Regular scatter plot
plt.subplot(1, 2, 1)
sns.scatterplot(x=x, y=y, alpha=0.6)
plt.title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')

# Scatter plot with density contours
plt.subplot(1, 2, 2)
sns.kdeplot(data=pd.DataFrame({'x': x, 'y': y}), x='x', y='y', fill=True, alpha=0.6)
plt.title('Density Contour Plot')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.show()
```

### Hexbin Plots

```python
# Create hexbin plot for large datasets
plt.figure(figsize=(10, 6))
sns.jointplot(x=x, y=y, kind='hex', height=8)
plt.suptitle('Hexbin Plot for Large Dataset', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

## Joint Plots

### Basic Joint Plot

```python
# Create basic joint plot
sns.jointplot(x=x, y=y, kind='scatter', height=8)
plt.suptitle('Joint Plot - Scatter', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### Joint Plot with KDE

```python
# Create joint plot with KDE
sns.jointplot(x=x, y=y, kind='kde', height=8)
plt.suptitle('Joint Plot - KDE', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### Joint Plot with Histogram

```python
# Create joint plot with histogram
sns.jointplot(x=x, y=y, kind='hist', height=8)
plt.suptitle('Joint Plot - Histogram', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### Joint Plot with Regression

```python
# Create joint plot with regression
sns.jointplot(x=x, y=y, kind='reg', height=8)
plt.suptitle('Joint Plot - Regression', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

## Pair Plots

### Basic Pair Plot

```python
# Generate multivariate data
np.random.seed(42)
n_samples = 200

# Create correlated variables
data = np.random.multivariate_normal(
    [0, 0, 0], 
    [[1, 0.7, 0.3], [0.7, 1, 0.5], [0.3, 0.5, 1]], 
    n_samples
)

# Create DataFrame
df_pair = pd.DataFrame(data, columns=['X1', 'X2', 'X3'])

# Create pair plot
sns.pairplot(df_pair, diag_kind='kde')
plt.suptitle('Pair Plot', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### Pair Plot with Categorical Variable

```python
# Add categorical variable
df_pair['Category'] = np.random.choice(['A', 'B', 'C'], n_samples)

# Create pair plot with categorical coloring
sns.pairplot(df_pair, hue='Category', diag_kind='kde')
plt.suptitle('Pair Plot with Categorical Variable', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### Pair Plot with Different Plot Types

```python
# Create pair plot with different plot types
sns.pairplot(df_pair, diag_kind='hist', plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plot with Histograms', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

## Distribution Comparisons

### Multiple Distribution Comparison

```python
# Create comprehensive distribution comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Histogram comparison
for group in df_dist['group'].unique():
    data = df_dist[df_dist['group'] == group]['value']
    sns.histplot(data=data, kde=True, alpha=0.6, label=group, ax=axes[0, 0], bins=30)
axes[0, 0].set_title('Histogram Comparison')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()

# KDE comparison
for group in df_dist['group'].unique():
    data = df_dist[df_dist['group'] == group]['value']
    sns.kdeplot(data=data, label=group, ax=axes[0, 1], linewidth=2)
axes[0, 1].set_title('KDE Comparison')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()

# Box plot comparison
sns.boxplot(data=df_dist, x='group', y='value', ax=axes[1, 0])
axes[1, 0].set_title('Box Plot Comparison')
axes[1, 0].set_xlabel('Group')
axes[1, 0].set_ylabel('Value')

# Violin plot comparison
sns.violinplot(data=df_dist, x='group', y='value', ax=axes[1, 1])
axes[1, 1].set_title('Violin Plot Comparison')
axes[1, 1].set_xlabel('Group')
axes[1, 1].set_ylabel('Value')

plt.suptitle('Comprehensive Distribution Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Before and After Comparison

```python
# Generate before and after data
np.random.seed(42)
before = np.random.normal(0, 1, 100)
after = before + np.random.normal(1, 0.5, 100)  # Treatment effect

# Create before/after comparison
df_before_after = pd.DataFrame({
    'value': np.concatenate([before, after]),
    'time': ['Before'] * 100 + ['After'] * 100
})

# Create comparison plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Histogram comparison
sns.histplot(data=df_before_after, x='value', hue='time', alpha=0.6, ax=axes[0])
axes[0].set_title('Histogram Comparison')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Box plot comparison
sns.boxplot(data=df_before_after, x='time', y='value', ax=axes[1])
axes[1].set_title('Box Plot Comparison')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Value')

# Violin plot comparison
sns.violinplot(data=df_before_after, x='time', y='value', ax=axes[2])
axes[2].set_title('Violin Plot Comparison')
axes[2].set_xlabel('Time')
axes[2].set_ylabel('Value')

plt.suptitle('Before vs After Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Advanced Distribution Analysis

### Q-Q Plots for Normality

```python
# Create Q-Q plots for normality testing
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Normal data
from scipy.stats import probplot
probplot(normal_data, dist="norm", plot=axes[0])
axes[0].set_title('Normal Data - Q-Q Plot')

# Exponential data
probplot(exponential_data, dist="norm", plot=axes[1])
axes[1].set_title('Exponential Data - Q-Q Plot')

# Uniform data
probplot(uniform_data, dist="norm", plot=axes[2])
axes[2].set_title('Uniform Data - Q-Q Plot')

plt.suptitle('Q-Q Plots for Normality Testing', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Distribution Fitting

```python
# Fit different distributions to data
from scipy.stats import norm, expon, uniform

# Create subplots for different fits
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Normal fit
sns.histplot(data=normal_data, kde=True, ax=axes[0], bins=30, alpha=0.7)
x_norm = np.linspace(normal_data.min(), normal_data.max(), 100)
y_norm = norm.pdf(x_norm, normal_data.mean(), normal_data.std())
axes[0].plot(x_norm, y_norm * len(normal_data) * (normal_data.max() - normal_data.min()) / 30, 
             'r-', linewidth=2, label='Normal Fit')
axes[0].set_title('Normal Distribution Fit')
axes[0].legend()

# Exponential fit
sns.histplot(data=exponential_data, kde=True, ax=axes[1], bins=30, alpha=0.7)
x_exp = np.linspace(exponential_data.min(), exponential_data.max(), 100)
y_exp = expon.pdf(x_exp, scale=exponential_data.mean())
axes[1].plot(x_exp, y_exp * len(exponential_data) * (exponential_data.max() - exponential_data.min()) / 30, 
             'r-', linewidth=2, label='Exponential Fit')
axes[1].set_title('Exponential Distribution Fit')
axes[1].legend()

# Uniform fit
sns.histplot(data=uniform_data, kde=True, ax=axes[2], bins=30, alpha=0.7)
x_uni = np.linspace(uniform_data.min(), uniform_data.max(), 100)
y_uni = uniform.pdf(x_uni, uniform_data.min(), uniform_data.max() - uniform_data.min())
axes[2].plot(x_uni, y_uni * len(uniform_data) * (uniform_data.max() - uniform_data.min()) / 30, 
             'r-', linewidth=2, label='Uniform Fit')
axes[2].set_title('Uniform Distribution Fit')
axes[2].legend()

plt.suptitle('Distribution Fitting', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Statistical Summary Plots

```python
# Create comprehensive statistical summary
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Histogram with KDE
sns.histplot(data=normal_data, kde=True, ax=axes[0, 0], bins=30)
axes[0, 0].set_title('Distribution')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')

# Box plot
sns.boxplot(data=normal_data, ax=axes[0, 1])
axes[0, 1].set_title('Box Plot')
axes[0, 1].set_ylabel('Value')

# Violin plot
sns.violinplot(data=normal_data, ax=axes[1, 0])
axes[1, 0].set_title('Violin Plot')
axes[1, 0].set_ylabel('Value')

# Q-Q plot
probplot(normal_data, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot for Normality')

plt.suptitle('Comprehensive Statistical Summary', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Distribution with Outliers

```python
# Create data with outliers
np.random.seed(42)
clean_data = np.random.normal(0, 1, 1000)
outlier_data = np.concatenate([clean_data, [5, -5, 8, -8, 10, -10]])

# Create comparison plots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Histogram comparison
sns.histplot(data=clean_data, kde=True, ax=axes[0], bins=30, alpha=0.7, label='Clean Data')
sns.histplot(data=outlier_data, kde=True, ax=axes[0], bins=30, alpha=0.7, label='With Outliers')
axes[0].set_title('Histogram Comparison')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')
axes[0].legend()

# Box plot comparison
sns.boxplot(data=clean_data, ax=axes[1], color='lightblue')
axes[1].set_title('Clean Data - Box Plot')
axes[1].set_ylabel('Value')

sns.boxplot(data=outlier_data, ax=axes[2], color='lightcoral')
axes[2].set_title('Data with Outliers - Box Plot')
axes[2].set_ylabel('Value')

plt.suptitle('Effect of Outliers on Distribution', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Summary

This guide covered comprehensive distribution analysis with Seaborn:

1. **Univariate Distributions**: Histograms, KDE, and rug plots for single variables
2. **Bivariate Distributions**: Scatter plots and density contours for two variables
3. **Joint Plots**: Combined univariate and bivariate analysis
4. **Pair Plots**: Multi-variable relationship analysis
5. **Distribution Comparisons**: Before/after and group comparisons
6. **Advanced Analysis**: Q-Q plots, distribution fitting, and outlier analysis

These distribution analysis techniques are essential for:
- **Data Exploration**: Understanding the shape and characteristics of your data
- **Model Assumptions**: Checking if your data meets statistical model requirements
- **Outlier Detection**: Identifying unusual data points
- **Statistical Inference**: Making informed decisions about data transformations

## Best Practices

1. **Start with Univariate Analysis**: Always examine individual variables first
2. **Use Multiple Plot Types**: Combine different visualization approaches
3. **Check for Outliers**: Always look for unusual data points
4. **Test Distribution Assumptions**: Use Q-Q plots and statistical tests
5. **Consider Data Transformations**: Log, square root, or other transformations if needed

## Next Steps

- Explore correlation analysis and heatmaps
- Learn about multi-plot grids for complex visualizations
- Master statistical testing and significance analysis
- Practice with real-world datasets
- Customize plots for specific publication requirements

Remember: Understanding your data's distribution is the foundation of any good statistical analysis! 