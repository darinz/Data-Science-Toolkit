# Matplotlib Statistical Plots: Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Box Plots](#box-plots)
3. [Violin Plots](#violin-plots)
4. [Histograms](#histograms)
5. [Density Plots](#density-plots)
6. [Correlation Matrices](#correlation-matrices)
7. [Statistical Analysis Plots](#statistical-analysis-plots)
8. [Distribution Comparison](#distribution-comparison)
9. [Advanced Statistical Visualizations](#advanced-statistical-visualizations)
10. [Best Practices](#best-practices)

## Introduction

Matplotlib provides powerful tools for creating statistical visualizations. This guide covers box plots, violin plots, histograms, density plots, correlation matrices, and other statistical analysis plots with practical examples.

## Box Plots

### Basic Box Plot
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate sample data
np.random.seed(42)
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(2, 1.5, 100)
data3 = np.random.normal(-1, 0.8, 100)

# Create box plot
fig, ax = plt.subplots(figsize=(10, 6))
box_data = [data1, data2, data3]
labels = ['Group A', 'Group B', 'Group C']

bp = ax.boxplot(box_data, labels=labels, patch_artist=True)

# Customize colors
colors = ['lightblue', 'lightcoral', 'lightgreen']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title('Basic Box Plot', fontsize=14, fontweight='bold')
ax.set_ylabel('Values')
ax.grid(True, alpha=0.3)
plt.show()
```

### Box Plot with Outliers
```python
# Generate data with outliers
np.random.seed(42)
normal_data = np.random.normal(0, 1, 100)
outlier_data = np.concatenate([normal_data, [5, -5, 8, -8]])  # Add outliers

fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot([normal_data, outlier_data], 
                labels=['No Outliers', 'With Outliers'],
                patch_artist=True)

# Customize appearance
colors = ['lightblue', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Customize outlier appearance
bp['fliers'][0].set(marker='o', markerfacecolor='red', markersize=8)
bp['fliers'][1].set(marker='o', markerfacecolor='red', markersize=8)

ax.set_title('Box Plot with Outliers', fontsize=14, fontweight='bold')
ax.set_ylabel('Values')
ax.grid(True, alpha=0.3)
plt.show()
```

### Grouped Box Plot
```python
# Create grouped data
np.random.seed(42)
categories = ['A', 'B', 'C', 'D']
groups = ['Control', 'Treatment']

# Generate data for each combination
data = {}
for cat in categories:
    for group in groups:
        key = f"{cat}_{group}"
        if group == 'Control':
            data[key] = np.random.normal(0, 1, 50)
        else:
            data[key] = np.random.normal(1, 1.2, 50)

# Prepare data for plotting
control_data = [data[f"{cat}_Control"] for cat in categories]
treatment_data = [data[f"{cat}_Treatment"] for cat in categories]

# Create grouped box plot
fig, ax = plt.subplots(figsize=(12, 6))

# Position boxes
pos1 = np.arange(len(categories)) - 0.2
pos2 = np.arange(len(categories)) + 0.2

bp1 = ax.boxplot(control_data, positions=pos1, patch_artist=True, 
                 labels=categories, widths=0.35)
bp2 = ax.boxplot(treatment_data, positions=pos2, patch_artist=True, 
                 labels=categories, widths=0.35)

# Customize colors
for patch in bp1['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)

for patch in bp2['boxes']:
    patch.set_facecolor('lightcoral')
    patch.set_alpha(0.7)

ax.set_title('Grouped Box Plot', fontsize=14, fontweight='bold')
ax.set_ylabel('Values')
ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Control', 'Treatment'])
ax.grid(True, alpha=0.3)
plt.show()
```

## Violin Plots

### Basic Violin Plot
```python
# Generate sample data
np.random.seed(42)
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(2, 1.5, 100)
data3 = np.random.normal(-1, 0.8, 100)

fig, ax = plt.subplots(figsize=(10, 6))
violin_data = [data1, data2, data3]
labels = ['Group A', 'Group B', 'Group C']

parts = ax.violinplot(violin_data, positions=range(len(labels)))

# Customize violin plots
for pc in parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.7)

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_title('Basic Violin Plot', fontsize=14, fontweight='bold')
ax.set_ylabel('Values')
ax.grid(True, alpha=0.3)
plt.show()
```

### Violin Plot with Box Plot Overlay
```python
fig, ax = plt.subplots(figsize=(10, 6))

# Create violin plot
parts = ax.violinplot(violin_data, positions=range(len(labels)))

# Customize violin plots
for pc in parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.5)

# Add box plot overlay
bp = ax.boxplot(violin_data, positions=range(len(labels)), 
                widths=0.3, patch_artist=True)

# Customize box plots
for patch in bp['boxes']:
    patch.set_facecolor('white')
    patch.set_alpha(0.8)

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_title('Violin Plot with Box Plot Overlay', fontsize=14, fontweight='bold')
ax.set_ylabel('Values')
ax.grid(True, alpha=0.3)
plt.show()
```

## Histograms

### Basic Histogram
```python
# Generate sample data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')

ax.set_title('Basic Histogram', fontsize=14, fontweight='bold')
ax.set_xlabel('Values')
ax.set_ylabel('Frequency')
ax.grid(True, alpha=0.3)
plt.show()
```

### Histogram with Multiple Datasets
```python
# Generate multiple datasets
np.random.seed(42)
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(2, 1.5, 1000)
data3 = np.random.normal(-1, 0.8, 1000)

fig, ax = plt.subplots(figsize=(10, 6))

ax.hist(data1, bins=30, alpha=0.7, label='Dataset 1', color='skyblue')
ax.hist(data2, bins=30, alpha=0.7, label='Dataset 2', color='lightcoral')
ax.hist(data3, bins=30, alpha=0.7, label='Dataset 3', color='lightgreen')

ax.set_title('Histogram with Multiple Datasets', fontsize=14, fontweight='bold')
ax.set_xlabel('Values')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### Stacked Histogram
```python
fig, ax = plt.subplots(figsize=(10, 6))

# Create stacked histogram
ax.hist([data1, data2, data3], bins=30, alpha=0.7, 
        label=['Dataset 1', 'Dataset 2', 'Dataset 3'],
        color=['skyblue', 'lightcoral', 'lightgreen'])

ax.set_title('Stacked Histogram', fontsize=14, fontweight='bold')
ax.set_xlabel('Values')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### Cumulative Histogram
```python
fig, ax = plt.subplots(figsize=(10, 6))

# Create cumulative histogram
ax.hist(data1, bins=30, cumulative=True, alpha=0.7, 
        color='skyblue', edgecolor='black')

ax.set_title('Cumulative Histogram', fontsize=14, fontweight='bold')
ax.set_xlabel('Values')
ax.set_ylabel('Cumulative Frequency')
ax.grid(True, alpha=0.3)
plt.show()
```

## Density Plots

### Kernel Density Estimation (KDE)
```python
from scipy import stats

# Generate sample data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

fig, ax = plt.subplots(figsize=(10, 6))

# Create histogram
ax.hist(data, bins=30, alpha=0.3, color='skyblue', density=True, 
        label='Histogram')

# Create KDE
kde = stats.gaussian_kde(data)
x_range = np.linspace(data.min(), data.max(), 200)
ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

ax.set_title('Kernel Density Estimation', fontsize=14, fontweight='bold')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

### Multiple KDE Plots
```python
fig, ax = plt.subplots(figsize=(10, 6))

# Generate multiple datasets
datasets = [
    np.random.normal(0, 1, 1000),
    np.random.normal(2, 1.5, 1000),
    np.random.normal(-1, 0.8, 1000)
]

colors = ['blue', 'red', 'green']
labels = ['Dataset 1', 'Dataset 2', 'Dataset 3']

for data, color, label in zip(datasets, colors, labels):
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 200)
    ax.plot(x_range, kde(x_range), color=color, linewidth=2, label=label)

ax.set_title('Multiple KDE Plots', fontsize=14, fontweight='bold')
ax.set_xlabel('Values')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

## Correlation Matrices

### Basic Correlation Matrix
```python
# Generate correlated data
np.random.seed(42)
n_samples = 1000

# Create correlated variables
x1 = np.random.normal(0, 1, n_samples)
x2 = 0.7 * x1 + np.random.normal(0, 0.5, n_samples)
x3 = -0.5 * x1 + np.random.normal(0, 0.8, n_samples)
x4 = np.random.normal(0, 1, n_samples)  # Independent

# Create DataFrame
df = pd.DataFrame({
    'Variable 1': x1,
    'Variable 2': x2,
    'Variable 3': x3,
    'Variable 4': x4
})

# Calculate correlation matrix
corr_matrix = df.corr()

# Create heatmap
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Correlation Coefficient')

# Add text annotations
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                      ha='center', va='center', fontsize=12)

ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45)
ax.set_yticklabels(corr_matrix.columns)

ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Correlation Matrix with Seaborn Style
```python
import seaborn as sns

# Create correlation matrix heatmap with seaborn
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, ax=ax, fmt='.2f')

ax.set_title('Correlation Matrix (Seaborn Style)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Statistical Analysis Plots

### Q-Q Plot (Quantile-Quantile)
```python
from scipy import stats

# Generate sample data
np.random.seed(42)
data = np.random.normal(0, 1, 1000)

fig, ax = plt.subplots(figsize=(8, 8))

# Create Q-Q plot
stats.probplot(data, dist="norm", plot=ax)

ax.set_title('Q-Q Plot (Normal Distribution)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.show()
```

### Residual Plot
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Generate sample data
np.random.seed(42)
x = np.linspace(0, 10, 100)
y_true = 2 * x + 1
y = y_true + np.random.normal(0, 1, 100)

# Fit linear regression
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)
y_pred = model.predict(x.reshape(-1, 1))

# Calculate residuals
residuals = y - y_pred

# Create residual plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Original data and regression line
ax1.scatter(x, y, alpha=0.6, label='Data')
ax1.plot(x, y_pred, 'r-', linewidth=2, label=f'Regression (R²={r2_score(y, y_pred):.3f})')
ax1.set_title('Linear Regression')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Residual plot
ax2.scatter(y_pred, residuals, alpha=0.6)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax2.set_title('Residual Plot')
ax2.set_xlabel('Predicted Values')
ax2.set_ylabel('Residuals')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Bland-Altman Plot
```python
# Generate sample data for method comparison
np.random.seed(42)
method1 = np.random.normal(100, 10, 50)
method2 = method1 + np.random.normal(0, 2, 50)  # Method 2 with some bias

# Calculate means and differences
means = (method1 + method2) / 2
differences = method1 - method2

# Calculate limits of agreement
mean_diff = np.mean(differences)
std_diff = np.std(differences)
loa_upper = mean_diff + 1.96 * std_diff
loa_lower = mean_diff - 1.96 * std_diff

# Create Bland-Altman plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(means, differences, alpha=0.6)
ax.axhline(y=mean_diff, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_diff:.2f}')
ax.axhline(y=loa_upper, color='red', linestyle='--', linewidth=2, 
           label=f'Upper LoA: {loa_upper:.2f}')
ax.axhline(y=loa_lower, color='red', linestyle='--', linewidth=2, 
           label=f'Lower LoA: {loa_lower:.2f}')

ax.set_title('Bland-Altman Plot', fontsize=14, fontweight='bold')
ax.set_xlabel('Mean of Two Methods')
ax.set_ylabel('Difference (Method 1 - Method 2)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

## Distribution Comparison

### Multiple Distribution Comparison
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Generate different distributions
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)
uniform_data = np.random.uniform(-3, 3, 1000)
exponential_data = np.random.exponential(1, 1000)
gamma_data = np.random.gamma(2, 1, 1000)

distributions = [
    (normal_data, 'Normal Distribution'),
    (uniform_data, 'Uniform Distribution'),
    (exponential_data, 'Exponential Distribution'),
    (gamma_data, 'Gamma Distribution')
]

for i, (data, title) in enumerate(distributions):
    row, col = i // 2, i % 2
    ax = axes[row, col]
    
    # Create histogram
    ax.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
    
    # Add KDE
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 200)
    ax.plot(x_range, kde(x_range), 'r-', linewidth=2)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Values')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Distribution Comparison with Box Plots
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Box plot comparison
bp = ax1.boxplot([normal_data, uniform_data, exponential_data, gamma_data],
                 labels=['Normal', 'Uniform', 'Exponential', 'Gamma'],
                 patch_artist=True)

colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_title('Distribution Comparison - Box Plots', fontsize=14, fontweight='bold')
ax1.set_ylabel('Values')
ax1.grid(True, alpha=0.3)

# Violin plot comparison
parts = ax2.violinplot([normal_data, uniform_data, exponential_data, gamma_data],
                       positions=range(4))

for pc in parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.7)

ax2.set_xticks(range(4))
ax2.set_xticklabels(['Normal', 'Uniform', 'Exponential', 'Gamma'])
ax2.set_title('Distribution Comparison - Violin Plots', fontsize=14, fontweight='bold')
ax2.set_ylabel('Values')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Advanced Statistical Visualizations

### Statistical Summary Dashboard
```python
def create_statistical_dashboard(data_dict, figsize=(15, 10)):
    """Create a comprehensive statistical dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    for i, (name, data) in enumerate(data_dict.items()):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        # Create multiple visualizations for each dataset
        # Histogram
        ax.hist(data, bins=30, alpha=0.7, color='skyblue', density=True)
        
        # KDE
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2)
        
        # Add statistics text
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.text(0.02, 0.98, f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'{name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Values')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Statistical Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

# Example usage
np.random.seed(42)
data_dict = {
    'Normal': np.random.normal(0, 1, 1000),
    'Skewed': np.random.gamma(2, 1, 1000),
    'Bimodal': np.concatenate([np.random.normal(-2, 0.5, 500),
                              np.random.normal(2, 0.5, 500)]),
    'Heavy Tailed': np.random.standard_t(3, 1000),
    'Uniform': np.random.uniform(-3, 3, 1000),
    'Exponential': np.random.exponential(1, 1000)
}

fig = create_statistical_dashboard(data_dict)
plt.show()
```

### Statistical Test Visualization
```python
from scipy.stats import ttest_ind, mannwhitneyu

# Generate sample data for statistical tests
np.random.seed(42)
group1 = np.random.normal(0, 1, 100)
group2 = np.random.normal(0.5, 1, 100)

# Perform statistical tests
t_stat, t_p = ttest_ind(group1, group2)
u_stat, u_p = mannwhitneyu(group1, group2)

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Box plot comparison
bp = ax1.boxplot([group1, group2], labels=['Group 1', 'Group 2'], patch_artist=True)
colors = ['lightblue', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_title('Group Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Values')
ax1.grid(True, alpha=0.3)

# Statistical test results
test_names = ['t-test', 'Mann-Whitney U']
p_values = [t_p, u_p]

bars = ax2.bar(test_names, p_values, color=['skyblue', 'lightcoral'], alpha=0.7)
ax2.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')

# Add p-value labels
for bar, p_val in zip(bars, p_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'p = {p_val:.3f}', ha='center', va='bottom')

ax2.set_title('Statistical Test Results', fontsize=14, fontweight='bold')
ax2.set_ylabel('p-value')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Best Practices

1. **Choose appropriate plots** for your data type and analysis goals
2. **Use consistent styling** across related plots
3. **Include proper labels** and titles for clarity
4. **Consider your audience** when choosing complexity
5. **Test with different data** to ensure robustness
6. **Use color effectively** but ensure accessibility
7. **Include statistical context** when appropriate
8. **Validate assumptions** before applying statistical tests

## Resources

- [Matplotlib Statistical Plots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html)
- [SciPy Statistical Functions](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Statistical Visualization Best Practices](https://www.data-to-viz.com/)

---

**Master statistical visualization!** 📊 