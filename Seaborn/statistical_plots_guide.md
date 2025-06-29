# Statistical Plots Guide

A comprehensive guide to creating statistical visualizations with Seaborn, covering distribution plots, regression analysis, and statistical annotations.

## Table of Contents

1. [Distribution Plots](#distribution-plots)
2. [Regression Analysis](#regression-analysis)
3. [Statistical Annotations](#statistical-annotations)
4. [Residual Analysis](#residual-analysis)
5. [Statistical Testing Visualization](#statistical-testing-visualization)
6. [Advanced Statistical Plots](#advanced-statistical-plots)

## Distribution Plots

### Histograms with Kernel Density Estimation

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

# Create subplots for different distributions
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

### Multiple Distribution Comparison

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

### Kernel Density Estimation (KDE)

```python
# Create KDE plots for different distributions
plt.figure(figsize=(12, 8))

# Plot KDE for each group
for group in df_dist['group'].unique():
    data = df_dist[df_dist['group'] == group]['value']
    sns.kdeplot(data=data, label=group, linewidth=2)

plt.title('Kernel Density Estimation by Group', fontsize=14, fontweight='bold')
plt.xlabel('Value', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Group', title_fontsize=12)
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

## Regression Analysis

### Linear Regression

```python
# Generate sample data with linear relationship
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

# Create regression plot
plt.figure(figsize=(10, 6))
sns.regplot(x=x, y=y, scatter_kws={'alpha': 0.6, 's': 50}, 
            line_kws={'color': 'red', 'linewidth': 2})

plt.title('Linear Regression Analysis', fontsize=14, fontweight='bold')
plt.xlabel('Independent Variable (X)', fontsize=12)
plt.ylabel('Dependent Variable (Y)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
```

### Polynomial Regression

```python
# Generate sample data with non-linear relationship
np.random.seed(42)
x = np.random.randn(100)
y = x**2 + np.random.randn(100) * 0.5

# Create polynomial regression plot
plt.figure(figsize=(12, 5))

# Linear fit
plt.subplot(1, 2, 1)
sns.regplot(x=x, y=y, order=1, scatter_kws={'alpha': 0.6})
plt.title('Linear Fit (Order 1)')
plt.xlabel('X')
plt.ylabel('Y')

# Quadratic fit
plt.subplot(1, 2, 2)
sns.regplot(x=x, y=y, order=2, scatter_kws={'alpha': 0.6})
plt.title('Quadratic Fit (Order 2)')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.show()
```

### Multiple Regression with Categorical Variables

```python
# Create sample data with categorical variable
np.random.seed(42)
n_samples = 200

# Generate data
x = np.random.randn(n_samples)
category = np.random.choice(['A', 'B', 'C'], n_samples)

# Create different relationships for each category
y = np.zeros(n_samples)
for i, cat in enumerate(['A', 'B', 'C']):
    mask = category == cat
    if cat == 'A':
        y[mask] = 1.5 * x[mask] + np.random.randn(sum(mask)) * 0.3
    elif cat == 'B':
        y[mask] = 0.5 * x[mask] + np.random.randn(sum(mask)) * 0.3
    else:
        y[mask] = -0.5 * x[mask] + np.random.randn(sum(mask)) * 0.3

# Create DataFrame
df_reg = pd.DataFrame({'x': x, 'y': y, 'category': category})

# Create regression plot with categorical variable
plt.figure(figsize=(10, 6))
sns.lmplot(data=df_reg, x='x', y='y', hue='category', 
           scatter_kws={'alpha': 0.6}, line_kws={'linewidth': 2})

plt.suptitle('Multiple Regression by Category', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### Robust Regression

```python
# Generate data with outliers
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

# Add outliers
x = np.append(x, [3, 4, -3, -4])
y = np.append(y, [8, 10, -8, -10])

# Create robust regression plot
plt.figure(figsize=(12, 5))

# Standard regression
plt.subplot(1, 2, 1)
sns.regplot(x=x, y=y, scatter_kws={'alpha': 0.6})
plt.title('Standard Regression')
plt.xlabel('X')
plt.ylabel('Y')

# Robust regression
plt.subplot(1, 2, 2)
sns.regplot(x=x, y=y, robust=True, scatter_kws={'alpha': 0.6})
plt.title('Robust Regression')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.show()
```

## Statistical Annotations

### Correlation Analysis

```python
# Generate correlated data
np.random.seed(42)
n_samples = 100

# Create different correlation scenarios
data_high = np.random.multivariate_normal([0, 0], [[1, 0.8], [0.8, 1]], n_samples)
data_medium = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples)
data_low = np.random.multivariate_normal([0, 0], [[1, 0.2], [0.2, 1]], n_samples)

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# High correlation
sns.regplot(x=data_high[:, 0], y=data_high[:, 1], ax=axes[0], 
            scatter_kws={'alpha': 0.6})
corr_high = np.corrcoef(data_high[:, 0], data_high[:, 1])[0, 1]
axes[0].set_title(f'High Correlation (r = {corr_high:.3f})')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')

# Medium correlation
sns.regplot(x=data_medium[:, 0], y=data_medium[:, 1], ax=axes[1], 
            scatter_kws={'alpha': 0.6})
corr_medium = np.corrcoef(data_medium[:, 0], data_medium[:, 1])[0, 1]
axes[1].set_title(f'Medium Correlation (r = {corr_medium:.3f})')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')

# Low correlation
sns.regplot(x=data_low[:, 0], y=data_low[:, 1], ax=axes[2], 
            scatter_kws={'alpha': 0.6})
corr_low = np.corrcoef(data_low[:, 0], data_low[:, 1])[0, 1]
axes[2].set_title(f'Low Correlation (r = {corr_low:.3f})')
axes[2].set_xlabel('X')
axes[2].set_ylabel('Y')

plt.tight_layout()
plt.show()
```

### Statistical Significance Testing

```python
from scipy.stats import ttest_ind

# Generate sample data for statistical testing
np.random.seed(42)
group_a = np.random.normal(0, 1, 50)
group_b = np.random.normal(1, 1, 50)

# Perform t-test
t_stat, p_value = ttest_ind(group_a, group_b)

# Create visualization with statistical annotation
plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.DataFrame({
    'value': np.concatenate([group_a, group_b]),
    'group': ['Group A'] * 50 + ['Group B'] * 50
}), x='group', y='value')

plt.title('Statistical Comparison Between Groups', fontsize=14, fontweight='bold')
plt.xlabel('Group', fontsize=12)
plt.ylabel('Value', fontsize=12)

# Add statistical annotation
if p_value < 0.001:
    sig_text = 'p < 0.001'
elif p_value < 0.01:
    sig_text = 'p < 0.01'
elif p_value < 0.05:
    sig_text = 'p < 0.05'
else:
    sig_text = f'p = {p_value:.3f}'

plt.text(0.5, plt.ylim()[1] * 0.9, f't-test: {sig_text}', 
         ha='center', va='center', fontsize=12,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

plt.show()
```

## Residual Analysis

### Residual Plots

```python
# Generate sample data
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

# Create residual plot
plt.figure(figsize=(12, 5))

# Main regression plot
plt.subplot(1, 2, 1)
sns.regplot(x=x, y=y, scatter_kws={'alpha': 0.6})
plt.title('Regression Plot')
plt.xlabel('X')
plt.ylabel('Y')

# Residual plot
plt.subplot(1, 2, 2)
sns.residplot(x=x, y=y, scatter_kws={'alpha': 0.6})
plt.title('Residual Plot')
plt.xlabel('X')
plt.ylabel('Residuals')
plt.axhline(y=0, color='red', linestyle='--', alpha=0.8)

plt.tight_layout()
plt.show()
```

### Residual Distribution

```python
# Calculate residuals
from sklearn.linear_model import LinearRegression

# Fit linear regression
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)
y_pred = model.predict(x.reshape(-1, 1))
residuals = y - y_pred

# Create residual distribution plot
plt.figure(figsize=(12, 5))

# Residual histogram
plt.subplot(1, 2, 1)
sns.histplot(data=residuals, kde=True, bins=20)
plt.title('Residual Distribution')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

# Q-Q plot for normality
plt.subplot(1, 2, 2)
from scipy.stats import probplot
probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot for Normality')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')

plt.tight_layout()
plt.show()
```

## Statistical Testing Visualization

### T-Test Visualization

```python
# Generate data for t-test
np.random.seed(42)
group1 = np.random.normal(0, 1, 30)
group2 = np.random.normal(1.5, 1, 30)

# Perform t-test
from scipy.stats import ttest_ind
t_stat, p_value = ttest_ind(group1, group2)

# Create visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Box plot
sns.boxplot(data=pd.DataFrame({
    'value': np.concatenate([group1, group2]),
    'group': ['Group 1'] * 30 + ['Group 2'] * 30
}), x='group', y='value', ax=axes[0])
axes[0].set_title('Box Plot Comparison')
axes[0].set_xlabel('Group')
axes[0].set_ylabel('Value')

# Violin plot
sns.violinplot(data=pd.DataFrame({
    'value': np.concatenate([group1, group2]),
    'group': ['Group 1'] * 30 + ['Group 2'] * 30
}), x='group', y='value', ax=axes[1])
axes[1].set_title('Violin Plot Comparison')
axes[1].set_xlabel('Group')
axes[1].set_ylabel('Value')

# Add statistical annotation
fig.suptitle(f'T-Test: t = {t_stat:.3f}, p = {p_value:.3f}', 
             fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
```

### ANOVA Visualization

```python
# Generate data for ANOVA
np.random.seed(42)
group1 = np.random.normal(0, 1, 25)
group2 = np.random.normal(1, 1, 25)
group3 = np.random.normal(2, 1, 25)

# Perform ANOVA
from scipy.stats import f_oneway
f_stat, p_value = f_oneway(group1, group2, group3)

# Create visualization
plt.figure(figsize=(12, 6))

# Box plot for ANOVA
sns.boxplot(data=pd.DataFrame({
    'value': np.concatenate([group1, group2, group3]),
    'group': ['Group 1'] * 25 + ['Group 2'] * 25 + ['Group 3'] * 25
}), x='group', y='value')

plt.title(f'ANOVA: F = {f_stat:.3f}, p = {p_value:.3f}', 
          fontsize=14, fontweight='bold')
plt.xlabel('Group', fontsize=12)
plt.ylabel('Value', fontsize=12)

plt.show()
```

## Advanced Statistical Plots

### Joint Distribution Analysis

```python
# Generate correlated data
np.random.seed(42)
x = np.random.randn(1000)
y = 0.7 * x + np.random.randn(1000) * 0.5

# Create joint plot
sns.jointplot(x=x, y=y, kind='hex', height=8)
plt.suptitle('Joint Distribution Analysis', y=1.02, fontsize=14, fontweight='bold')
plt.show()

# Joint plot with KDE
sns.jointplot(x=x, y=y, kind='kde', height=8)
plt.suptitle('Joint Distribution with KDE', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### Pairwise Relationships

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
plt.suptitle('Pairwise Relationships', y=1.02, fontsize=14, fontweight='bold')
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
from scipy.stats import probplot
probplot(normal_data, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot for Normality')

plt.suptitle('Comprehensive Statistical Summary', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Summary

This guide covered advanced statistical plotting with Seaborn:

1. **Distribution Plots**: Histograms, KDE, and distribution comparisons
2. **Regression Analysis**: Linear, polynomial, and multiple regression
3. **Statistical Annotations**: Correlation analysis and significance testing
4. **Residual Analysis**: Residual plots and distribution analysis
5. **Statistical Testing**: T-tests, ANOVA, and other statistical test visualizations
6. **Advanced Plots**: Joint distributions and pairwise relationships

These statistical plots are essential for:
- **Exploratory Data Analysis**: Understanding data distributions and relationships
- **Model Diagnostics**: Checking assumptions and model fit
- **Statistical Inference**: Visualizing test results and significance
- **Publication Quality**: Creating professional statistical graphics

## Best Practices

1. **Choose Appropriate Tests**: Select statistical tests based on your data and research question
2. **Check Assumptions**: Always verify the assumptions of your statistical tests
3. **Interpret Results**: Don't just report p-values; interpret the practical significance
4. **Use Multiple Views**: Combine different plot types for comprehensive analysis
5. **Document Your Analysis**: Include clear titles, labels, and statistical annotations

## Next Steps

- Explore categorical plots for discrete data analysis
- Learn about correlation analysis and heatmaps
- Master multi-plot grids for complex visualizations
- Practice with real-world datasets and statistical problems
- Customize plots for specific publication requirements

Remember: Statistical visualization is not just about making pretty plots—it's about communicating the story in your data clearly and accurately! 