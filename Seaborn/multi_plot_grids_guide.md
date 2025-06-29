# Multi-Plot Grids Guide

A comprehensive guide to creating complex multi-panel visualizations with Seaborn using FacetGrid, PairGrid, and JointGrid.

## Table of Contents

1. [FacetGrid](#facetgrid)
2. [PairGrid](#pairgrid)
3. [JointGrid](#jointgrid)
4. [Complex Multi-Panel Layouts](#complex-multi-panel-layouts)
5. [Custom Grid Functions](#custom-grid-functions)
6. [Advanced Grid Techniques](#advanced-grid-techniques)

## FacetGrid

### Basic FacetGrid

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up the plotting style
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)

# Generate sample data
np.random.seed(42)
n_samples = 200

# Create sample data with categorical variables
data = []
for category in ['A', 'B', 'C']:
    for group in ['Group 1', 'Group 2']:
        for _ in range(n_samples // 6):
            data.append({
                'x': np.random.randn(),
                'y': np.random.randn(),
                'category': category,
                'group': group
            })

df = pd.DataFrame(data)

# Create basic FacetGrid
g = sns.FacetGrid(df, col='category', row='group', height=4, aspect=1.2)
g.map_dataframe(sns.scatterplot, x='x', y='y', alpha=0.6)
g.set_titles(col_template='{col_name}', row_template='{row_name}')
g.fig.suptitle('Basic FacetGrid', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### FacetGrid with Different Plot Types

```python
# Create FacetGrid with different plot types
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Scatter plot
g1 = sns.FacetGrid(df, col='category', height=4, aspect=1.2)
g1.map_dataframe(sns.scatterplot, x='x', y='y', alpha=0.6)
g1.set_titles(col_template='{col_name}')
g1.fig.suptitle('Scatter Plot FacetGrid', y=1.02, fontsize=12, fontweight='bold')

# Box plot
g2 = sns.FacetGrid(df, col='category', height=4, aspect=1.2)
g2.map_dataframe(sns.boxplot, x='group', y='y')
g2.set_titles(col_template='{col_name}')
g2.fig.suptitle('Box Plot FacetGrid', y=1.02, fontsize=12, fontweight='bold')

# Histogram
g3 = sns.FacetGrid(df, col='category', height=4, aspect=1.2)
g3.map_dataframe(sns.histplot, x='x', bins=20)
g3.set_titles(col_template='{col_name}')
g3.fig.suptitle('Histogram FacetGrid', y=1.02, fontsize=12, fontweight='bold')

# Violin plot
g4 = sns.FacetGrid(df, col='category', height=4, aspect=1.2)
g4.map_dataframe(sns.violinplot, x='group', y='y')
g4.set_titles(col_template='{col_name}')
g4.fig.suptitle('Violin Plot FacetGrid', y=1.02, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
```

### FacetGrid with Regression

```python
# Create FacetGrid with regression plots
g = sns.FacetGrid(df, col='category', row='group', height=4, aspect=1.2)
g.map_dataframe(sns.regplot, x='x', y='y', scatter_kws={'alpha': 0.6})
g.set_titles(col_template='{col_name}', row_template='{row_name}')
g.fig.suptitle('Regression FacetGrid', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### FacetGrid with Custom Functions

```python
# Define custom plotting function
def custom_plot(data, **kwargs):
    """Custom plotting function for FacetGrid"""
    sns.scatterplot(data=data, x='x', y='y', alpha=0.6, **kwargs)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)

# Create FacetGrid with custom function
g = sns.FacetGrid(df, col='category', row='group', height=4, aspect=1.2)
g.map_dataframe(custom_plot)
g.set_titles(col_template='{col_name}', row_template='{row_name}')
g.fig.suptitle('Custom Function FacetGrid', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

## PairGrid

### Basic PairGrid

```python
# Generate multivariate data
np.random.seed(42)
n_samples = 100

# Create correlated variables
data = np.random.multivariate_normal(
    [0, 0, 0], 
    [[1, 0.7, 0.3], [0.7, 1, 0.5], [0.3, 0.5, 1]], 
    n_samples
)

# Create DataFrame
df_pair = pd.DataFrame(data, columns=['X1', 'X2', 'X3'])

# Create basic PairGrid
g = sns.PairGrid(df_pair, height=3)
g.map_upper(sns.scatterplot, alpha=0.6)
g.map_lower(sns.kdeplot)
g.map_diag(sns.histplot, kde=True)
g.fig.suptitle('Basic PairGrid', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### PairGrid with Different Plot Types

```python
# Create PairGrid with different plot types
g = sns.PairGrid(df_pair, height=3)

# Upper triangle: scatter plots
g.map_upper(sns.scatterplot, alpha=0.6, s=50)

# Lower triangle: regression plots
g.map_lower(sns.regplot, scatter_kws={'alpha': 0.6, 's': 30})

# Diagonal: histograms with KDE
g.map_diag(sns.histplot, kde=True, bins=20)

g.fig.suptitle('PairGrid with Different Plot Types', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### PairGrid with Categorical Variable

```python
# Add categorical variable
df_pair['Category'] = np.random.choice(['A', 'B', 'C'], n_samples)

# Create PairGrid with categorical coloring
g = sns.PairGrid(df_pair, hue='Category', height=3)
g.map_upper(sns.scatterplot, alpha=0.6)
g.map_lower(sns.kdeplot)
g.map_diag(sns.histplot, kde=True)
g.add_legend()
g.fig.suptitle('PairGrid with Categorical Variable', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### PairGrid with Correlation Coefficients

```python
# Create PairGrid with correlation coefficients
def corrfunc(x, y, **kws):
    """Add correlation coefficient to plot"""
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate(f'r = {r:.2f}', xy=(0.05, 0.95), xycoords=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

g = sns.PairGrid(df_pair, height=3)
g.map_upper(sns.scatterplot, alpha=0.6)
g.map_lower(corrfunc)
g.map_diag(sns.histplot, kde=True)
g.fig.suptitle('PairGrid with Correlation Coefficients', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

## JointGrid

### Basic JointGrid

```python
# Create basic JointGrid
g = sns.JointGrid(data=df_pair, x='X1', y='X2', height=8)
g.plot_joint(sns.scatterplot, alpha=0.6)
g.plot_marginals(sns.histplot, kde=True)
g.fig.suptitle('Basic JointGrid', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### JointGrid with Different Plot Types

```python
# Create JointGrid with different plot types
g = sns.JointGrid(data=df_pair, x='X1', y='X2', height=8)

# Joint plot: hexbin
g.plot_joint(sns.kdeplot, fill=True, alpha=0.6)

# Marginal plots: histograms
g.plot_marginals(sns.histplot, kde=True, bins=20)

g.fig.suptitle('JointGrid with KDE and Histograms', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### JointGrid with Regression

```python
# Create JointGrid with regression
g = sns.JointGrid(data=df_pair, x='X1', y='X2', height=8)

# Joint plot: regression
g.plot_joint(sns.regplot, scatter_kws={'alpha': 0.6})

# Marginal plots: histograms
g.plot_marginals(sns.histplot, kde=True)

g.fig.suptitle('JointGrid with Regression', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### JointGrid with Multiple Variables

```python
# Create multiple JointGrids
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# X1 vs X2
g1 = sns.JointGrid(data=df_pair, x='X1', y='X2', height=6)
g1.plot_joint(sns.scatterplot, alpha=0.6)
g1.plot_marginals(sns.histplot, kde=True)
g1.ax_joint.set_title('X1 vs X2')

# X1 vs X3
g2 = sns.JointGrid(data=df_pair, x='X1', y='X3', height=6)
g2.plot_joint(sns.scatterplot, alpha=0.6)
g2.plot_marginals(sns.histplot, kde=True)
g2.ax_joint.set_title('X1 vs X3')

# X2 vs X3
g3 = sns.JointGrid(data=df_pair, x='X2', y='X3', height=6)
g3.plot_joint(sns.scatterplot, alpha=0.6)
g3.plot_marginals(sns.histplot, kde=True)
g3.ax_joint.set_title('X2 vs X3')

plt.suptitle('Multiple JointGrids', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Complex Multi-Panel Layouts

### Combined Grid Types

```python
# Create complex multi-panel layout
fig = plt.figure(figsize=(20, 12))

# Create grid layout
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Top row: FacetGrid
ax1 = fig.add_subplot(gs[0, :2])
g_facet = sns.FacetGrid(df, col='category', height=4, aspect=1.5)
g_facet.map_dataframe(sns.scatterplot, x='x', y='y', alpha=0.6)
g_facet.set_titles(col_template='{col_name}')

# Top right: PairGrid
ax2 = fig.add_subplot(gs[0, 2:])
g_pair = sns.PairGrid(df_pair[['X1', 'X2']], height=4)
g_pair.map_upper(sns.scatterplot, alpha=0.6)
g_pair.map_lower(sns.kdeplot)
g_pair.map_diag(sns.histplot, kde=True)

# Middle row: JointGrids
ax3 = fig.add_subplot(gs[1, :2])
g_joint1 = sns.JointGrid(data=df_pair, x='X1', y='X2', height=4)
g_joint1.plot_joint(sns.scatterplot, alpha=0.6)
g_joint1.plot_marginals(sns.histplot, kde=True)

ax4 = fig.add_subplot(gs[1, 2:])
g_joint2 = sns.JointGrid(data=df_pair, x='X1', y='X3', height=4)
g_joint2.plot_joint(sns.scatterplot, alpha=0.6)
g_joint2.plot_marginals(sns.histplot, kde=True)

# Bottom row: Individual plots
ax5 = fig.add_subplot(gs[2, 0])
sns.boxplot(data=df, x='category', y='y', ax=ax5)
ax5.set_title('Box Plot')

ax6 = fig.add_subplot(gs[2, 1])
sns.violinplot(data=df, x='category', y='y', ax=ax6)
ax6.set_title('Violin Plot')

ax7 = fig.add_subplot(gs[2, 2])
sns.histplot(data=df_pair['X1'], kde=True, ax=ax7)
ax7.set_title('X1 Distribution')

ax8 = fig.add_subplot(gs[2, 3])
sns.scatterplot(data=df_pair, x='X2', y='X3', alpha=0.6, ax=ax8)
ax8.set_title('X2 vs X3')

plt.suptitle('Complex Multi-Panel Layout', fontsize=16, fontweight='bold')
plt.show()
```

### Conditional Plotting

```python
# Create conditional plots based on data characteristics
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Conditional on category
for i, category in enumerate(['A', 'B']):
    subset = df[df['category'] == category]
    
    # Scatter plot
    sns.scatterplot(data=subset, x='x', y='y', hue='group', 
                   alpha=0.6, ax=axes[0, i])
    axes[0, i].set_title(f'Category {category}')
    axes[0, i].set_xlabel('X')
    axes[0, i].set_ylabel('Y')

# Conditional on group
for i, group in enumerate(['Group 1', 'Group 2']):
    subset = df[df['group'] == group]
    
    # Box plot
    sns.boxplot(data=subset, x='category', y='y', ax=axes[1, i])
    axes[1, i].set_title(f'{group}')
    axes[1, i].set_xlabel('Category')
    axes[1, i].set_ylabel('Y')

plt.suptitle('Conditional Plotting', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Custom Grid Functions

### Custom FacetGrid Function

```python
def create_custom_facetgrid(df, x_col, y_col, facet_col, plot_type='scatter'):
    """Create custom FacetGrid with specified plot type"""
    
    if plot_type == 'scatter':
        g = sns.FacetGrid(df, col=facet_col, height=4, aspect=1.2)
        g.map_dataframe(sns.scatterplot, x=x_col, y=y_col, alpha=0.6)
    elif plot_type == 'box':
        g = sns.FacetGrid(df, col=facet_col, height=4, aspect=1.2)
        g.map_dataframe(sns.boxplot, x='group', y=y_col)
    elif plot_type == 'hist':
        g = sns.FacetGrid(df, col=facet_col, height=4, aspect=1.2)
        g.map_dataframe(sns.histplot, x=x_col, bins=20)
    elif plot_type == 'reg':
        g = sns.FacetGrid(df, col=facet_col, height=4, aspect=1.2)
        g.map_dataframe(sns.regplot, x=x_col, y=y_col, scatter_kws={'alpha': 0.6})
    
    g.set_titles(col_template='{col_name}')
    return g

# Use custom function
g = create_custom_facetgrid(df, 'x', 'y', 'category', 'scatter')
g.fig.suptitle('Custom FacetGrid Function', y=1.02, fontsize=14, fontweight='bold')
plt.show()

g = create_custom_facetgrid(df, 'x', 'y', 'category', 'reg')
g.fig.suptitle('Custom FacetGrid Function - Regression', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### Custom PairGrid Function

```python
def create_custom_pairgrid(df, vars_list, plot_type='scatter'):
    """Create custom PairGrid with specified plot type"""
    
    g = sns.PairGrid(df[vars_list], height=3)
    
    if plot_type == 'scatter':
        g.map_upper(sns.scatterplot, alpha=0.6)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.histplot, kde=True)
    elif plot_type == 'reg':
        g.map_upper(sns.scatterplot, alpha=0.6)
        g.map_lower(sns.regplot, scatter_kws={'alpha': 0.6})
        g.map_diag(sns.histplot, kde=True)
    elif plot_type == 'hex':
        g.map_upper(sns.scatterplot, alpha=0.6)
        g.map_lower(sns.kdeplot, fill=True)
        g.map_diag(sns.histplot, kde=True)
    
    return g

# Use custom function
g = create_custom_pairgrid(df_pair, ['X1', 'X2', 'X3'], 'scatter')
g.fig.suptitle('Custom PairGrid Function', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

## Advanced Grid Techniques

### Dynamic Grid Creation

```python
def create_dynamic_grid(df, variables, grid_type='facet', **kwargs):
    """Create dynamic grid based on data characteristics"""
    
    if grid_type == 'facet':
        # Determine best facet variable
        categorical_vars = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_vars) > 0:
            facet_var = categorical_vars[0]
            g = sns.FacetGrid(df, col=facet_var, **kwargs)
            g.map_dataframe(sns.scatterplot, x=variables[0], y=variables[1], alpha=0.6)
            return g
        else:
            # No categorical variables, create simple plot
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=df, x=variables[0], y=variables[1], alpha=0.6)
            return ax
    
    elif grid_type == 'pair':
        g = sns.PairGrid(df[variables], **kwargs)
        g.map_upper(sns.scatterplot, alpha=0.6)
        g.map_lower(sns.kdeplot)
        g.map_diag(sns.histplot, kde=True)
        return g
    
    elif grid_type == 'joint':
        g = sns.JointGrid(data=df, x=variables[0], y=variables[1], **kwargs)
        g.plot_joint(sns.scatterplot, alpha=0.6)
        g.plot_marginals(sns.histplot, kde=True)
        return g

# Use dynamic grid creation
g = create_dynamic_grid(df, ['x', 'y'], 'facet', height=4, aspect=1.2)
if hasattr(g, 'fig'):
    g.fig.suptitle('Dynamic FacetGrid', y=1.02, fontsize=14, fontweight='bold')
else:
    g.set_title('Dynamic Plot (No Categorical Variables)')
plt.show()
```

### Interactive Grid Elements

```python
# Create grid with interactive elements
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Scatter with trend line
sns.scatterplot(data=df, x='x', y='y', hue='category', alpha=0.6, ax=axes[0, 0])
# Add trend line
z = np.polyfit(df['x'], df['y'], 1)
p = np.poly1d(z)
axes[0, 0].plot(df['x'], p(df['x']), "r--", alpha=0.8, linewidth=2)
axes[0, 0].set_title('Scatter with Trend Line')

# Plot 2: Box plot with individual points
sns.boxplot(data=df, x='category', y='y', ax=axes[0, 1])
sns.stripplot(data=df, x='category', y='y', color='red', alpha=0.5, size=4, ax=axes[0, 1])
axes[0, 1].set_title('Box Plot with Individual Points')

# Plot 3: Histogram with KDE
sns.histplot(data=df, x='x', hue='category', kde=True, alpha=0.6, ax=axes[1, 0])
axes[1, 0].set_title('Histogram with KDE')

# Plot 4: Violin plot with box plot inside
sns.violinplot(data=df, x='category', y='y', inner='box', ax=axes[1, 1])
axes[1, 1].set_title('Violin Plot with Box Plot Inside')

plt.suptitle('Interactive Grid Elements', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Statistical Annotations in Grids

```python
# Create grid with statistical annotations
from scipy.stats import ttest_ind

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Scatter with correlation
sns.scatterplot(data=df_pair, x='X1', y='X2', alpha=0.6, ax=axes[0, 0])
corr = df_pair['X1'].corr(df_pair['X2'])
axes[0, 0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0, 0].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
axes[0, 0].set_title('Scatter with Correlation')

# Plot 2: Box plot with t-test
sns.boxplot(data=df, x='category', y='y', ax=axes[0, 1])
# Perform t-test between categories
cat_a = df[df['category'] == 'A']['y']
cat_b = df[df['category'] == 'B']['y']
t_stat, p_val = ttest_ind(cat_a, cat_b)
axes[0, 1].text(0.5, 0.95, f't-test: p = {p_val:.3f}', 
                transform=axes[0, 1].transAxes, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
axes[0, 1].set_title('Box Plot with T-test')

# Plot 3: Histogram with statistics
sns.histplot(data=df_pair['X1'], kde=True, ax=axes[1, 0])
mean_val = df_pair['X1'].mean()
std_val = df_pair['X1'].std()
axes[1, 0].axvline(mean_val, color='red', linestyle='--', alpha=0.8)
axes[1, 0].text(0.05, 0.95, f'μ = {mean_val:.2f}\nσ = {std_val:.2f}', 
                transform=axes[1, 0].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
axes[1, 0].set_title('Histogram with Statistics')

# Plot 4: Regression with R²
sns.regplot(data=df_pair, x='X1', y='X2', scatter_kws={'alpha': 0.6}, ax=axes[1, 1])
r_squared = df_pair['X1'].corr(df_pair['X2']) ** 2
axes[1, 1].text(0.05, 0.95, f'R² = {r_squared:.3f}', 
                transform=axes[1, 1].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
axes[1, 1].set_title('Regression with R²')

plt.suptitle('Statistical Annotations in Grids', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Summary

This guide covered comprehensive multi-plot grid techniques with Seaborn:

1. **FacetGrid**: Creating conditional plots based on categorical variables
2. **PairGrid**: Multi-variable relationship analysis with customizable plot types
3. **JointGrid**: Combined univariate and bivariate analysis
4. **Complex Layouts**: Advanced multi-panel arrangements
5. **Custom Functions**: Reusable grid creation functions
6. **Advanced Techniques**: Dynamic grids and statistical annotations

These multi-plot grid techniques are essential for:
- **Data Exploration**: Comprehensive analysis of complex datasets
- **Conditional Analysis**: Understanding relationships across different groups
- **Publication Graphics**: Creating professional multi-panel figures
- **Statistical Analysis**: Combining multiple visualization approaches

## Best Practices

1. **Choose Appropriate Grid Types**: Use FacetGrid for categorical conditioning, PairGrid for relationships
2. **Maintain Consistency**: Use consistent styling across all panels
3. **Consider Layout**: Arrange panels logically and use appropriate sizes
4. **Add Context**: Include titles, labels, and statistical annotations
5. **Test Different Views**: Try multiple grid arrangements to find the best representation

## Next Steps

- Explore advanced statistical testing and significance analysis
- Learn about interactive visualizations and dashboards
- Master publication-quality figure creation
- Practice with real-world complex datasets
- Customize grids for specific analysis requirements

Remember: Multi-plot grids are powerful tools for comprehensive data analysis and storytelling! 