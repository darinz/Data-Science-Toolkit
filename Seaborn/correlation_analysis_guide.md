# Correlation Analysis Guide

A comprehensive guide to correlation analysis with Seaborn, covering correlation matrices, heatmaps, pair plots, and statistical significance testing.

## Table of Contents

1. [Correlation Matrices](#correlation-matrices)
2. [Heatmaps](#heatmaps)
3. [Pair Plots](#pair-plots)
4. [Clustermaps](#clustermaps)
5. [Correlation Significance](#correlation-significance)
6. [Advanced Correlation Analysis](#advanced-correlation-analysis)

## Correlation Matrices

### Basic Correlation Matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Set up the plotting style
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)

# Generate sample correlated data
np.random.seed(42)
n_samples = 100

# Create correlated variables
data = np.random.multivariate_normal(
    [0, 0, 0, 0], 
    [[1, 0.8, 0.6, 0.2], 
     [0.8, 1, 0.4, 0.3], 
     [0.6, 0.4, 1, 0.1], 
     [0.2, 0.3, 0.1, 1]], 
    n_samples
)

# Create DataFrame
df = pd.DataFrame(data, columns=['Variable_A', 'Variable_B', 'Variable_C', 'Variable_D'])

# Calculate correlation matrix
correlation_matrix = df.corr()

# Display correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)
```

### Correlation Matrix Heatmap

```python
# Create correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})

plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Different Correlation Methods

```python
# Calculate different types of correlations
pearson_corr = df.corr(method='pearson')
spearman_corr = df.corr(method='spearman')
kendall_corr = df.corr(method='kendall')

# Create subplots for different correlation methods
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Pearson correlation
sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, ax=axes[0])
axes[0].set_title('Pearson Correlation')

# Spearman correlation
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, ax=axes[1])
axes[1].set_title('Spearman Correlation')

# Kendall correlation
sns.heatmap(kendall_corr, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, ax=axes[2])
axes[2].set_title('Kendall Correlation')

plt.suptitle('Different Correlation Methods', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Heatmaps

### Basic Heatmap

```python
# Create basic heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, cmap='viridis', annot=True, fmt='.2f')

plt.title('Basic Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Customized Heatmap

```python
# Create customized heatmap
plt.figure(figsize=(10, 8))

# Create mask for upper triangle (optional)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', 
            center=0, square=True, linewidths=0.5, 
            cbar_kws={"shrink": 0.8}, fmt='.2f')

plt.title('Customized Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Heatmap with Different Color Schemes

```python
# Create heatmaps with different color schemes
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Coolwarm
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, ax=axes[0, 0], fmt='.2f')
axes[0, 0].set_title('Coolwarm')

# RdBu_r
sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, 
            square=True, linewidths=0.5, ax=axes[0, 1], fmt='.2f')
axes[0, 1].set_title('RdBu_r')

# Viridis
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', 
            square=True, linewidths=0.5, ax=axes[1, 0], fmt='.2f')
axes[1, 0].set_title('Viridis')

# Plasma
sns.heatmap(correlation_matrix, annot=True, cmap='plasma', 
            square=True, linewidths=0.5, ax=axes[1, 1], fmt='.2f')
axes[1, 1].set_title('Plasma')

plt.suptitle('Heatmaps with Different Color Schemes', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Pair Plots

### Basic Pair Plot

```python
# Create basic pair plot
sns.pairplot(df, diag_kind='kde')
plt.suptitle('Pair Plot', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### Pair Plot with Correlation Coefficients

```python
# Create pair plot with correlation coefficients
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate(f'r = {r:.2f}', xy=(0.05, 0.95), xycoords=ax.transAxes)

g = sns.pairplot(df, diag_kind='kde')
g.map_lower(corrfunc)
plt.suptitle('Pair Plot with Correlation Coefficients', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### Pair Plot with Different Plot Types

```python
# Create pair plot with different plot types
sns.pairplot(df, diag_kind='hist', plot_kws={'alpha': 0.6})
plt.suptitle('Pair Plot with Histograms', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### Pair Plot with Categorical Variable

```python
# Add categorical variable to the data
df['Category'] = np.random.choice(['A', 'B', 'C'], n_samples)

# Create pair plot with categorical coloring
sns.pairplot(df, hue='Category', diag_kind='kde')
plt.suptitle('Pair Plot with Categorical Variable', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

## Clustermaps

### Basic Clustermap

```python
# Create basic clustermap
plt.figure(figsize=(10, 8))
sns.clustermap(correlation_matrix, cmap='coolwarm', center=0, 
               square=True, linewidths=0.5, figsize=(10, 8))

plt.suptitle('Clustermap', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### Clustermap with Customization

```python
# Create customized clustermap
plt.figure(figsize=(10, 8))
sns.clustermap(correlation_matrix, cmap='RdBu_r', center=0, 
               square=True, linewidths=0.5, 
               cbar_kws={"shrink": 0.8}, 
               dendrogram_ratio=(0.1, 0.2),
               figsize=(10, 8))

plt.suptitle('Customized Clustermap', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### Clustermap with Row and Column Clustering

```python
# Create clustermap with different clustering options
plt.figure(figsize=(10, 8))
sns.clustermap(correlation_matrix, cmap='viridis', 
               square=True, linewidths=0.5,
               row_cluster=True, col_cluster=True,
               figsize=(10, 8))

plt.suptitle('Clustermap with Row and Column Clustering', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

## Correlation Significance

### Correlation with P-values

```python
# Calculate correlation with p-values
def correlation_with_pvalues(df):
    """Calculate correlation matrix with p-values"""
    corr_matrix = df.corr()
    p_matrix = pd.DataFrame(index=df.columns, columns=df.columns)
    
    for i in df.columns:
        for j in df.columns:
            if i == j:
                p_matrix.loc[i, j] = 1.0
            else:
                r, p = stats.pearsonr(df[i], df[j])
                p_matrix.loc[i, j] = p
    
    return corr_matrix, p_matrix

corr_matrix, p_matrix = correlation_with_pvalues(df)

print("Correlation Matrix:")
print(corr_matrix)
print("\nP-values Matrix:")
print(p_matrix)
```

### Heatmap with Significance Stars

```python
# Create heatmap with significance stars
def add_significance_stars(ax, corr_matrix, p_matrix):
    """Add significance stars to heatmap"""
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            if i != j:
                p_val = p_matrix.iloc[i, j]
                if p_val < 0.001:
                    star = '***'
                elif p_val < 0.01:
                    star = '**'
                elif p_val < 0.05:
                    star = '*'
                else:
                    star = ''
                
                text = f"{corr_matrix.iloc[i, j]:.2f}{star}"
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center')

# Create heatmap with significance
plt.figure(figsize=(10, 8))
ax = sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                 square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})

add_significance_stars(ax, corr_matrix, p_matrix)

plt.title('Correlation Heatmap with Significance Stars\n* p<0.05, ** p<0.01, *** p<0.001', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Correlation Confidence Intervals

```python
# Calculate correlation confidence intervals
def correlation_ci(x, y, confidence=0.95):
    """Calculate correlation confidence interval"""
    r, p = stats.pearsonr(x, y)
    n = len(x)
    
    # Fisher's z transformation
    z = np.arctanh(r)
    
    # Standard error
    se = 1 / np.sqrt(n - 3)
    
    # Confidence interval
    z_score = stats.norm.ppf((1 + confidence) / 2)
    ci_lower = z - z_score * se
    ci_upper = z + z_score * se
    
    # Transform back to correlation scale
    r_lower = np.tanh(ci_lower)
    r_upper = np.tanh(ci_upper)
    
    return r, r_lower, r_upper, p

# Calculate confidence intervals for all pairs
ci_results = {}
for i in range(len(df.columns)):
    for j in range(i+1, len(df.columns)):
        var1, var2 = df.columns[i], df.columns[j]
        r, r_lower, r_upper, p = correlation_ci(df[var1], df[var2])
        ci_results[f"{var1}-{var2}"] = {
            'correlation': r,
            'ci_lower': r_lower,
            'ci_upper': r_upper,
            'p_value': p
        }

# Display results
print("Correlation Confidence Intervals (95%):")
for pair, result in ci_results.items():
    print(f"{pair}: r = {result['correlation']:.3f} "
          f"({result['ci_lower']:.3f}, {result['ci_upper']:.3f}), "
          f"p = {result['p_value']:.3f}")
```

## Advanced Correlation Analysis

### Partial Correlation

```python
# Calculate partial correlations
def partial_correlation(df):
    """Calculate partial correlation matrix"""
    from scipy.stats import pearsonr
    
    n_vars = len(df.columns)
    partial_corr = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                partial_corr[i, j] = 1.0
            else:
                # Get all other variables
                other_vars = [k for k in range(n_vars) if k not in [i, j]]
                
                if len(other_vars) == 0:
                    # No control variables, use regular correlation
                    r, _ = pearsonr(df.iloc[:, i], df.iloc[:, j])
                    partial_corr[i, j] = r
                else:
                    # Calculate partial correlation
                    from scipy import linalg
                    
                    # Create correlation matrix for variables i, j, and controls
                    vars_subset = [i, j] + other_vars
                    corr_subset = df.iloc[:, vars_subset].corr().values
                    
                    # Calculate partial correlation
                    try:
                        inv_corr = linalg.inv(corr_subset)
                        partial_corr[i, j] = -inv_corr[0, 1] / np.sqrt(inv_corr[0, 0] * inv_corr[1, 1])
                    except:
                        partial_corr[i, j] = np.nan
    
    return pd.DataFrame(partial_corr, index=df.columns, columns=df.columns)

# Calculate partial correlations
partial_corr_matrix = partial_correlation(df)

# Create heatmap for partial correlations
plt.figure(figsize=(10, 8))
sns.heatmap(partial_corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})

plt.title('Partial Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Correlation Network

```python
# Create correlation network visualization
import networkx as nx

# Create network from correlation matrix
# Only include correlations above threshold
threshold = 0.3
network_matrix = correlation_matrix.copy()
network_matrix[abs(network_matrix) < threshold] = 0

# Create graph
G = nx.from_pandas_adjacency(network_matrix)

# Create network plot
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, k=1, iterations=50)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                      node_size=1000, alpha=0.7)

# Draw edges with correlation values as weights
edges = G.edges()
weights = [abs(G[u][v]['weight']) for u, v in edges]
colors = ['red' if G[u][v]['weight'] < 0 else 'blue' for u, v in edges]

nx.draw_networkx_edges(G, pos, width=weights, edge_color=colors, 
                      alpha=0.6, edge_cmap=plt.cm.RdYlBu)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

plt.title('Correlation Network (|r| > 0.3)', fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()
```

### Time Series Correlation

```python
# Create time series data for correlation analysis
np.random.seed(42)
n_timepoints = 100
time_series_data = pd.DataFrame({
    'Time': range(n_timepoints),
    'Series_A': np.cumsum(np.random.randn(n_timepoints)),
    'Series_B': np.cumsum(np.random.randn(n_timepoints)) * 0.7 + np.random.randn(n_timepoints) * 0.3,
    'Series_C': np.cumsum(np.random.randn(n_timepoints)) * 0.3 + np.random.randn(n_timepoints) * 0.7
})

# Calculate rolling correlation
rolling_corr = time_series_data[['Series_A', 'Series_B']].rolling(window=20).corr()

# Plot time series and rolling correlation
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Time series
axes[0].plot(time_series_data['Time'], time_series_data['Series_A'], label='Series A', linewidth=2)
axes[0].plot(time_series_data['Time'], time_series_data['Series_B'], label='Series B', linewidth=2)
axes[0].set_title('Time Series Data', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Value')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Rolling correlation
rolling_corr_ab = rolling_corr.loc[rolling_corr.index.get_level_values(1) == 'Series_B', 'Series_A']
axes[1].plot(time_series_data['Time'], rolling_corr_ab, linewidth=2, color='red')
axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1].set_title('Rolling Correlation (Window=20)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Correlation')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Time Series Correlation Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Summary

This guide covered comprehensive correlation analysis with Seaborn:

1. **Correlation Matrices**: Basic correlation calculations and visualization
2. **Heatmaps**: Various heatmap styles and customizations
3. **Pair Plots**: Multi-variable correlation visualization
4. **Clustermaps**: Hierarchical clustering of correlations
5. **Correlation Significance**: P-values and confidence intervals
6. **Advanced Analysis**: Partial correlations, networks, and time series

These correlation analysis techniques are essential for:
- **Data Exploration**: Understanding relationships between variables
- **Feature Selection**: Identifying important variables for modeling
- **Multicollinearity Detection**: Finding highly correlated predictors
- **Statistical Inference**: Testing correlation significance

## Best Practices

1. **Choose Appropriate Correlation Methods**: Use Pearson for linear, Spearman for monotonic relationships
2. **Check for Significance**: Always test correlation significance
3. **Consider Sample Size**: Larger samples provide more reliable correlations
4. **Look for Non-linear Relationships**: Correlation only captures linear relationships
5. **Use Multiple Visualizations**: Combine different plot types for comprehensive analysis

## Next Steps

- Explore multi-plot grids for complex visualizations
- Learn about regression analysis and model diagnostics
- Master statistical testing and significance analysis
- Practice with real-world datasets
- Customize plots for specific publication requirements

Remember: Correlation does not imply causation! Always consider the context and potential confounding variables. 