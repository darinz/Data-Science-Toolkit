# Categorical Plots Guide

A comprehensive guide to creating categorical data visualizations with Seaborn, covering bar plots, box plots, violin plots, and more.

## Table of Contents

1. [Bar Plots](#bar-plots)
2. [Count Plots](#count-plots)
3. [Box Plots](#box-plots)
4. [Violin Plots](#violin-plots)
5. [Strip Plots](#strip-plots)
6. [Swarm Plots](#swarm-plots)
7. [Point Plots](#point-plots)
8. [Factor Plots](#factor-plots)
9. [Categorical Regression](#categorical-regression)
10. [Advanced Categorical Plots](#advanced-categorical-plots)

## Bar Plots

### Basic Bar Plot

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up the plotting style
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)

# Create sample categorical data
categories = ['A', 'B', 'C', 'D', 'E']
values = np.random.rand(5)

# Create basic bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=categories, y=values, palette='viridis')
plt.title('Basic Bar Plot', fontsize=14, fontweight='bold')
plt.xlabel('Categories', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.show()
```

### Horizontal Bar Plot

```python
# Create horizontal bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=values, y=categories, palette='Set3', orient='h')
plt.title('Horizontal Bar Plot', fontsize=14, fontweight='bold')
plt.xlabel('Values', fontsize=12)
plt.ylabel('Categories', fontsize=12)
plt.show()
```

### Grouped Bar Plot

```python
# Create sample data with groups
np.random.seed(42)
data = {
    'category': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'],
    'group': ['Group 1', 'Group 2'] * 4,
    'value': np.random.rand(8)
}
df_grouped = pd.DataFrame(data)

# Create grouped bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=df_grouped, x='category', y='value', hue='group', palette='Set2')
plt.title('Grouped Bar Plot', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(title='Group', title_fontsize=12)
plt.show()
```

### Bar Plot with Error Bars

```python
# Create sample data with multiple observations per category
np.random.seed(42)
n_obs = 50
data = []
for cat in ['A', 'B', 'C', 'D']:
    for _ in range(n_obs):
        data.append({
            'category': cat,
            'value': np.random.normal(ord(cat) - ord('A') + 1, 0.5)
        })
df_error = pd.DataFrame(data)

# Create bar plot with error bars
plt.figure(figsize=(10, 6))
sns.barplot(data=df_error, x='category', y='value', palette='rocket')
plt.title('Bar Plot with Error Bars', fontsize=14, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.show()
```

## Count Plots

### Basic Count Plot

```python
# Create sample categorical data
np.random.seed(42)
categories = np.random.choice(['Red', 'Blue', 'Green', 'Yellow', 'Purple'], 100)

# Create count plot
plt.figure(figsize=(10, 6))
sns.countplot(data=pd.DataFrame({'Color': categories}), x='Color', palette='Set1')
plt.title('Count Plot', fontsize=14, fontweight='bold')
plt.xlabel('Color', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.show()
```

### Horizontal Count Plot

```python
# Create horizontal count plot
plt.figure(figsize=(10, 6))
sns.countplot(data=pd.DataFrame({'Color': categories}), y='Color', palette='Set1')
plt.title('Horizontal Count Plot', fontsize=14, fontweight='bold')
plt.xlabel('Count', fontsize=12)
plt.ylabel('Color', fontsize=12)
plt.show()
```

### Grouped Count Plot

```python
# Create sample data with groups
np.random.seed(42)
data = []
for color in ['Red', 'Blue', 'Green']:
    for group in ['Group 1', 'Group 2']:
        count = np.random.randint(10, 30)
        data.extend([{'Color': color, 'Group': group}] * count)

df_count = pd.DataFrame(data)

# Create grouped count plot
plt.figure(figsize=(12, 6))
sns.countplot(data=df_count, x='Color', hue='Group', palette='Set2')
plt.title('Grouped Count Plot', fontsize=14, fontweight='bold')
plt.xlabel('Color', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Group', title_fontsize=12)
plt.show()
```

## Box Plots

### Basic Box Plot

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

# Create basic box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Group', y='Value', palette='viridis')
plt.title('Basic Box Plot', fontsize=14, fontweight='bold')
plt.xlabel('Group', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.show()
```

### Horizontal Box Plot

```python
# Create horizontal box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Value', y='Group', palette='viridis', orient='h')
plt.title('Horizontal Box Plot', fontsize=14, fontweight='bold')
plt.xlabel('Value', fontsize=12)
plt.ylabel('Group', fontsize=12)
plt.show()
```

### Grouped Box Plot

```python
# Create sample data with nested groups
np.random.seed(42)
data_nested = []
for group in ['A', 'B']:
    for subgroup in ['X', 'Y']:
        for _ in range(30):
            data_nested.append({
                'Group': group,
                'Subgroup': subgroup,
                'Value': np.random.normal(ord(group) - ord('A') + ord(subgroup) - ord('X'), 0.5)
            })

df_nested = pd.DataFrame(data_nested)

# Create grouped box plot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_nested, x='Group', y='Value', hue='Subgroup', palette='Set2')
plt.title('Grouped Box Plot', fontsize=14, fontweight='bold')
plt.xlabel('Group', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(title='Subgroup', title_fontsize=12)
plt.show()
```

### Box Plot with Outliers

```python
# Create data with outliers
np.random.seed(42)
normal_data = np.random.normal(0, 1, 100)
outlier_data = np.concatenate([normal_data, [5, -5, 8, -8]])

# Create box plot showing outliers
plt.figure(figsize=(10, 6))
sns.boxplot(data=outlier_data, palette='lightblue')
plt.title('Box Plot with Outliers', fontsize=14, fontweight='bold')
plt.ylabel('Value', fontsize=12)
plt.show()
```

## Violin Plots

### Basic Violin Plot

```python
# Create basic violin plot
plt.figure(figsize=(10, 6))
sns.violinplot(data=data, x='Group', y='Value', palette='viridis')
plt.title('Basic Violin Plot', fontsize=14, fontweight='bold')
plt.xlabel('Group', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.show()
```

### Violin Plot with Box Plot Inside

```python
# Create violin plot with box plot inside
plt.figure(figsize=(10, 6))
sns.violinplot(data=data, x='Group', y='Value', palette='viridis', inner='box')
plt.title('Violin Plot with Box Plot Inside', fontsize=14, fontweight='bold')
plt.xlabel('Group', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.show()
```

### Split Violin Plot

```python
# Create split violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=df_nested, x='Group', y='Value', hue='Subgroup', 
               palette='Set2', split=True)
plt.title('Split Violin Plot', fontsize=14, fontweight='bold')
plt.xlabel('Group', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(title='Subgroup', title_fontsize=12)
plt.show()
```

## Strip Plots

### Basic Strip Plot

```python
# Create basic strip plot
plt.figure(figsize=(10, 6))
sns.stripplot(data=data, x='Group', y='Value', palette='viridis', alpha=0.7)
plt.title('Basic Strip Plot', fontsize=14, fontweight='bold')
plt.xlabel('Group', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.show()
```

### Jittered Strip Plot

```python
# Create jittered strip plot
plt.figure(figsize=(10, 6))
sns.stripplot(data=data, x='Group', y='Value', palette='viridis', 
              jitter=0.3, alpha=0.7, size=4)
plt.title('Jittered Strip Plot', fontsize=14, fontweight='bold')
plt.xlabel('Group', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.show()
```

### Grouped Strip Plot

```python
# Create grouped strip plot
plt.figure(figsize=(12, 6))
sns.stripplot(data=df_nested, x='Group', y='Value', hue='Subgroup', 
              palette='Set2', alpha=0.7, jitter=0.2)
plt.title('Grouped Strip Plot', fontsize=14, fontweight='bold')
plt.xlabel('Group', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(title='Subgroup', title_fontsize=12)
plt.show()
```

## Swarm Plots

### Basic Swarm Plot

```python
# Create basic swarm plot
plt.figure(figsize=(10, 6))
sns.swarmplot(data=data, x='Group', y='Value', palette='viridis', size=4)
plt.title('Basic Swarm Plot', fontsize=14, fontweight='bold')
plt.xlabel('Group', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.show()
```

### Swarm Plot with Box Plot

```python
# Create swarm plot with box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Group', y='Value', palette='lightgray', alpha=0.5)
sns.swarmplot(data=data, x='Group', y='Value', palette='viridis', size=4)
plt.title('Swarm Plot with Box Plot', fontsize=14, fontweight='bold')
plt.xlabel('Group', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.show()
```

### Grouped Swarm Plot

```python
# Create grouped swarm plot
plt.figure(figsize=(12, 6))
sns.swarmplot(data=df_nested, x='Group', y='Value', hue='Subgroup', 
              palette='Set2', size=3)
plt.title('Grouped Swarm Plot', fontsize=14, fontweight='bold')
plt.xlabel('Group', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(title='Subgroup', title_fontsize=12)
plt.show()
```

## Point Plots

### Basic Point Plot

```python
# Create basic point plot
plt.figure(figsize=(10, 6))
sns.pointplot(data=data, x='Group', y='Value', palette='viridis')
plt.title('Basic Point Plot', fontsize=14, fontweight='bold')
plt.xlabel('Group', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.show()
```

### Point Plot with Error Bars

```python
# Create point plot with error bars
plt.figure(figsize=(10, 6))
sns.pointplot(data=data, x='Group', y='Value', palette='viridis', 
              capsize=0.2, markers='o', scale=0.8)
plt.title('Point Plot with Error Bars', fontsize=14, fontweight='bold')
plt.xlabel('Group', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.show()
```

### Grouped Point Plot

```python
# Create grouped point plot
plt.figure(figsize=(12, 6))
sns.pointplot(data=df_nested, x='Group', y='Value', hue='Subgroup', 
              palette='Set2', capsize=0.2)
plt.title('Grouped Point Plot', fontsize=14, fontweight='bold')
plt.xlabel('Group', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend(title='Subgroup', title_fontsize=12)
plt.show()
```

## Factor Plots

### Basic Factor Plot

```python
# Create basic factor plot (now catplot)
plt.figure(figsize=(10, 6))
sns.catplot(data=data, x='Group', y='Value', kind='bar', palette='viridis', height=6)
plt.suptitle('Factor Plot (Bar)', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

### Multiple Factor Plots

```python
# Create multiple factor plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Bar plot
sns.barplot(data=data, x='Group', y='Value', ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('Bar Plot')

# Box plot
sns.boxplot(data=data, x='Group', y='Value', ax=axes[0, 1], palette='viridis')
axes[0, 1].set_title('Box Plot')

# Violin plot
sns.violinplot(data=data, x='Group', y='Value', ax=axes[1, 0], palette='viridis')
axes[1, 0].set_title('Violin Plot')

# Point plot
sns.pointplot(data=data, x='Group', y='Value', ax=axes[1, 1], palette='viridis')
axes[1, 1].set_title('Point Plot')

plt.suptitle('Multiple Categorical Plots', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Categorical Regression

### Categorical Regression with Error Bars

```python
# Create sample data for categorical regression
np.random.seed(42)
n_samples = 200
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
df_cat_reg = pd.DataFrame({'x': x, 'y': y, 'category': category})

# Create categorical regression plot
plt.figure(figsize=(10, 6))
sns.lmplot(data=df_cat_reg, x='x', y='y', hue='category', 
           scatter_kws={'alpha': 0.6}, line_kws={'linewidth': 2})

plt.suptitle('Categorical Regression', y=1.02, fontsize=14, fontweight='bold')
plt.show()
```

## Advanced Categorical Plots

### Combined Categorical Plots

```python
# Create a comprehensive categorical analysis
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Bar plot
sns.barplot(data=data, x='Group', y='Value', ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('Bar Plot')

# Box plot
sns.boxplot(data=data, x='Group', y='Value', ax=axes[0, 1], palette='viridis')
axes[0, 1].set_title('Box Plot')

# Violin plot
sns.violinplot(data=data, x='Group', y='Value', ax=axes[0, 2], palette='viridis')
axes[0, 2].set_title('Violin Plot')

# Strip plot
sns.stripplot(data=data, x='Group', y='Value', ax=axes[1, 0], palette='viridis', alpha=0.7)
axes[1, 0].set_title('Strip Plot')

# Swarm plot
sns.swarmplot(data=data, x='Group', y='Value', ax=axes[1, 1], palette='viridis', size=4)
axes[1, 1].set_title('Swarm Plot')

# Point plot
sns.pointplot(data=data, x='Group', y='Value', ax=axes[1, 2], palette='viridis')
axes[1, 2].set_title('Point Plot')

plt.suptitle('Comprehensive Categorical Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Statistical Annotations in Categorical Plots

```python
from scipy.stats import ttest_ind

# Perform statistical test
group1_data = data[data['Group'] == 'Group 1']['Value']
group2_data = data[data['Group'] == 'Group 2']['Value']
t_stat, p_value = ttest_ind(group1_data, group2_data)

# Create box plot with statistical annotation
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Group', y='Value', palette='viridis')

plt.title('Box Plot with Statistical Comparison', fontsize=14, fontweight='bold')
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

## Summary

This guide covered comprehensive categorical plotting with Seaborn:

1. **Bar Plots**: Basic, horizontal, and grouped bar plots
2. **Count Plots**: Frequency analysis of categorical variables
3. **Box Plots**: Distribution summaries with quartiles and outliers
4. **Violin Plots**: Density-based distribution visualization
5. **Strip Plots**: Individual data point visualization
6. **Swarm Plots**: Non-overlapping point plots
7. **Point Plots**: Mean estimates with error bars
8. **Factor Plots**: Multi-plot categorical analysis
9. **Categorical Regression**: Relationship analysis by category
10. **Advanced Plots**: Combined and annotated visualizations

These categorical plots are essential for:
- **Categorical Data Analysis**: Understanding discrete variable distributions
- **Group Comparisons**: Statistical comparisons between categories
- **Data Exploration**: Quick insights into categorical relationships
- **Publication Graphics**: Professional categorical visualizations

## Best Practices

1. **Choose Appropriate Plot Types**: Select plots based on your data and question
2. **Use Consistent Styling**: Maintain visual consistency across plots
3. **Add Statistical Context**: Include significance tests and confidence intervals
4. **Consider Your Audience**: Adjust complexity based on viewer expertise
5. **Document Your Choices**: Explain why you chose specific visualizations

## Next Steps

- Explore distribution analysis for continuous variables
- Learn about correlation analysis and heatmaps
- Master multi-plot grids for complex visualizations
- Practice with real-world categorical datasets
- Customize plots for specific publication requirements

Remember: Categorical plots are powerful tools for understanding discrete data relationships and making data-driven decisions! 