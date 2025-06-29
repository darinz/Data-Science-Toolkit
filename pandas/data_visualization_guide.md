# Pandas Data Visualization: A Comprehensive Guide

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-blue.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11+-blue.svg)](https://seaborn.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

## Table of Contents

1. [Introduction to Data Visualization](#introduction-to-data-visualization)
2. [Basic Plotting with Pandas](#basic-plotting-with-pandas)
3. [Statistical Plots](#statistical-plots)
4. [Distribution Plots](#distribution-plots)
5. [Relationship Plots](#relationship-plots)
6. [Categorical Plots](#categorical-plots)
7. [Time Series Plots](#time-series-plots)
8. [Advanced Customization](#advanced-customization)
9. [Best Practices](#best-practices)
10. [Common Pitfalls](#common-pitfalls)

## Introduction to Data Visualization

Data visualization is crucial for understanding patterns, trends, and insights in data. Pandas provides built-in plotting capabilities and integrates well with matplotlib and seaborn.

### Setup and Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create sample dataset
np.random.seed(42)
n_samples = 1000

data = {
    'employee_id': range(1, n_samples + 1),
    'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing', 'Finance'], n_samples),
    'position': np.random.choice(['Junior', 'Senior', 'Manager', 'Director'], n_samples),
    'age': np.random.normal(35, 10, n_samples).astype(int),
    'salary': np.random.normal(60000, 20000, n_samples).astype(int),
    'experience_years': np.random.exponential(5, n_samples).astype(int),
    'performance_rating': np.random.normal(3.5, 0.8, n_samples),
    'hire_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Boston', 'Seattle'], n_samples),
    'gender': np.random.choice(['M', 'F'], n_samples)
}

df = pd.DataFrame(data)

# Clean data
df['age'] = df['age'].clip(22, 65)
df['salary'] = df['salary'].clip(30000, 150000)
df['experience_years'] = df['experience_years'].clip(0, 25)
df['performance_rating'] = df['performance_rating'].clip(1, 5)

print(f"Dataset shape: {df.shape}")
```

## Basic Plotting with Pandas

### 1. Line Plots

```python
# Simple line plot
df['salary'].plot(kind='line', figsize=(10, 6))
plt.title('Salary Distribution Over Index')
plt.xlabel('Employee Index')
plt.ylabel('Salary')
plt.show()

# Line plot with rolling average
rolling_avg = df['salary'].rolling(window=50).mean()
df['salary'].plot(figsize=(10, 6), alpha=0.5, label='Individual Salaries')
rolling_avg.plot(label='50-point Rolling Average')
plt.title('Salary with Rolling Average')
plt.xlabel('Employee Index')
plt.ylabel('Salary')
plt.legend()
plt.show()
```

### 2. Bar Plots

```python
# Department counts
dept_counts = df['department'].value_counts()
dept_counts.plot(kind='bar', figsize=(10, 6))
plt.title('Employee Count by Department')
plt.xlabel('Department')
plt.ylabel('Number of Employees')
plt.xticks(rotation=45)
plt.show()

# Average salary by department
dept_salary = df.groupby('department')['salary'].mean()
dept_salary.plot(kind='bar', figsize=(10, 6), color='skyblue')
plt.title('Average Salary by Department')
plt.xlabel('Department')
plt.ylabel('Average Salary')
plt.xticks(rotation=45)
plt.show()
```

### 3. Histograms

```python
# Salary distribution
df['salary'].plot(kind='hist', bins=30, figsize=(10, 6), alpha=0.7)
plt.title('Salary Distribution')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.show()

# Multiple histograms
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

df['age'].plot(kind='hist', bins=20, ax=axes[0,0], title='Age Distribution')
df['experience_years'].plot(kind='hist', bins=20, ax=axes[0,1], title='Experience Distribution')
df['performance_rating'].plot(kind='hist', bins=20, ax=axes[1,0], title='Performance Rating Distribution')
df['salary'].plot(kind='hist', bins=30, ax=axes[1,1], title='Salary Distribution')

plt.tight_layout()
plt.show()
```

## Statistical Plots

### 1. Box Plots

```python
# Salary by department
df.boxplot(column='salary', by='department', figsize=(10, 6))
plt.title('Salary Distribution by Department')
plt.suptitle('')  # Remove default title
plt.show()

# Multiple variables
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

df.boxplot(column='salary', by='department', ax=axes[0,0])
axes[0,0].set_title('Salary by Department')

df.boxplot(column='age', by='position', ax=axes[0,1])
axes[0,1].set_title('Age by Position')

df.boxplot(column='performance_rating', by='department', ax=axes[1,0])
axes[1,0].set_title('Performance by Department')

df.boxplot(column='experience_years', by='position', ax=axes[1,1])
axes[1,1].set_title('Experience by Position')

plt.tight_layout()
plt.show()
```

### 2. Violin Plots

```python
# Using seaborn for violin plots
plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='department', y='salary')
plt.title('Salary Distribution by Department (Violin Plot)')
plt.xticks(rotation=45)
plt.show()

# Multiple violin plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

sns.violinplot(data=df, x='department', y='salary', ax=axes[0,0])
axes[0,0].set_title('Salary by Department')

sns.violinplot(data=df, x='position', y='age', ax=axes[0,1])
axes[0,1].set_title('Age by Position')

sns.violinplot(data=df, x='department', y='performance_rating', ax=axes[1,0])
axes[1,0].set_title('Performance by Department')

sns.violinplot(data=df, x='position', y='experience_years', ax=axes[1,1])
axes[1,1].set_title('Experience by Position')

plt.tight_layout()
plt.show()
```

## Distribution Plots

### 1. Density Plots

```python
# Kernel density estimation
df['salary'].plot(kind='density', figsize=(10, 6))
plt.title('Salary Density Distribution')
plt.xlabel('Salary')
plt.ylabel('Density')
plt.show()

# Multiple density plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

df['age'].plot(kind='density', ax=axes[0,0], title='Age Density')
df['experience_years'].plot(kind='density', ax=axes[0,1], title='Experience Density')
df['performance_rating'].plot(kind='density', ax=axes[1,0], title='Performance Density')
df['salary'].plot(kind='density', ax=axes[1,1], title='Salary Density')

plt.tight_layout()
plt.show()
```

### 2. Q-Q Plots

```python
from scipy import stats

# Q-Q plot for salary
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

stats.probplot(df['salary'], dist="norm", plot=axes[0,0])
axes[0,0].set_title('Q-Q Plot: Salary')

stats.probplot(df['age'], dist="norm", plot=axes[0,1])
axes[0,1].set_title('Q-Q Plot: Age')

stats.probplot(df['experience_years'], dist="norm", plot=axes[1,0])
axes[1,0].set_title('Q-Q Plot: Experience')

stats.probplot(df['performance_rating'], dist="norm", plot=axes[1,1])
axes[1,1].set_title('Q-Q Plot: Performance Rating')

plt.tight_layout()
plt.show()
```

## Relationship Plots

### 1. Scatter Plots

```python
# Basic scatter plot
df.plot.scatter(x='age', y='salary', figsize=(10, 6), alpha=0.6)
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

# Scatter plot with color coding
plt.figure(figsize=(10, 6))
for dept in df['department'].unique():
    dept_data = df[df['department'] == dept]
    plt.scatter(dept_data['age'], dept_data['salary'], 
               label=dept, alpha=0.6, s=50)

plt.title('Age vs Salary by Department')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Scatter matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df[['age', 'salary', 'experience_years', 'performance_rating']], 
              figsize=(12, 12), alpha=0.6)
plt.suptitle('Scatter Matrix of Numeric Variables')
plt.show()
```

### 2. Correlation Heatmaps

```python
# Correlation matrix
numeric_cols = ['age', 'salary', 'experience_years', 'performance_rating']
correlation_matrix = df[numeric_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()

# Using pandas plotting
correlation_matrix.plot(kind='heatmap', figsize=(8, 6), cmap='coolwarm')
plt.title('Correlation Heatmap (Pandas)')
plt.show()
```

## Categorical Plots

### 1. Count Plots

```python
# Department counts
plt.figure(figsize=(10, 6))
df['department'].value_counts().plot(kind='bar')
plt.title('Employee Count by Department')
plt.xlabel('Department')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Position counts by department
dept_pos_counts = df.groupby(['department', 'position']).size().unstack()
dept_pos_counts.plot(kind='bar', figsize=(12, 6))
plt.title('Position Count by Department')
plt.xlabel('Department')
plt.ylabel('Count')
plt.legend(title='Position')
plt.xticks(rotation=45)
plt.show()
```

### 2. Stacked Bar Plots

```python
# Gender distribution by department
gender_dept = df.groupby(['department', 'gender']).size().unstack()
gender_dept.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Gender Distribution by Department')
plt.xlabel('Department')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.xticks(rotation=45)
plt.show()

# Position distribution by department
pos_dept = df.groupby(['department', 'position']).size().unstack()
pos_dept.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Position Distribution by Department')
plt.xlabel('Department')
plt.ylabel('Count')
plt.legend(title='Position')
plt.xticks(rotation=45)
plt.show()
```

## Time Series Plots

### 1. Time Series Analysis

```python
# Set hire_date as index
df_time = df.set_index('hire_date').sort_index()

# Monthly hiring trends
monthly_hires = df_time.resample('M').size()
monthly_hires.plot(kind='line', figsize=(12, 6))
plt.title('Monthly Hiring Trends')
plt.xlabel('Date')
plt.ylabel('Number of Hires')
plt.show()

# Salary trends over time
monthly_salary = df_time.resample('M')['salary'].mean()
monthly_salary.plot(kind='line', figsize=(12, 6))
plt.title('Average Salary Over Time')
plt.xlabel('Date')
plt.ylabel('Average Salary')
plt.show()
```

### 2. Seasonal Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Resample to monthly data
monthly_data = df_time.resample('M')['salary'].mean()

# Seasonal decomposition
decomposition = seasonal_decompose(monthly_data, period=12)

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=axes[0], title='Observed')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()
```

## Advanced Customization

### 1. Custom Styling

```python
# Custom color palette
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

# Department salary comparison with custom colors
dept_salary = df.groupby('department')['salary'].mean()
ax = dept_salary.plot(kind='bar', figsize=(10, 6), color=colors)
plt.title('Average Salary by Department', fontsize=16, fontweight='bold')
plt.xlabel('Department', fontsize=12)
plt.ylabel('Average Salary ($)', fontsize=12)
plt.xticks(rotation=45)

# Add value labels on bars
for i, v in enumerate(dept_salary):
    ax.text(i, v + 1000, f'${v:,.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()
```

### 2. Subplots and Layout

```python
# Create comprehensive dashboard
fig = plt.figure(figsize=(20, 12))

# Salary distribution
ax1 = plt.subplot(2, 3, 1)
df['salary'].hist(bins=30, ax=ax1, alpha=0.7, color='skyblue')
ax1.set_title('Salary Distribution')
ax1.set_xlabel('Salary')
ax1.set_ylabel('Frequency')

# Age vs Salary scatter
ax2 = plt.subplot(2, 3, 2)
ax2.scatter(df['age'], df['salary'], alpha=0.6, color='coral')
ax2.set_title('Age vs Salary')
ax2.set_xlabel('Age')
ax2.set_ylabel('Salary')

# Department counts
ax3 = plt.subplot(2, 3, 3)
df['department'].value_counts().plot(kind='bar', ax=ax3, color='lightgreen')
ax3.set_title('Employee Count by Department')
ax3.set_xlabel('Department')
ax3.set_ylabel('Count')
ax3.tick_params(axis='x', rotation=45)

# Performance rating distribution
ax4 = plt.subplot(2, 3, 4)
df['performance_rating'].hist(bins=20, ax=ax4, alpha=0.7, color='gold')
ax4.set_title('Performance Rating Distribution')
ax4.set_xlabel('Performance Rating')
ax4.set_ylabel('Frequency')

# Experience vs Salary
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(df['experience_years'], df['salary'], alpha=0.6, color='purple')
ax5.set_title('Experience vs Salary')
ax5.set_xlabel('Experience (Years)')
ax5.set_ylabel('Salary')

# Salary by position
ax6 = plt.subplot(2, 3, 6)
df.boxplot(column='salary', by='position', ax=ax6)
ax6.set_title('Salary by Position')
ax6.set_xlabel('Position')
ax6.set_ylabel('Salary')

plt.suptitle('Employee Data Analysis Dashboard', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Best Practices

### 1. Color and Style Guidelines

```python
# Consistent color scheme
def create_consistent_plot(df, title, xlabel, ylabel):
    """Create a plot with consistent styling."""
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    dept_salary = df.groupby('department')['salary'].mean()
    bars = ax.bar(dept_salary.index, dept_salary.values, color=colors[:len(dept_salary)])
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1000,
                f'${height:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# Usage
fig = create_consistent_plot(df, 'Average Salary by Department', 'Department', 'Average Salary ($)')
plt.show()
```

### 2. Interactive Plots

```python
# Using plotly for interactive plots (if available)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Interactive scatter plot
    fig = px.scatter(df, x='age', y='salary', color='department', 
                     title='Age vs Salary by Department',
                     hover_data=['name', 'position'])
    fig.show()
    
    # Interactive box plot
    fig = px.box(df, x='department', y='salary', 
                 title='Salary Distribution by Department')
    fig.show()
    
except ImportError:
    print("Plotly not available. Install with: pip install plotly")
```

### 3. Saving Plots

```python
def save_plot_with_metadata(fig, filename, title, description=""):
    """Save plot with metadata."""
    # Add metadata
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Save with high DPI
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    
    # Save metadata
    metadata = {
        'title': title,
        'description': description,
        'created_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_shape': df.shape
    }
    
    # Save metadata to text file
    with open(filename.replace('.png', '_metadata.txt'), 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Plot saved as {filename}")

# Example usage
fig, ax = plt.subplots(figsize=(10, 6))
df['salary'].hist(bins=30, ax=ax, alpha=0.7)
ax.set_title('Salary Distribution')
ax.set_xlabel('Salary')
ax.set_ylabel('Frequency')

save_plot_with_metadata(fig, 'salary_distribution.png', 
                       'Employee Salary Distribution',
                       'Histogram showing the distribution of employee salaries')
```

## Common Pitfalls

### 1. Overplotting

```python
# Problem: Too many points in scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['age'], df['salary'], alpha=0.1)  # Very transparent
plt.title('Age vs Salary (Overplotted)')
plt.show()

# Solution: Use hexbin or 2D histogram
plt.figure(figsize=(10, 6))
plt.hexbin(df['age'], df['salary'], gridsize=20, cmap='Blues')
plt.colorbar(label='Count')
plt.title('Age vs Salary (Hexbin)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()
```

### 2. Misleading Scales

```python
# Problem: Different scales can be misleading
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Different scales
ax1.plot(df.groupby('department')['salary'].mean())
ax1.set_title('Salary by Department (Different Scales)')

ax2.plot(df.groupby('department')['age'].mean())
ax2.set_title('Age by Department (Different Scales)')

plt.tight_layout()
plt.show()

# Solution: Use subplots with shared y-axis or normalize data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Normalize data
salary_norm = df.groupby('department')['salary'].mean() / df.groupby('department')['salary'].mean().max()
age_norm = df.groupby('department')['age'].mean() / df.groupby('department')['age'].mean().max()

ax1.plot(salary_norm)
ax1.set_title('Normalized Salary by Department')

ax2.plot(age_norm)
ax2.set_title('Normalized Age by Department')

plt.tight_layout()
plt.show()
```

### 3. Poor Color Choices

```python
# Problem: Poor color contrast
plt.figure(figsize=(10, 6))
colors_bad = ['yellow', 'lightyellow', 'white', 'lightblue', 'blue']
df['department'].value_counts().plot(kind='bar', color=colors_bad)
plt.title('Poor Color Choice')
plt.show()

# Solution: Use colorblind-friendly palette
plt.figure(figsize=(10, 6))
colors_good = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
df['department'].value_counts().plot(kind='bar', color=colors_good)
plt.title('Good Color Choice')
plt.show()
```

## Summary

This guide covered comprehensive data visualization with pandas:

1. **Basic Plotting**: Line, bar, and histogram plots
2. **Statistical Plots**: Box plots and violin plots
3. **Distribution Plots**: Density plots and Q-Q plots
4. **Relationship Plots**: Scatter plots and correlation heatmaps
5. **Categorical Plots**: Count plots and stacked bar plots
6. **Time Series Plots**: Trend analysis and seasonal decomposition
7. **Advanced Customization**: Styling and layout
8. **Best Practices**: Color schemes and interactive plots
9. **Common Pitfalls**: Overplotting and misleading scales

### Key Takeaways

- **Choose appropriate plot types** for your data and analysis goals
- **Use consistent styling** for professional-looking visualizations
- **Consider your audience** when selecting colors and complexity
- **Validate your visualizations** to ensure they accurately represent the data
- **Optimize for clarity** over complexity

### Next Steps

- Practice with real-world datasets
- Explore interactive visualization libraries
- Learn about dashboard creation
- Study advanced statistical plotting
- Master publication-quality graphics

### Additional Resources

- [Matplotlib Documentation](https://matplotlib.org/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Plotly Documentation](https://plotly.com/python/)
- [Data Visualization Best Practices](https://www.storytellingwithdata.com/)

---

**Ready to create stunning visualizations? Start exploring your data with these plotting techniques!** 