# Plotly Statistical Visualization Guide

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Plotly](https://img.shields.io/badge/Plotly-5.0+-purple.svg)](https://plotly.com/python/)
[![Statistics](https://img.shields.io/badge/Statistical-Analysis-orange.svg)](https://plotly.com/python/statistical-charts/)

A comprehensive guide to creating statistical visualizations with Plotly, including box plots, violin plots, histograms, density plots, correlation matrices, and advanced statistical analysis techniques.

## Table of Contents

1. [Introduction to Statistical Visualization](#introduction-to-statistical-visualization)
2. [Distribution Analysis](#distribution-analysis)
3. [Box Plots and Violin Plots](#box-plots-and-violin-plots)
4. [Correlation Analysis](#correlation-analysis)
5. [Statistical Tests Visualization](#statistical-tests-visualization)
6. [Regression Analysis](#regression-analysis)
7. [Time Series Statistics](#time-series-statistics)
8. [Advanced Statistical Plots](#advanced-statistical-plots)
9. [Performance Optimization](#performance-optimization)
10. [Best Practices](#best-practices)

## Introduction to Statistical Visualization

Statistical visualizations help you understand data distributions, relationships, and patterns through graphical representations of statistical concepts.

### Why Statistical Visualization?

- **Data Distribution** - Understand how data is spread and shaped
- **Relationship Analysis** - Identify correlations and associations
- **Outlier Detection** - Find unusual data points
- **Statistical Inference** - Visualize hypothesis tests and confidence intervals

### Basic Setup

```python
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns

# Create sample data
np.random.seed(42)
data = {
    'group_a': np.random.normal(100, 15, 100),
    'group_b': np.random.normal(110, 20, 100),
    'group_c': np.random.normal(95, 12, 100)
}
df = pd.DataFrame(data)
```

## Distribution Analysis

### Histograms

```python
# Basic histogram
fig = px.histogram(
    df['group_a'],
    title="Distribution of Group A",
    labels={'value': 'Value', 'count': 'Frequency'},
    nbins=20
)

fig.show()
```

### Multiple Histograms

```python
# Multiple histograms
fig = go.Figure()

for group in ['group_a', 'group_b', 'group_c']:
    fig.add_trace(go.Histogram(
        x=df[group],
        name=group.replace('_', ' ').title(),
        opacity=0.7,
        nbinsx=20
    ))

fig.update_layout(
    title="Distribution Comparison",
    xaxis_title="Value",
    yaxis_title="Frequency",
    barmode='overlay'
)

fig.show()
```

### Density Plots

```python
# Create density plot
fig = px.histogram(
    df['group_a'],
    title="Density Plot of Group A",
    labels={'value': 'Value', 'density': 'Density'},
    nbins=20,
    histnorm='density'  # Normalize to density
)

fig.show()
```

### Kernel Density Estimation

```python
# Kernel density estimation
from scipy.stats import gaussian_kde

# Calculate KDE
kde = gaussian_kde(df['group_a'])
x_range = np.linspace(df['group_a'].min(), df['group_a'].max(), 100)
density = kde(x_range)

fig = go.Figure()

# Add histogram
fig.add_trace(go.Histogram(
    x=df['group_a'],
    name='Histogram',
    opacity=0.7,
    nbinsx=20,
    histnorm='density'
))

# Add KDE line
fig.add_trace(go.Scatter(
    x=x_range,
    y=density,
    mode='lines',
    name='KDE',
    line=dict(color='red', width=2)
))

fig.update_layout(
    title="Histogram with Kernel Density Estimation",
    xaxis_title="Value",
    yaxis_title="Density",
    barmode='overlay'
)

fig.show()
```

## Box Plots and Violin Plots

### Basic Box Plot

```python
# Create box plot
fig = px.box(
    df,
    title="Box Plot Comparison",
    labels={'variable': 'Group', 'value': 'Value'}
)

fig.show()
```

### Custom Box Plot

```python
# Custom box plot with statistics
fig = go.Figure()

for group in ['group_a', 'group_b', 'group_c']:
    fig.add_trace(go.Box(
        y=df[group],
        name=group.replace('_', ' ').title(),
        boxpoints='outliers',  # Show outliers
        jitter=0.3,  # Add jitter to points
        pointpos=-1.8  # Position points
    ))

fig.update_layout(
    title="Custom Box Plot with Outliers",
    yaxis_title="Value",
    showlegend=True
)

fig.show()
```

### Violin Plots

```python
# Create violin plot
fig = px.violin(
    df,
    title="Violin Plot Comparison",
    labels={'variable': 'Group', 'value': 'Value'}
)

fig.show()
```

### Custom Violin Plot

```python
# Custom violin plot
fig = go.Figure()

for group in ['group_a', 'group_b', 'group_c']:
    fig.add_trace(go.Violin(
        y=df[group],
        name=group.replace('_', ' ').title(),
        box_visible=True,  # Show box inside violin
        meanline_visible=True,  # Show mean line
        line_color='black'
    ))

fig.update_layout(
    title="Custom Violin Plot with Box and Mean",
    yaxis_title="Value",
    showlegend=True
)

fig.show()
```

### Combined Box and Violin

```python
# Create subplot with both box and violin plots
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=('Box Plot', 'Violin Plot')
)

# Box plot
for group in ['group_a', 'group_b', 'group_c']:
    fig.add_trace(go.Box(
        y=df[group],
        name=group.replace('_', ' ').title(),
        showlegend=False
    ), row=1, col=1)

# Violin plot
for group in ['group_a', 'group_b', 'group_c']:
    fig.add_trace(go.Violin(
        y=df[group],
        name=group.replace('_', ' ').title(),
        box_visible=True,
        meanline_visible=True,
        showlegend=False
    ), row=1, col=2)

fig.update_layout(
    title="Distribution Comparison: Box vs Violin Plots",
    height=500
)

fig.show()
```

## Correlation Analysis

### Correlation Matrix

```python
# Calculate correlation matrix
correlation_matrix = df.corr()

fig = px.imshow(
    correlation_matrix,
    title="Correlation Matrix",
    color_continuous_scale='RdBu',
    aspect='auto'
)

fig.show()
```

### Custom Correlation Matrix

```python
# Create custom correlation matrix
fig = go.Figure(data=go.Heatmap(
    z=correlation_matrix.values,
    x=correlation_matrix.columns,
    y=correlation_matrix.columns,
    colorscale='RdBu',
    zmid=0,
    text=np.round(correlation_matrix.values, 2),
    texttemplate="%{text}",
    textfont={"size": 12},
    hoverongaps=False
))

fig.update_layout(
    title="Correlation Matrix with Values",
    xaxis_title="Variables",
    yaxis_title="Variables"
)

fig.show()
```

### Scatter Plot Matrix

```python
# Create scatter plot matrix
fig = px.scatter_matrix(
    df,
    title="Scatter Plot Matrix",
    dimensions=['group_a', 'group_b', 'group_c']
)

fig.show()
```

### Correlation Scatter Plots

```python
# Create correlation scatter plots
fig = px.scatter(
    df,
    x='group_a',
    y='group_b',
    title="Correlation: Group A vs Group B",
    trendline="ols"  # Add regression line
)

fig.show()
```

## Statistical Tests Visualization

### T-Test Visualization

```python
# Perform t-test and visualize
from scipy.stats import ttest_ind

# Perform t-test
t_stat, p_value = ttest_ind(df['group_a'], df['group_b'])

# Create visualization
fig = go.Figure()

# Add box plots
fig.add_trace(go.Box(y=df['group_a'], name='Group A'))
fig.add_trace(go.Box(y=df['group_b'], name='Group B'))

# Add annotation with test results
fig.add_annotation(
    x=0.5,
    y=1.1,
    xref="paper",
    yref="paper",
    text=f"T-Test Results:<br>t-statistic: {t_stat:.3f}<br>p-value: {p_value:.3f}",
    showarrow=False,
    bgcolor="lightblue",
    bordercolor="black",
    borderwidth=1
)

fig.update_layout(
    title="T-Test: Group A vs Group B",
    yaxis_title="Value"
)

fig.show()
```

### ANOVA Visualization

```python
# Perform ANOVA and visualize
from scipy.stats import f_oneway

# Perform ANOVA
f_stat, p_value = f_oneway(df['group_a'], df['group_b'], df['group_c'])

# Create visualization
fig = go.Figure()

# Add box plots for all groups
for group in ['group_a', 'group_b', 'group_c']:
    fig.add_trace(go.Box(
        y=df[group],
        name=group.replace('_', ' ').title()
    ))

# Add annotation with ANOVA results
fig.add_annotation(
    x=0.5,
    y=1.1,
    xref="paper",
    yref="paper",
    text=f"ANOVA Results:<br>F-statistic: {f_stat:.3f}<br>p-value: {p_value:.3f}",
    showarrow=False,
    bgcolor="lightgreen",
    bordercolor="black",
    borderwidth=1
)

fig.update_layout(
    title="ANOVA: All Groups Comparison",
    yaxis_title="Value"
)

fig.show()
```

### Confidence Intervals

```python
# Calculate and visualize confidence intervals
def calculate_ci(data, confidence=0.95):
    """Calculate confidence interval"""
    mean = np.mean(data)
    std_err = stats.sem(data)
    ci = stats.t.interval(confidence, len(data)-1, loc=mean, scale=std_err)
    return mean, ci

# Calculate CIs for each group
ci_data = {}
for group in ['group_a', 'group_b', 'group_c']:
    mean, ci = calculate_ci(df[group])
    ci_data[group] = {'mean': mean, 'ci_lower': ci[0], 'ci_upper': ci[1]}

# Create visualization
fig = go.Figure()

for i, group in enumerate(['group_a', 'group_b', 'group_c']):
    data = ci_data[group]
    fig.add_trace(go.Scatter(
        x=[i, i],
        y=[data['ci_lower'], data['ci_upper']],
        mode='lines',
        name=f"{group.replace('_', ' ').title()} CI",
        line=dict(width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[i],
        y=[data['mean']],
        mode='markers',
        name=f"{group.replace('_', ' ').title()} Mean",
        marker=dict(size=10, symbol='diamond'),
        showlegend=False
    ))

fig.update_layout(
    title="95% Confidence Intervals",
    xaxis_title="Groups",
    yaxis_title="Value",
    xaxis=dict(tickvals=[0, 1, 2], ticktext=['Group A', 'Group B', 'Group C'])
)

fig.show()
```

## Regression Analysis

### Linear Regression

```python
# Create linear regression visualization
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Prepare data
X = df['group_a'].values.reshape(-1, 1)
y = df['group_b'].values

# Fit regression
reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)
r2 = r2_score(y, y_pred)

# Create visualization
fig = go.Figure()

# Add scatter plot
fig.add_trace(go.Scatter(
    x=df['group_a'],
    y=df['group_b'],
    mode='markers',
    name='Data Points',
    marker=dict(color='blue', opacity=0.6)
))

# Add regression line
fig.add_trace(go.Scatter(
    x=df['group_a'],
    y=y_pred,
    mode='lines',
    name=f'Regression Line (R² = {r2:.3f})',
    line=dict(color='red', width=2)
))

fig.update_layout(
    title="Linear Regression: Group A vs Group B",
    xaxis_title="Group A",
    yaxis_title="Group B"
)

fig.show()
```

### Residual Plot

```python
# Create residual plot
residuals = y - y_pred

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=y_pred,
    y=residuals,
    mode='markers',
    name='Residuals',
    marker=dict(color='blue', opacity=0.6)
))

# Add horizontal line at y=0
fig.add_hline(y=0, line_dash="dash", line_color="red")

fig.update_layout(
    title="Residual Plot",
    xaxis_title="Predicted Values",
    yaxis_title="Residuals"
)

fig.show()
```

### Multiple Regression

```python
# Create multiple regression visualization
# Use group_a and group_b to predict group_c
X_multi = df[['group_a', 'group_b']].values
y_multi = df['group_c'].values

# Fit multiple regression
reg_multi = LinearRegression()
reg_multi.fit(X_multi, y_multi)
y_pred_multi = reg_multi.predict(X_multi)
r2_multi = r2_score(y_multi, y_pred_multi)

# Create 3D scatter plot
fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=df['group_a'],
    y=df['group_b'],
    z=df['group_c'],
    mode='markers',
    name='Data Points',
    marker=dict(
        size=5,
        color=df['group_c'],
        colorscale='Viridis',
        opacity=0.8
    )
))

fig.update_layout(
    title=f"Multiple Regression (R² = {r2_multi:.3f})",
    scene=dict(
        xaxis_title="Group A",
        yaxis_title="Group B",
        zaxis_title="Group C"
    )
)

fig.show()
```

## Time Series Statistics

### Moving Averages

```python
# Create time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
time_series = np.cumsum(np.random.randn(100)) + 100

# Calculate moving averages
df_ts = pd.DataFrame({
    'date': dates,
    'value': time_series,
    'ma_7': pd.Series(time_series).rolling(window=7).mean(),
    'ma_30': pd.Series(time_series).rolling(window=30).mean()
})

# Create visualization
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_ts['date'],
    y=df_ts['value'],
    mode='lines',
    name='Original Data',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=df_ts['date'],
    y=df_ts['ma_7'],
    mode='lines',
    name='7-day Moving Average',
    line=dict(color='red')
))

fig.add_trace(go.Scatter(
    x=df_ts['date'],
    y=df_ts['ma_30'],
    mode='lines',
    name='30-day Moving Average',
    line=dict(color='green')
))

fig.update_layout(
    title="Time Series with Moving Averages",
    xaxis_title="Date",
    yaxis_title="Value"
)

fig.show()
```

### Seasonal Decomposition

```python
# Create seasonal data
t = np.arange(100)
seasonal = 10 * np.sin(2 * np.pi * t / 12)  # 12-period seasonality
trend = 0.1 * t
noise = np.random.normal(0, 1, 100)
seasonal_data = trend + seasonal + noise

# Create visualization
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t,
    y=seasonal_data,
    mode='lines',
    name='Original Data',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=t,
    y=trend,
    mode='lines',
    name='Trend',
    line=dict(color='red', dash='dash')
))

fig.add_trace(go.Scatter(
    x=t,
    y=seasonal,
    mode='lines',
    name='Seasonal',
    line=dict(color='green', dash='dot')
))

fig.update_layout(
    title="Time Series Decomposition",
    xaxis_title="Time",
    yaxis_title="Value"
)

fig.show()
```

## Advanced Statistical Plots

### Q-Q Plot

```python
# Create Q-Q plot
from scipy.stats import probplot

# Calculate theoretical quantiles
theoretical_quantiles, sample_quantiles = probplot(df['group_a'], dist="norm")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=theoretical_quantiles,
    y=sample_quantiles,
    mode='markers',
    name='Q-Q Plot',
    marker=dict(color='blue', opacity=0.6)
))

# Add diagonal line
min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
fig.add_trace(go.Scatter(
    x=[min_val, max_val],
    y=[min_val, max_val],
    mode='lines',
    name='Normal Line',
    line=dict(color='red', dash='dash')
))

fig.update_layout(
    title="Q-Q Plot: Group A vs Normal Distribution",
    xaxis_title="Theoretical Quantiles",
    yaxis_title="Sample Quantiles"
)

fig.show()
```

### P-P Plot

```python
# Create P-P plot
from scipy.stats import norm

# Calculate empirical and theoretical CDFs
sorted_data = np.sort(df['group_a'])
empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
theoretical_cdf = norm.cdf(sorted_data, np.mean(sorted_data), np.std(sorted_data))

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=theoretical_cdf,
    y=empirical_cdf,
    mode='markers',
    name='P-P Plot',
    marker=dict(color='blue', opacity=0.6)
))

# Add diagonal line
fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    name='Perfect Fit',
    line=dict(color='red', dash='dash')
))

fig.update_layout(
    title="P-P Plot: Group A vs Normal Distribution",
    xaxis_title="Theoretical CDF",
    yaxis_title="Empirical CDF"
)

fig.show()
```

### Statistical Power Analysis

```python
# Create power analysis visualization
from scipy.stats import norm

# Calculate power for different effect sizes
effect_sizes = np.linspace(0.1, 1.0, 20)
sample_sizes = [20, 50, 100]
alpha = 0.05

power_data = {}
for n in sample_sizes:
    power_values = []
    for effect in effect_sizes:
        # Calculate power
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = effect * np.sqrt(n/2) - z_alpha
        power = 1 - norm.cdf(z_beta)
        power_values.append(power)
    power_data[n] = power_values

# Create visualization
fig = go.Figure()

for n in sample_sizes:
    fig.add_trace(go.Scatter(
        x=effect_sizes,
        y=power_data[n],
        mode='lines+markers',
        name=f'n = {n}',
        line=dict(width=2)
    ))

fig.update_layout(
    title="Statistical Power Analysis",
    xaxis_title="Effect Size",
    yaxis_title="Power",
    yaxis=dict(range=[0, 1])
)

fig.show()
```

## Performance Optimization

### Large Dataset Handling

```python
# Optimize statistical plots for large datasets
def create_optimized_histogram(data, max_bins=50):
    """Create optimized histogram for large datasets"""
    if len(data) > 10000:
        # Downsample for visualization
        data_sample = np.random.choice(data, 10000, replace=False)
    else:
        data_sample = data
    
    # Calculate optimal number of bins
    n_bins = min(max_bins, int(np.sqrt(len(data_sample))))
    
    fig = px.histogram(
        data_sample,
        nbins=n_bins,
        title="Optimized Histogram for Large Dataset"
    )
    
    return fig

# Example usage
large_data = np.random.normal(100, 15, 100000)
fig = create_optimized_histogram(large_data)
fig.show()
```

### Efficient Statistical Calculations

```python
# Efficient statistical calculations
def efficient_statistics(data):
    """Calculate multiple statistics efficiently"""
    stats_dict = {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75)
    }
    
    return stats_dict

# Example usage
stats_results = efficient_statistics(df['group_a'])
print("Statistical Summary:", stats_results)
```

## Best Practices

### 1. Appropriate Plot Selection

```python
# Choose appropriate plots for different data types
def select_appropriate_plot(data_type, data):
    """Select appropriate statistical plot based on data type"""
    if data_type == 'continuous':
        # Use histogram or density plot
        fig = px.histogram(data, title="Continuous Data Distribution")
    elif data_type == 'categorical':
        # Use bar plot or pie chart
        fig = px.bar(data, title="Categorical Data")
    elif data_type == 'correlation':
        # Use scatter plot or correlation matrix
        fig = px.scatter(data, title="Correlation Analysis")
    
    return fig
```

### 2. Color and Contrast

```python
# Use appropriate colors for statistical plots
fig = px.box(
    df,
    title="Statistical Plot with Good Color Contrast",
    color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c']
)

fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white'
)

fig.show()
```

### 3. Statistical Significance

```python
# Include statistical significance in visualizations
# Example: Add p-values to correlation plot
correlation, p_value = stats.pearsonr(df['group_a'], df['group_b'])

fig = px.scatter(
    df,
    x='group_a',
    y='group_b',
    title=f"Correlation: r={correlation:.3f}, p={p_value:.3f}"
)

fig.show()
```

### 4. Accessibility

```python
# Make statistical plots accessible
fig = px.histogram(
    df['group_a'],
    title="Accessible Statistical Plot",
    nbins=20
)

fig.update_layout(
    font=dict(size=14),  # Larger fonts
    xaxis_title="Value",
    yaxis_title="Frequency"
)

fig.show()
```

## Summary

Plotly statistical visualization provides powerful tools for data analysis:

- **Distribution Analysis**: Histograms, density plots, and KDE
- **Comparison Plots**: Box plots, violin plots, and statistical tests
- **Correlation Analysis**: Correlation matrices and scatter plots
- **Regression Analysis**: Linear and multiple regression visualization
- **Time Series Statistics**: Moving averages and seasonal decomposition
- **Advanced Plots**: Q-Q plots, P-P plots, and power analysis
- **Performance**: Optimization for large datasets
- **Best Practices**: Appropriate plot selection and statistical significance

Master these statistical visualization techniques to create compelling, informative plots that effectively communicate statistical insights from your data.

## Next Steps

- Explore [Plotly Statistical Charts](https://plotly.com/python/statistical-charts/) for more examples
- Learn [Graph Objects](https://plotly.com/python/graph-objects/) for advanced customization
- Study [Statistical Analysis](https://plotly.com/python/scientific-charts/) techniques
- Practice [Interactive Features](https://plotly.com/python/interactive-plots/) for statistical plots

---

**Happy Statistical Plotting!** 📊✨ 