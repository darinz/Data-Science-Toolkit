# Plotly Subplots and Layouts Guide

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Plotly](https://img.shields.io/badge/Plotly-5.0+-purple.svg)](https://plotly.com/python/)
[![Subplots](https://img.shields.io/badge/Subplots-Layouts-green.svg)](https://plotly.com/python/subplots/)

A comprehensive guide to creating multi-panel visualizations with Plotly, including subplots, complex layouts, shared axes, and advanced figure composition techniques.

## Table of Contents

1. [Introduction to Subplots](#introduction-to-subplots)
2. [Basic Subplots](#basic-subplots)
3. [Complex Layouts](#complex-layouts)
4. [Shared Axes and Ranges](#shared-axes-and-ranges)
5. [Figure Composition](#figure-composition)
6. [Multi-Panel Dashboards](#multi-panel-dashboards)
7. [Advanced Layout Techniques](#advanced-layout-techniques)
8. [Performance Optimization](#performance-optimization)
9. [Best Practices](#best-practices)
10. [Common Patterns](#common-patterns)

## Introduction to Subplots

Subplots allow you to create multi-panel visualizations that can display different aspects of your data simultaneously, making complex relationships easier to understand.

### Why Use Subplots?

- **Data Comparison** - Compare different datasets side by side
- **Multi-dimensional Analysis** - Show different views of the same data
- **Space Efficiency** - Maximize information density
- **Storytelling** - Guide viewers through a narrative

### Basic Concepts

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import pandas as pd

# Create sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
```

## Basic Subplots

### Simple 2x2 Grid

```python
# Create a 2x2 subplot grid
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Sine Wave', 'Cosine Wave', 'Tangent Wave', 'Combined'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# Add traces to subplots
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Sine'), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Cosine'), row=1, col=2)
fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', name='Tangent'), row=2, col=1)

# Combined plot
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Sine', line=dict(color='blue')), row=2, col=2)
fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Cosine', line=dict(color='red')), row=2, col=2)

fig.update_layout(height=600, title_text="Trigonometric Functions")
fig.show()
```

### Different Plot Types

```python
# Create subplots with different plot types
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Line Plot', 'Scatter Plot', 'Bar Chart', 'Histogram'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# Line plot
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Line'), row=1, col=1)

# Scatter plot
fig.add_trace(go.Scatter(x=x[::5], y=y2[::5], mode='markers', name='Scatter'), row=1, col=2)

# Bar chart
fig.add_trace(go.Bar(x=x[::10], y=y1[::10], name='Bar'), row=2, col=1)

# Histogram
fig.add_trace(go.Histogram(x=np.random.randn(1000), name='Histogram'), row=2, col=2)

fig.update_layout(height=600, title_text="Different Plot Types")
fig.show()
```

### Customizing Subplot Layout

```python
# Create subplots with custom spacing and sizing
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Plot 1', 'Plot 2', 'Plot 3', 'Plot 4'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]],
    horizontal_spacing=0.1,  # Space between columns
    vertical_spacing=0.1,    # Space between rows
    column_widths=[0.5, 0.5],  # Column widths (fractions)
    row_heights=[0.5, 0.5]     # Row heights (fractions)
)

# Add traces
for i in range(1, 3):
    for j in range(1, 3):
        fig.add_trace(
            go.Scatter(x=x, y=np.sin(x + i*j), mode='lines', name=f'Trace {i}{j}'),
            row=i, col=j
        )

fig.update_layout(
    height=600,
    title_text="Custom Subplot Layout",
    showlegend=False
)

fig.show()
```

## Complex Layouts

### Uneven Grid Layouts

```python
# Create uneven grid (2 rows, 3 columns with different heights)
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=('Plot 1', 'Plot 2', 'Plot 3', 'Plot 4', 'Plot 5', 'Plot 6'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]],
    row_heights=[0.7, 0.3]  # First row takes 70%, second row takes 30%
)

# Add traces
for i in range(1, 3):
    for j in range(1, 4):
        fig.add_trace(
            go.Scatter(x=x, y=np.sin(x + i*j), mode='lines', name=f'Trace {i}{j}'),
            row=i, col=j
        )

fig.update_layout(height=600, title_text="Uneven Grid Layout")
fig.show()
```

### Mixed Plot Types with Secondary Axes

```python
# Create subplots with secondary axes
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Primary Only', 'With Secondary Y', 'With Secondary X', 'Both Secondary'),
    specs=[[{"secondary_y": False}, {"secondary_y": True}],
           [{"secondary_y": False, "secondary_x": True}, {"secondary_y": True, "secondary_x": True}]]
)

# Primary only
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Primary'), row=1, col=1)

# With secondary Y
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Primary Y'), row=1, col=2)
fig.add_trace(go.Scatter(x=x, y=y2*10, mode='lines', name='Secondary Y', line=dict(color='red')), 
              row=1, col=2, secondary_y=True)

# With secondary X
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Primary X'), row=2, col=1)
fig.add_trace(go.Scatter(x=x*2, y=y2, mode='lines', name='Secondary X', line=dict(color='green')), 
              row=2, col=1, secondary_x=True)

# Both secondary
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Primary'), row=2, col=2)
fig.add_trace(go.Scatter(x=x*2, y=y2*10, mode='lines', name='Secondary', line=dict(color='purple')), 
              row=2, col=2, secondary_y=True, secondary_x=True)

fig.update_layout(height=600, title_text="Secondary Axes")
fig.show()
```

### Insets and Overlays

```python
# Create main plot with inset
fig = go.Figure()

# Main plot
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Main Plot'))

# Add inset using shapes and annotations
fig.add_shape(
    type="rect",
    x0=2, y0=-0.5, x1=4, y1=0.5,
    fillcolor="lightblue",
    opacity=0.3,
    layer="below",
    line_width=0
)

fig.add_annotation(
    x=3, y=0.8,
    text="Inset Area",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="red",
    ax=40,
    ay=-40
)

# Create inset subplot
fig_inset = go.Figure()
fig_inset.add_trace(go.Scatter(x=x[20:40], y=y1[20:40], mode='lines', name='Inset'))

fig_inset.update_layout(
    width=200, height=150,
    margin=dict(l=0, r=0, t=0, b=0),
    showlegend=False,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)

fig.update_layout(title="Main Plot with Inset")
fig.show()
fig_inset.show()
```

## Shared Axes and Ranges

### Shared X-Axis

```python
# Create subplots with shared X-axis
fig = make_subplots(
    rows=3, cols=1,
    subplot_titles=('Plot 1', 'Plot 2', 'Plot 3'),
    shared_xaxes=True,  # Share X-axis
    vertical_spacing=0.05
)

# Add traces
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Sine'), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Cosine'), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', name='Tangent'), row=3, col=1)

fig.update_layout(height=600, title_text="Shared X-Axis")
fig.show()
```

### Shared Y-Axis

```python
# Create subplots with shared Y-axis
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=('Plot 1', 'Plot 2', 'Plot 3'),
    shared_yaxes=True,  # Share Y-axis
    horizontal_spacing=0.05
)

# Add traces
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Sine'), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Cosine'), row=1, col=2)
fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', name='Tangent'), row=1, col=3)

fig.update_layout(height=400, title_text="Shared Y-Axis")
fig.show()
```

### Synchronized Ranges

```python
# Create subplots with synchronized ranges
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Plot 1', 'Plot 2', 'Plot 3', 'Plot 4')
)

# Add traces with different ranges
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Sine'), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Cosine'), row=1, col=2)
fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', name='Tangent'), row=2, col=1)
fig.add_trace(go.Scatter(x=x, y=y1+y2, mode='lines', name='Sum'), row=2, col=2)

# Synchronize Y-axis ranges
fig.update_yaxes(range=[-2, 2], row=1, col=1)
fig.update_yaxes(range=[-2, 2], row=1, col=2)
fig.update_yaxes(range=[-2, 2], row=2, col=1)
fig.update_yaxes(range=[-2, 2], row=2, col=2)

fig.update_layout(height=600, title_text="Synchronized Ranges")
fig.show()
```

## Figure Composition

### Complex Multi-Panel Layout

```python
# Create a complex dashboard-like layout
fig = make_subplots(
    rows=3, cols=4,
    subplot_titles=('Time Series', 'Distribution', 'Correlation', 'Summary',
                   'Trends', 'Outliers', 'Seasonality', 'Forecast',
                   'Metrics', 'KPIs', 'Targets', 'Performance'),
    specs=[[{"colspan": 2}, None, {"colspan": 2}, None],
           [{"colspan": 2}, None, {"colspan": 2}, None],
           [{"colspan": 1}, {"colspan": 1}, {"colspan": 1}, {"colspan": 1}]],
    row_heights=[0.4, 0.4, 0.2]
)

# Time series (spans 2 columns)
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Time Series'), row=1, col=1)

# Distribution (spans 2 columns)
fig.add_trace(go.Histogram(x=np.random.randn(1000), name='Distribution'), row=1, col=3)

# Correlation (spans 2 columns)
fig.add_trace(go.Scatter(x=y1, y=y2, mode='markers', name='Correlation'), row=2, col=1)

# Summary (spans 2 columns)
fig.add_trace(go.Bar(x=['A', 'B', 'C'], y=[10, 20, 15], name='Summary'), row=2, col=3)

# Individual panels in bottom row
fig.add_trace(go.Indicator(mode="gauge+number", value=75, title={'text': "Metric 1"}), row=3, col=1)
fig.add_trace(go.Indicator(mode="gauge+number", value=85, title={'text': "Metric 2"}), row=3, col=2)
fig.add_trace(go.Indicator(mode="gauge+number", value=65, title={'text': "Metric 3"}), row=3, col=3)
fig.add_trace(go.Indicator(mode="gauge+number", value=90, title={'text': "Metric 4"}), row=3, col=4)

fig.update_layout(height=800, title_text="Complex Dashboard Layout")
fig.show()
```

### Annotated Subplots

```python
# Create subplots with annotations
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Annotated Plot 1', 'Annotated Plot 2', 'Annotated Plot 3', 'Annotated Plot 4')
)

# Add traces
for i in range(1, 3):
    for j in range(1, 3):
        fig.add_trace(
            go.Scatter(x=x, y=np.sin(x + i*j), mode='lines', name=f'Trace {i}{j}'),
            row=i, col=j
        )

# Add annotations to each subplot
annotations = [
    dict(x=0.25, y=0.75, xref="paper", yref="paper", text="Annotation 1", showarrow=True),
    dict(x=0.75, y=0.75, xref="paper", yref="paper", text="Annotation 2", showarrow=True),
    dict(x=0.25, y=0.25, xref="paper", yref="paper", text="Annotation 3", showarrow=True),
    dict(x=0.75, y=0.25, xref="paper", yref="paper", text="Annotation 4", showarrow=True)
]

fig.update_layout(
    height=600,
    title_text="Annotated Subplots",
    annotations=annotations
)

fig.show()
```

## Multi-Panel Dashboards

### Interactive Dashboard Layout

```python
# Create an interactive dashboard
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=('Main Chart', 'Distribution', 'Trends',
                   'Metrics', 'Comparison', 'Summary'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
)

# Main chart
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Main'), row=1, col=1)

# Distribution
fig.add_trace(go.Histogram(x=np.random.randn(1000), name='Dist'), row=1, col=2)

# Trends
fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Trend'), row=1, col=3)

# Metrics (using indicators)
fig.add_trace(go.Indicator(mode="number+delta", value=75, delta={'reference': 70}), row=2, col=1)

# Comparison
fig.add_trace(go.Bar(x=['A', 'B', 'C'], y=[10, 20, 15], name='Comp'), row=2, col=2)

# Summary
fig.add_trace(go.Pie(labels=['A', 'B', 'C'], values=[30, 40, 30], name='Summary'), row=2, col=3)

fig.update_layout(
    height=600,
    title_text="Interactive Dashboard",
    showlegend=True
)

fig.show()
```

### Responsive Dashboard

```python
# Create a responsive dashboard layout
fig = make_subplots(
    rows=3, cols=4,
    subplot_titles=('Chart 1', 'Chart 2', 'Chart 3', 'Chart 4',
                   'Chart 5', 'Chart 6', 'Chart 7', 'Chart 8',
                   'Metric 1', 'Metric 2', 'Metric 3', 'Metric 4'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
)

# Add various chart types
chart_types = [
    go.Scatter(x=x, y=y1, mode='lines'),
    go.Bar(x=['A', 'B', 'C'], y=[10, 20, 15]),
    go.Scatter(x=x, y=y2, mode='markers'),
    go.Histogram(x=np.random.randn(1000)),
    go.Scatter(x=x, y=y3, mode='lines+markers'),
    go.Bar(x=['X', 'Y', 'Z'], y=[25, 35, 30]),
    go.Scatter(x=y1, y=y2, mode='markers'),
    go.Histogram(x=np.random.exponential(1, 1000)),
    go.Indicator(mode="number", value=75),
    go.Indicator(mode="number", value=85),
    go.Indicator(mode="number", value=65),
    go.Indicator(mode="number", value=90)
]

for i, chart in enumerate(chart_types):
    row = (i // 4) + 1
    col = (i % 4) + 1
    fig.add_trace(chart, row=row, col=col)

fig.update_layout(
    height=800,
    title_text="Responsive Dashboard",
    showlegend=False
)

fig.show()
```

## Advanced Layout Techniques

### Dynamic Subplot Creation

```python
def create_dynamic_subplots(n_plots, cols=2):
    """Create subplots dynamically based on number of plots"""
    rows = (n_plots + cols - 1) // cols  # Ceiling division
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'Plot {i+1}' for i in range(n_plots)]
    )
    
    return fig, rows, cols

# Example usage
n_plots = 7
fig, rows, cols = create_dynamic_subplots(n_plots, cols=3)

# Add traces
for i in range(n_plots):
    row = (i // cols) + 1
    col = (i % cols) + 1
    fig.add_trace(
        go.Scatter(x=x, y=np.sin(x + i), mode='lines', name=f'Trace {i+1}'),
        row=row, col=col
    )

fig.update_layout(height=400, title_text=f"Dynamic Subplots ({n_plots} plots)")
fig.show()
```

### Conditional Subplot Layout

```python
def create_conditional_layout(data_type, data):
    """Create layout based on data type"""
    if data_type == "time_series":
        # 2x2 layout for time series analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Time Series', 'Distribution', 'Autocorrelation', 'Seasonality')
        )
        # Add time series specific plots
        fig.add_trace(go.Scatter(x=data['time'], y=data['values'], mode='lines'), row=1, col=1)
        fig.add_trace(go.Histogram(x=data['values']), row=1, col=2)
        # Add more traces...
        
    elif data_type == "comparison":
        # 1x3 layout for comparison
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Group A', 'Group B', 'Comparison')
        )
        # Add comparison specific plots
        
    return fig

# Example usage
sample_data = {'time': x, 'values': y1}
fig = create_conditional_layout("time_series", sample_data)
fig.update_layout(title_text="Conditional Layout")
fig.show()
```

## Performance Optimization

### Efficient Subplot Creation

```python
# Batch add traces for better performance
def create_efficient_subplots(data_list, plot_type='scatter'):
    """Create subplots efficiently"""
    n_plots = len(data_list)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'Plot {i+1}' for i in range(n_plots)]
    )
    
    # Batch create traces
    traces = []
    for i, data in enumerate(data_list):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        if plot_type == 'scatter':
            trace = go.Scatter(x=data['x'], y=data['y'], mode='lines', name=f'Trace {i+1}')
        elif plot_type == 'bar':
            trace = go.Bar(x=data['x'], y=data['y'], name=f'Trace {i+1}')
        
        traces.append((trace, row, col))
    
    # Add all traces at once
    for trace, row, col in traces:
        fig.add_trace(trace, row=row, col=col)
    
    return fig

# Example usage
data_list = [
    {'x': x, 'y': y1},
    {'x': x, 'y': y2},
    {'x': x, 'y': y3},
    {'x': x, 'y': y1 + y2}
]

fig = create_efficient_subplots(data_list, 'scatter')
fig.update_layout(height=400, title_text="Efficient Subplots")
fig.show()
```

### Memory Management

```python
# Use downsampling for large datasets in subplots
def create_downsampled_subplots(large_data_list, sample_size=1000):
    """Create subplots with downsampled data"""
    def downsample(data, n_points):
        if len(data) > n_points:
            step = len(data) // n_points
            return data[::step]
        return data
    
    n_plots = len(large_data_list)
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f'Plot {i+1}' for i in range(n_plots)]
    )
    
    for i, data in enumerate(large_data_list):
        row = (i // cols) + 1
        col = (i % cols) + 1
        
        # Downsample data
        x_sampled = downsample(data['x'], sample_size)
        y_sampled = downsample(data['y'], sample_size)
        
        fig.add_trace(
            go.Scatter(x=x_sampled, y=y_sampled, mode='lines', name=f'Trace {i+1}'),
            row=row, col=col
        )
    
    return fig
```

## Best Practices

### 1. Layout Consistency

```python
# Use consistent spacing and sizing
def create_consistent_layout(n_plots, plot_type='analysis'):
    """Create consistent subplot layout"""
    if plot_type == 'analysis':
        cols = 2
        row_heights = [0.6, 0.4] if n_plots > 2 else [1.0]
    elif plot_type == 'dashboard':
        cols = 3
        row_heights = [0.5, 0.3, 0.2]
    
    rows = (n_plots + cols - 1) // cols
    
    fig = make_subplots(
        rows=rows, cols=cols,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        row_heights=row_heights[:rows]
    )
    
    return fig
```

### 2. Responsive Design

```python
# Create responsive subplots
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Plot 1', 'Plot 2', 'Plot 3', 'Plot 4')
)

# Add traces
for i in range(1, 3):
    for j in range(1, 3):
        fig.add_trace(
            go.Scatter(x=x, y=np.sin(x + i*j), mode='lines', name=f'Trace {i}{j}'),
            row=i, col=j
        )

# Responsive layout
fig.update_layout(
    autosize=True,
    margin=dict(l=50, r=50, t=50, b=50),
    height=600
)

fig.show()
```

### 3. Accessibility

```python
# Accessible subplot layout
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Plot 1', 'Plot 2', 'Plot 3', 'Plot 4')
)

# Add traces with high contrast colors
colors = ['#000000', '#FF0000', '#00FF00', '#0000FF']

for i, color in enumerate(colors):
    row = (i // 2) + 1
    col = (i % 2) + 1
    fig.add_trace(
        go.Scatter(x=x, y=np.sin(x + i), mode='lines', 
                  name=f'Trace {i+1}', line=dict(color=color, width=3)),
        row=row, col=col
    )

# Large fonts for accessibility
fig.update_layout(
    font=dict(size=14),
    title=dict(font=dict(size=18)),
    height=600
)

fig.show()
```

## Common Patterns

### 1. Analysis Dashboard

```python
# Common analysis dashboard pattern
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=('Main Analysis', 'Distribution', 'Trends',
                   'Outliers', 'Correlations', 'Summary'),
    specs=[[{"colspan": 2}, None, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]],
    row_heights=[0.6, 0.4]
)

# Add analysis-specific plots
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Main'), row=1, col=1)
fig.add_trace(go.Histogram(x=np.random.randn(1000), name='Dist'), row=1, col=3)
fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Trend'), row=2, col=1)
fig.add_trace(go.Box(x=np.random.randn(100), name='Outliers'), row=2, col=2)
fig.add_trace(go.Scatter(x=y1, y=y2, mode='markers', name='Corr'), row=2, col=3)

fig.update_layout(height=600, title_text="Analysis Dashboard")
fig.show()
```

### 2. Monitoring Dashboard

```python
# Monitoring dashboard pattern
fig = make_subplots(
    rows=3, cols=4,
    subplot_titles=('Real-time Data', 'Performance', 'Alerts', 'Status',
                   'Metrics 1', 'Metrics 2', 'Metrics 3', 'Metrics 4',
                   'KPI 1', 'KPI 2', 'KPI 3', 'KPI 4'),
    specs=[[{"colspan": 2}, None, {"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]],
    row_heights=[0.4, 0.3, 0.3]
)

# Add monitoring-specific plots
fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Real-time'), row=1, col=1)
fig.add_trace(go.Indicator(mode="gauge+number", value=75, title={'text': "Performance"}), row=1, col=3)
fig.add_trace(go.Indicator(mode="number+delta", value=85, delta={'reference': 80}), row=1, col=4)

# Add metrics and KPIs
for i in range(8):
    row = (i // 4) + 2
    col = (i % 4) + 1
    if row == 2:
        fig.add_trace(go.Indicator(mode="number", value=70+i*2), row=row, col=col)
    else:
        fig.add_trace(go.Indicator(mode="gauge", value=60+i*3), row=row, col=col)

fig.update_layout(height=800, title_text="Monitoring Dashboard")
fig.show()
```

## Summary

Plotly subplots and layouts provide powerful tools for creating complex visualizations:

- **Basic Subplots**: Simple grid layouts with different plot types
- **Complex Layouts**: Uneven grids, secondary axes, and mixed plot types
- **Shared Axes**: Synchronized ranges and coordinated views
- **Figure Composition**: Multi-panel dashboards and annotated layouts
- **Performance**: Efficient creation and memory management
- **Best Practices**: Consistency, responsiveness, and accessibility

Master these techniques to create professional, informative multi-panel visualizations that effectively communicate complex data relationships.

## Next Steps

- Explore [Plotly Subplots](https://plotly.com/python/subplots/) for more examples
- Learn [Graph Objects](https://plotly.com/python/graph-objects/) for advanced customization
- Study [Dash Applications](https://dash.plotly.com/) for interactive dashboards
- Practice [Layout Templates](https://plotly.com/python/templates/) for consistency

---

**Happy Subplotting!** 📊✨ 