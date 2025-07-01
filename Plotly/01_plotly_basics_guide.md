# Plotly Basics: Complete Guide

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Plotly](https://img.shields.io/badge/Plotly-5.0+-purple.svg)](https://plotly.com/python/)
[![Dash](https://img.shields.io/badge/Dash-2.0+-blue.svg)](https://dash.plotly.com/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

## Table of Contents
1. [Introduction](#introduction)
2. [Plotly Architecture](#plotly-architecture)
3. [Installation and Setup](#installation-and-setup)
4. [Plotly Express (px)](#plotly-express-px)
5. [Graph Objects (go)](#graph-objects-go)
6. [Basic Plot Types](#basic-plot-types)
7. [Interactive Features](#interactive-features)
8. [Customization](#customization)
9. [Exporting and Sharing](#exporting-and-sharing)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)

## Introduction

Plotly is a powerful Python library for creating interactive, publication-quality graphs and dashboards. It offers both high-level (Plotly Express) and low-level (Graph Objects) interfaces for creating stunning visualizations.

### Key Features
- **Interactive by Default** - All plots support zoom, pan, hover, and selection
- **Multiple Output Formats** - HTML, PNG, SVG, PDF, and more
- **Web-Ready** - Perfect for web applications and dashboards
- **Rich Customization** - Complete control over appearance and behavior
- **Cross-Platform** - Works consistently across different systems

### Plotly Express vs Graph Objects
- **Plotly Express (px)** - High-level interface for quick plotting
- **Graph Objects (go)** - Low-level interface for complete customization

## Plotly Architecture

Plotly uses a JSON-based figure specification that can be rendered in various environments:
- **Figure** - The complete plot object containing data and layout
- **Data** - Array of traces (plot elements)
- **Layout** - Overall appearance and behavior
- **Frames** - For animations (optional)

```python
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Create sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Using Plotly Express (high-level)
fig_px = px.line(x=x, y=y, title='Sine Wave with Plotly Express')
fig_px.show()

# Using Graph Objects (low-level)
fig_go = go.Figure()
fig_go.add_trace(go.Scatter(x=x, y=y, mode='lines', name='sin(x)'))
fig_go.update_layout(title='Sine Wave with Graph Objects')
fig_go.show()
```

## Installation and Setup

### Basic Installation
```bash
pip install plotly
```

### Complete Setup (Recommended)
```bash
pip install plotly dash numpy pandas scipy scikit-learn
```

### Verify Installation
```python
import plotly
import plotly.express as px
import plotly.graph_objects as go
print(f"Plotly version: {plotly.__version__}")
```

## Plotly Express (px)

Plotly Express is the high-level interface that makes it easy to create common plot types with minimal code.

### Basic Line Plot
```python
import plotly.express as px
import numpy as np

# Create data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create line plot
fig = px.line(x=x, y=y, title='Simple Line Plot')
fig.show()
```

### Scatter Plot with Data Frame
```python
import plotly.express as px
import pandas as pd
import numpy as np

# Create sample data
np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'size': np.random.randint(10, 50, 100)
})

# Create scatter plot
fig = px.scatter(df, x='x', y='y', 
                 color='category', 
                 size='size',
                 title='Interactive Scatter Plot')
fig.show()
```

### Bar Chart
```python
import plotly.express as px

# Sample data
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

fig = px.bar(x=categories, y=values, 
             title='Simple Bar Chart',
             labels={'x': 'Category', 'y': 'Value'})
fig.show()
```

### Histogram
```python
import plotly.express as px
import numpy as np

# Generate random data
data = np.random.normal(0, 1, 1000)

fig = px.histogram(data, 
                   title='Histogram of Normal Distribution',
                   labels={'value': 'Value', 'count': 'Frequency'})
fig.show()
```

## Graph Objects (go)

Graph Objects provide complete control over plot creation and customization.

### Basic Line Plot with go
```python
import plotly.graph_objects as go
import numpy as np

# Create data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create figure
fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(
    x=x, y=y1,
    mode='lines',
    name='sin(x)',
    line=dict(color='red', width=2)
))

fig.add_trace(go.Scatter(
    x=x, y=y2,
    mode='lines',
    name='cos(x)',
    line=dict(color='blue', width=2)
))

# Update layout
fig.update_layout(
    title='Trigonometric Functions',
    xaxis_title='x',
    yaxis_title='y',
    hovermode='x unified'
)

fig.show()
```

### Scatter Plot with Multiple Traces
```python
import plotly.graph_objects as go
import numpy as np

# Generate data
np.random.seed(42)
x1 = np.random.randn(50)
y1 = np.random.randn(50)
x2 = np.random.randn(50) + 2
y2 = np.random.randn(50) + 2

fig = go.Figure()

# Add scatter traces
fig.add_trace(go.Scatter(
    x=x1, y=y1,
    mode='markers',
    name='Group 1',
    marker=dict(size=10, color='red', opacity=0.7)
))

fig.add_trace(go.Scatter(
    x=x2, y=y2,
    mode='markers',
    name='Group 2',
    marker=dict(size=10, color='blue', opacity=0.7)
))

fig.update_layout(
    title='Scatter Plot with Multiple Groups',
    xaxis_title='X',
    yaxis_title='Y'
)

fig.show()
```

## Basic Plot Types

### Line Plot
```python
import plotly.express as px
import numpy as np

x = np.linspace(0, 4*np.pi, 100)
y = np.sin(x)

fig = px.line(x=x, y=y, 
               title='Sine Wave',
               labels={'x': 'Angle (radians)', 'y': 'sin(x)'})
fig.show()
```

### Scatter Plot
```python
import plotly.express as px
import numpy as np

np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)

fig = px.scatter(x=x, y=y, 
                 title='Random Scatter Plot',
                 labels={'x': 'X', 'y': 'Y'})
fig.show()
```

### Bar Chart
```python
import plotly.express as px

categories = ['Category A', 'Category B', 'Category C', 'Category D']
values = [23, 45, 56, 78]

fig = px.bar(x=categories, y=values,
             title='Bar Chart Example',
             labels={'x': 'Categories', 'y': 'Values'})
fig.show()
```

### Pie Chart
```python
import plotly.express as px

labels = ['A', 'B', 'C', 'D']
values = [30, 25, 20, 25]

fig = px.pie(values=values, names=labels,
             title='Pie Chart Example')
fig.show()
```

### Area Plot
```python
import plotly.express as px
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig = px.area(x=x, y=y,
              title='Area Plot Example',
              labels={'x': 'X', 'y': 'sin(x)'})
fig.show()
```

## Interactive Features

### Hover Information
```python
import plotly.express as px
import pandas as pd
import numpy as np

# Create data with custom hover text
np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.randn(50),
    'y': np.random.randn(50),
    'category': np.random.choice(['A', 'B'], 50),
    'description': [f'Point {i}: ({x:.2f}, {y:.2f})' 
                   for i, (x, y) in enumerate(zip(np.random.randn(50), np.random.randn(50)))]
})

fig = px.scatter(df, x='x', y='y', 
                 color='category',
                 hover_data=['description'],
                 title='Scatter Plot with Custom Hover')
fig.show()
```

### Zoom and Pan
```python
import plotly.express as px
import numpy as np

# Create data with multiple patterns
x = np.linspace(0, 20, 200)
y1 = np.sin(x)
y2 = np.sin(x/2) * 0.5

fig = px.line(x=x, y=[y1, y2], 
               title='Interactive Line Plot (Try zooming and panning)',
               labels={'x': 'X', 'y': 'Y', 'variable': 'Function'})
fig.show()
```

### Selection Tools
```python
import plotly.express as px
import numpy as np

np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)
categories = np.random.choice(['A', 'B', 'C'], 100)

fig = px.scatter(x=x, y=y, color=categories,
                 title='Scatter Plot with Selection Tools')
fig.show()
```

## Customization

### Colors and Themes
```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Create data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create subplots with different themes
fig = make_subplots(rows=1, cols=2, subplot_titles=('Default Theme', 'Dark Theme'))

# First subplot
fig.add_trace(go.Scatter(x=x, y=y1, name='sin(x)', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=x, y=y2, name='cos(x)', line=dict(color='blue')), row=1, col=1)

# Second subplot with dark theme
fig.add_trace(go.Scatter(x=x, y=y1, name='sin(x)', line=dict(color='lightcoral')), row=1, col=2)
fig.add_trace(go.Scatter(x=x, y=y2, name='cos(x)', line=dict(color='lightblue')), row=1, col=2)

# Update layout
fig.update_layout(
    title='Customization Examples',
    showlegend=True,
    height=400
)

# Apply dark theme to second subplot
fig.update_xaxes(gridcolor='lightgray', row=1, col=2)
fig.update_yaxes(gridcolor='lightgray', row=1, col=2)
fig.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font=dict(color='white')
)

fig.show()
```

### Layout Customization
```python
import plotly.express as px
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig = px.line(x=x, y=y, title='Customized Layout')

# Customize layout
fig.update_layout(
    title={
        'text': 'Customized Sine Wave Plot',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'color': 'darkblue'}
    },
    xaxis_title='X Axis (radians)',
    yaxis_title='Y Axis (amplitude)',
    font=dict(family='Arial', size=12, color='black'),
    plot_bgcolor='lightgray',
    paper_bgcolor='white',
    width=800,
    height=500
)

# Customize axes
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='white',
    zeroline=True,
    zerolinecolor='black',
    zerolinewidth=2
)

fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='white',
    zeroline=True,
    zerolinecolor='black',
    zerolinewidth=2
)

fig.show()
```

### Annotations
```python
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

x = np.linspace(0, 4*np.pi, 100)
y = np.sin(x)

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='sin(x)'))

# Add annotations
fig.add_annotation(
    x=np.pi/2, y=1,
    text="Peak",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="red"
)

fig.add_annotation(
    x=3*np.pi/2, y=-1,
    text="Trough",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="blue"
)

fig.update_layout(
    title='Sine Wave with Annotations',
    xaxis_title='x (radians)',
    yaxis_title='sin(x)'
)

fig.show()
```

## Exporting and Sharing

### Save as HTML
```python
import plotly.express as px
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig = px.line(x=x, y=y, title='Export Example')
fig.write_html('sine_wave.html')
```

### Save as Static Image
```python
import plotly.express as px
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig = px.line(x=x, y=y, title='Static Export Example')

# Save as PNG
fig.write_image('sine_wave.png', width=800, height=600)

# Save as SVG
fig.write_image('sine_wave.svg', width=800, height=600)

# Save as PDF
fig.write_image('sine_wave.pdf', width=800, height=600)
```

### Embed in Web Pages
```python
import plotly.express as px
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig = px.line(x=x, y=y, title='Embeddable Plot')

# Get HTML string for embedding
html_string = fig.to_html(include_plotlyjs=True, full_html=True)
print("HTML string generated for embedding")
```

## Best Practices

### 1. Choose the Right Interface
```python
# Use Plotly Express for quick plots
import plotly.express as px
fig = px.scatter(df, x='x', y='y', color='category')

# Use Graph Objects for complex customization
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers'))
```

### 2. Consistent Styling
```python
# Define a consistent color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Apply to multiple plots
fig = px.scatter(df, x='x', y='y', color='category', 
                 color_discrete_sequence=colors)
```

### 3. Responsive Design
```python
# Make plots responsive
fig.update_layout(
    autosize=True,
    width=None,
    height=None
)
```

### 4. Performance Optimization
```python
# For large datasets, use downsampling
import plotly.express as px

# Sample data for large datasets
df_sampled = df.sample(n=1000) if len(df) > 1000 else df
fig = px.scatter(df_sampled, x='x', y='y')
```

### 5. Accessibility
```python
# Add descriptive titles and labels
fig.update_layout(
    title='Descriptive Title',
    xaxis_title='Clear X-axis Label',
    yaxis_title='Clear Y-axis Label'
)
```

## Troubleshooting

### Common Issues

1. **Plot not showing in Jupyter**
```python
# Make sure to import plotly.offline
import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)
```

2. **Static image export not working**
```python
# Install kaleido for static image export
# pip install kaleido
```

3. **Large file sizes**
```python
# Use compression for HTML files
fig.write_html('plot.html', include_plotlyjs='cdn')
```

4. **Performance issues with large datasets**
```python
# Use downsampling or aggregation
df_agg = df.groupby('category').mean().reset_index()
fig = px.bar(df_agg, x='category', y='value')
```

### Debugging Tips

1. **Check data types**
```python
print(df.dtypes)
print(df.head())
```

2. **Verify plot structure**
```python
print(fig.to_dict())
```

3. **Test with minimal data**
```python
# Create minimal example
fig = px.scatter(x=[1, 2, 3], y=[1, 2, 3])
fig.show()
```

## Resources

- [Plotly Python Documentation](https://plotly.com/python/)
- [Plotly Express Reference](https://plotly.com/python-api-reference/plotly.express.html)
- [Graph Objects Reference](https://plotly.com/python-api-reference/plotly.graph_objects.html)
- [Plotly Community Forum](https://community.plotly.com/)
- [Plotly Gallery](https://plotly.com/python/plotly-fundamentals/)

---

**Happy Interactive Plotting!** 📊✨ 