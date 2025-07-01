# Plotly 3D Visualization Guide

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Plotly](https://img.shields.io/badge/Plotly-5.0+-purple.svg)](https://plotly.com/python/)
[![3D](https://img.shields.io/badge/3D-Visualization-red.svg)](https://plotly.com/python/3d-plots/)

A comprehensive guide to creating stunning 3D visualizations with Plotly, including scatter plots, surface plots, wireframes, and advanced 3D plotting techniques for scientific and data analysis applications.

## Table of Contents

1. [Introduction to 3D Visualization](#introduction-to-3d-visualization)
2. [3D Scatter Plots](#3d-scatter-plots)
3. [Surface Plots](#surface-plots)
4. [Wireframe Plots](#wireframe-plots)
5. [Contour Plots](#contour-plots)
6. [3D Heatmaps](#3d-heatmaps)
7. [Scientific 3D Plotting](#scientific-3d-plotting)
8. [Interactive 3D Features](#interactive-3d-features)
9. [Performance Optimization](#performance-optimization)
10. [Best Practices](#best-practices)

## Introduction to 3D Visualization

3D visualizations add an extra dimension to data exploration, allowing you to visualize complex relationships and spatial data that would be difficult to understand in 2D.

### Why 3D Visualization?

- **Spatial Relationships** - Understand data in 3D space
- **Complex Patterns** - Discover patterns not visible in 2D
- **Scientific Applications** - Visualize mathematical functions and physical phenomena
- **Interactive Exploration** - Rotate, zoom, and explore data from different angles

### Basic Setup

```python
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

# Create sample 3D data
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
```

## 3D Scatter Plots

### Basic 3D Scatter Plot

```python
# Create 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=np.random.randn(100),
    y=np.random.randn(100),
    z=np.random.randn(100),
    mode='markers',
    marker=dict(
        size=8,
        color=np.random.randn(100),
        colorscale='Viridis',
        opacity=0.8
    )
)])

fig.update_layout(
    title="3D Scatter Plot",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis"
    )
)

fig.show()
```

### 3D Scatter with Categories

```python
# Create categorical 3D scatter plot
categories = np.random.choice(['A', 'B', 'C'], 100)
colors = {'A': 'red', 'B': 'blue', 'C': 'green'}

fig = go.Figure()

for category in ['A', 'B', 'C']:
    mask = categories == category
    fig.add_trace(go.Scatter3d(
        x=np.random.randn(100)[mask],
        y=np.random.randn(100)[mask],
        z=np.random.randn(100)[mask],
        mode='markers',
        name=category,
        marker=dict(
            size=8,
            color=colors[category],
            opacity=0.8
        )
    ))

fig.update_layout(
    title="3D Scatter Plot with Categories",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis"
    )
)

fig.show()
```

### 3D Scatter with Size Mapping

```python
# Create 3D scatter with size variation
fig = go.Figure(data=[go.Scatter3d(
    x=np.random.randn(50),
    y=np.random.randn(50),
    z=np.random.randn(50),
    mode='markers',
    marker=dict(
        size=np.random.randint(5, 20, 50),
        color=np.random.randn(50),
        colorscale='Plasma',
        opacity=0.8,
        colorbar=dict(title="Value")
    ),
    text=[f'Point {i}' for i in range(50)],
    hovertemplate='<b>%{text}</b><br>' +
                  'X: %{x}<br>' +
                  'Y: %{y}<br>' +
                  'Z: %{z}<br>' +
                  'Size: %{marker.size}<extra></extra>'
)])

fig.update_layout(
    title="3D Scatter Plot with Size Mapping",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis"
    )
)

fig.show()
```

## Surface Plots

### Basic Surface Plot

```python
# Create surface plot
fig = go.Figure(data=[go.Surface(
    x=X,
    y=Y,
    z=Z,
    colorscale='Viridis',
    colorbar=dict(title="Z Value")
)])

fig.update_layout(
    title="3D Surface Plot",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis"
    )
)

fig.show()
```

### Multiple Surfaces

```python
# Create multiple surfaces
Z1 = np.sin(np.sqrt(X**2 + Y**2))
Z2 = np.cos(np.sqrt(X**2 + Y**2))

fig = go.Figure()

fig.add_trace(go.Surface(
    x=X, y=Y, z=Z1,
    colorscale='Viridis',
    name='Surface 1',
    showscale=False
))

fig.add_trace(go.Surface(
    x=X, y=Y, z=Z2,
    colorscale='Plasma',
    name='Surface 2',
    showscale=True
))

fig.update_layout(
    title="Multiple 3D Surfaces",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis"
    )
)

fig.show()
```

### Custom Surface Colors

```python
# Create surface with custom colors
fig = go.Figure(data=[go.Surface(
    x=X,
    y=Y,
    z=Z,
    surfacecolor=np.cos(X) * np.sin(Y),  # Custom color mapping
    colorscale='RdBu',
    colorbar=dict(title="Surface Color")
)])

fig.update_layout(
    title="3D Surface with Custom Colors",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis"
    )
)

fig.show()
```

## Wireframe Plots

### Basic Wireframe

```python
# Create wireframe plot
fig = go.Figure(data=[go.Surface(
    x=X,
    y=Y,
    z=Z,
    colorscale='Viridis',
    showscale=False,
    opacity=0.8,
    contours=dict(
        x=dict(show=True, color="red", width=2),
        y=dict(show=True, color="blue", width=2),
        z=dict(show=True, color="green", width=2)
    )
)])

fig.update_layout(
    title="3D Wireframe Plot",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis"
    )
)

fig.show()
```

### Custom Wireframe

```python
# Create custom wireframe with different styles
fig = go.Figure()

# Add surface with wireframe
fig.add_trace(go.Surface(
    x=X, y=Y, z=Z,
    colorscale='Blues',
    showscale=False,
    opacity=0.3,
    contours=dict(
        x=dict(show=True, color="red", width=1, start=-5, end=5, size=0.5),
        y=dict(show=True, color="blue", width=1, start=-5, end=5, size=0.5),
        z=dict(show=True, color="green", width=1, start=-1, end=1, size=0.1)
    )
))

fig.update_layout(
    title="Custom 3D Wireframe",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis"
    )
)

fig.show()
```

## Contour Plots

### 3D Contour Plot

```python
# Create 3D contour plot
fig = go.Figure(data=[go.Surface(
    x=X,
    y=Y,
    z=Z,
    colorscale='Viridis',
    showscale=False,
    contours=dict(
        z=dict(
            show=True,
            usecolormap=True,
            highlightcolor="#42f462",
            project_z=True
        )
    )
)])

fig.update_layout(
    title="3D Contour Plot",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis"
    )
)

fig.show()
```

### Multiple Contour Levels

```python
# Create contour plot with multiple levels
fig = go.Figure(data=[go.Surface(
    x=X,
    y=Y,
    z=Z,
    colorscale='Viridis',
    showscale=True,
    contours=dict(
        z=dict(
            show=True,
            usecolormap=True,
            highlightcolor="#42f462",
            project_z=True,
            start=-1,
            end=1,
            size=0.1
        )
    )
)])

fig.update_layout(
    title="3D Contour Plot with Multiple Levels",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis"
    )
)

fig.show()
```

## 3D Heatmaps

### Basic 3D Heatmap

```python
# Create 3D heatmap
fig = go.Figure(data=[go.Surface(
    x=X,
    y=Y,
    z=Z,
    colorscale='Hot',
    colorbar=dict(title="Temperature")
)])

fig.update_layout(
    title="3D Heatmap",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Temperature"
    )
)

fig.show()
```

### Custom 3D Heatmap

```python
# Create custom 3D heatmap with different data
data = np.random.rand(20, 20)
x_coords = np.arange(20)
y_coords = np.arange(20)
X_heat, Y_heat = np.meshgrid(x_coords, y_coords)

fig = go.Figure(data=[go.Surface(
    x=X_heat,
    y=Y_heat,
    z=data,
    colorscale='Viridis',
    colorbar=dict(title="Value")
)])

fig.update_layout(
    title="Custom 3D Heatmap",
    scene=dict(
        xaxis_title="X Index",
        yaxis_title="Y Index",
        zaxis_title="Value"
    )
)

fig.show()
```

## Scientific 3D Plotting

### Mathematical Functions

```python
# Create 3D plot of mathematical functions
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Different mathematical functions
functions = {
    'Paraboloid': X**2 + Y**2,
    'Saddle': X**2 - Y**2,
    'Ripple': np.sin(np.sqrt(X**2 + Y**2)),
    'Gaussian': np.exp(-(X**2 + Y**2)/2)
}

fig = go.Figure()

for name, Z in functions.items():
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        name=name,
        showscale=False
    ))

fig.update_layout(
    title="Mathematical Functions in 3D",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    )
)

fig.show()
```

### Vector Fields

```python
# Create 3D vector field
x = np.linspace(-2, 2, 10)
y = np.linspace(-2, 2, 10)
z = np.linspace(-2, 2, 10)
X, Y, Z = np.meshgrid(x, y, z)

# Vector field components
U = Y
V = -X
W = Z

fig = go.Figure(data=[go.Cone(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    u=U.flatten(),
    v=V.flatten(),
    w=W.flatten(),
    colorscale='Viridis',
    sizemode="absolute",
    sizeref=0.5
)])

fig.update_layout(
    title="3D Vector Field",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    )
)

fig.show()
```

### Molecular Visualization

```python
# Simple molecular-like structure
# Create atoms (points) and bonds (lines)
atoms = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
])

bonds = [
    [0, 1], [0, 2], [0, 3],  # Bonds from center atom
    [1, 4], [1, 5],          # Bonds from corner atoms
    [2, 4], [2, 6],
    [3, 5], [3, 6],
    [4, 7], [5, 7], [6, 7]   # Bonds to opposite corner
]

fig = go.Figure()

# Add atoms
fig.add_trace(go.Scatter3d(
    x=atoms[:, 0],
    y=atoms[:, 1],
    z=atoms[:, 2],
    mode='markers',
    marker=dict(
        size=10,
        color='red',
        opacity=0.8
    ),
    name='Atoms'
))

# Add bonds
for bond in bonds:
    fig.add_trace(go.Scatter3d(
        x=[atoms[bond[0], 0], atoms[bond[1], 0]],
        y=[atoms[bond[0], 1], atoms[bond[1], 1]],
        z=[atoms[bond[0], 2], atoms[bond[1], 2]],
        mode='lines',
        line=dict(color='blue', width=3),
        showlegend=False
    ))

fig.update_layout(
    title="Simple Molecular Structure",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z"
    )
)

fig.show()
```

## Interactive 3D Features

### Camera Controls

```python
# Create 3D plot with custom camera controls
fig = go.Figure(data=[go.Surface(
    x=X,
    y=Y,
    z=Z,
    colorscale='Viridis'
)])

fig.update_layout(
    title="3D Plot with Camera Controls",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis",
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5),  # Camera position
            center=dict(x=0, y=0, z=0)       # Look at point
        )
    )
)

fig.show()
```

### Animation in 3D

```python
# Create animated 3D surface
frames = []
for i in range(20):
    Z_animated = np.sin(np.sqrt(X**2 + Y**2) + i * 0.5)
    frame = go.Frame(
        data=[go.Surface(x=X, y=Y, z=Z_animated, colorscale='Viridis')],
        name=f'frame{i}'
    )
    frames.append(frame)

fig = go.Figure(
    data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')],
    frames=frames
)

fig.update_layout(
    title="Animated 3D Surface",
    updatemenus=[{
        'type': 'buttons',
        'showactive': False,
        'buttons': [
            {
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 100, 'redraw': True},
                    'fromcurrent': True
                }]
            },
            {
                'label': 'Pause',
                'method': 'animate',
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': False},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }
        ]
    }]
)

fig.show()
```

### Interactive Selection

```python
# Create 3D scatter with selection
fig = go.Figure(data=[go.Scatter3d(
    x=np.random.randn(100),
    y=np.random.randn(100),
    z=np.random.randn(100),
    mode='markers',
    marker=dict(
        size=8,
        color=np.random.randn(100),
        colorscale='Viridis',
        opacity=0.8
    ),
    selected=dict(
        marker=dict(color='red', size=12)
    ),
    unselected=dict(
        marker=dict(opacity=0.3)
    )
)])

fig.update_layout(
    title="3D Scatter with Selection",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis"
    )
)

fig.show()
```

## Performance Optimization

### Large Dataset Handling

```python
# Optimize 3D plots for large datasets
def create_optimized_3d_plot(data, max_points=10000):
    """Create optimized 3D plot for large datasets"""
    if len(data) > max_points:
        # Downsample data
        indices = np.random.choice(len(data), max_points, replace=False)
        data = data[indices]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color=data[:, 2],  # Color by z-value
            colorscale='Viridis',
            opacity=0.7
        )
    ))
    
    return fig

# Example usage
large_data = np.random.randn(50000, 3)
fig = create_optimized_3d_plot(large_data)
fig.update_layout(title="Optimized 3D Plot for Large Dataset")
fig.show()
```

### Efficient Surface Rendering

```python
# Create efficient surface plot
def create_efficient_surface(func, x_range=(-5, 5), y_range=(-5, 5), resolution=50):
    """Create efficient surface plot with custom resolution"""
    x = np.linspace(x_range[0], x_range[1], resolution)
    y = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    
    fig = go.Figure(data=[go.Surface(
        x=X,
        y=Y,
        z=Z,
        colorscale='Viridis',
        showscale=True
    )])
    
    return fig

# Example usage
def sample_function(X, Y):
    return np.sin(np.sqrt(X**2 + Y**2))

fig = create_efficient_surface(sample_function, resolution=30)
fig.update_layout(title="Efficient Surface Plot")
fig.show()
```

## Best Practices

### 1. Color and Contrast

```python
# Use appropriate color scales for 3D plots
fig = go.Figure(data=[go.Surface(
    x=X,
    y=Y,
    z=Z,
    colorscale='Viridis',  # Good for scientific data
    colorbar=dict(title="Value")
)])

fig.update_layout(
    title="3D Plot with Good Color Contrast",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis",
        bgcolor='white'  # White background for better contrast
    )
)

fig.show()
```

### 2. Axis Labels and Titles

```python
# Clear axis labels and titles
fig = go.Figure(data=[go.Surface(
    x=X,
    y=Y,
    z=Z,
    colorscale='Plasma'
)])

fig.update_layout(
    title="3D Plot with Clear Labels",
    scene=dict(
        xaxis_title="X Coordinate (units)",
        yaxis_title="Y Coordinate (units)",
        zaxis_title="Z Value (units)",
        xaxis=dict(showgrid=True, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridcolor='lightgray'),
        zaxis=dict(showgrid=True, gridcolor='lightgray')
    )
)

fig.show()
```

### 3. Interactive Features

```python
# Add helpful interactive features
fig = go.Figure(data=[go.Surface(
    x=X,
    y=Y,
    z=Z,
    colorscale='Viridis',
    hovertemplate='<b>Surface Point</b><br>' +
                  'X: %{x:.2f}<br>' +
                  'Y: %{y:.2f}<br>' +
                  'Z: %{z:.2f}<extra></extra>'
)])

fig.update_layout(
    title="3D Plot with Interactive Features",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis",
        camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.5)
        )
    )
)

fig.show()
```

### 4. Accessibility

```python
# Make 3D plots accessible
fig = go.Figure(data=[go.Surface(
    x=X,
    y=Y,
    z=Z,
    colorscale='Viridis',
    colorbar=dict(title="Value")
)])

fig.update_layout(
    title="Accessible 3D Plot",
    scene=dict(
        xaxis_title="X Axis",
        yaxis_title="Y Axis",
        zaxis_title="Z Axis",
        xaxis=dict(showgrid=True, gridcolor='black', gridwidth=1),
        yaxis=dict(showgrid=True, gridcolor='black', gridwidth=1),
        zaxis=dict(showgrid=True, gridcolor='black', gridwidth=1)
    ),
    font=dict(size=14)  # Larger fonts for accessibility
)

fig.show()
```

## Summary

Plotly 3D visualization provides powerful tools for exploring data in three dimensions:

- **3D Scatter Plots**: Visualize point clouds and categorical data
- **Surface Plots**: Display continuous functions and mathematical surfaces
- **Wireframes**: Show structural relationships and contours
- **Scientific Applications**: Mathematical functions, vector fields, and molecular structures
- **Interactive Features**: Camera controls, animations, and selection tools
- **Performance**: Optimization for large datasets and efficient rendering

Master these 3D visualization techniques to create compelling, interactive visualizations that reveal complex spatial relationships in your data.

## Next Steps

- Explore [Plotly 3D Plots](https://plotly.com/python/3d-plots/) for more examples
- Learn [Graph Objects](https://plotly.com/python/graph-objects/) for advanced 3D customization
- Study [Scientific Visualization](https://plotly.com/python/scientific-charts/) techniques
- Practice [Interactive Features](https://plotly.com/python/interactive-plots/) for 3D plots

---

**Happy 3D Plotting!** 📊✨ 