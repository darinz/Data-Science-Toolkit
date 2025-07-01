# Plotly Customization Guide

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Plotly](https://img.shields.io/badge/Plotly-5.0+-purple.svg)](https://plotly.com/python/)
[![Customization](https://img.shields.io/badge/Customization-Styling-orange.svg)](https://plotly.com/python/styling-plotly-express/)

A comprehensive guide to customizing Plotly visualizations, including colors, themes, layouts, annotations, and advanced styling techniques for creating professional and branded data visualizations.

## Table of Contents

1. [Introduction to Customization](#introduction-to-customization)
2. [Color Schemes and Palettes](#color-schemes-and-palettes)
3. [Layout and Templates](#layout-and-templates)
4. [Annotations and Shapes](#annotations-and-shapes)
5. [Axes and Grid Customization](#axes-and-grid-customization)
6. [Legend Customization](#legend-customization)
7. [Fonts and Typography](#fonts-and-typography)
8. [Themes and Templates](#themes-and-templates)
9. [Advanced Styling](#advanced-styling)
10. [Best Practices](#best-practices)

## Introduction to Customization

Plotly provides extensive customization options to make your visualizations look professional and match your brand identity.

### Why Customization Matters

- **Brand Consistency** - Match your organization's visual identity
- **Professional Appearance** - Create polished, publication-ready plots
- **User Experience** - Improve readability and comprehension
- **Accessibility** - Ensure plots work for all users

### Basic Customization Workflow

```python
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd

# Create sample data
df = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [10, 20, 15, 25, 30],
    'category': ['A', 'B', 'A', 'B', 'A']
})

# Basic plot
fig = px.scatter(df, x='x', y='y', color='category')
fig.show()
```

## Color Schemes and Palettes

### Built-in Color Palettes

```python
import plotly.express as px

# Available color sequences
color_sequences = [
    'plotly', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_reds',
    'viridis', 'plasma', 'inferno', 'magma', 'cividis'
]

# Create plots with different color schemes
fig = px.scatter(df, x='x', y='y', color='category',
                 color_discrete_sequence=px.colors.qualitative.Set1,
                 title="Custom Color Palette")

fig.show()
```

### Custom Color Palettes

```python
# Define custom colors
custom_colors = {
    'A': '#1f77b4',  # Blue
    'B': '#ff7f0e',  # Orange
    'C': '#2ca02c',  # Green
    'D': '#d62728'   # Red
}

fig = px.scatter(df, x='x', y='y', color='category',
                 color_discrete_map=custom_colors,
                 title="Custom Color Mapping")

fig.show()
```

### Continuous Color Scales

```python
# For continuous data
df_continuous = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'value': np.random.rand(100)
})

fig = px.scatter(df_continuous, x='x', y='y', color='value',
                 color_continuous_scale='viridis',
                 title="Continuous Color Scale")

fig.show()
```

### Colorblind-Friendly Palettes

```python
# Colorblind-friendly colors
colorblind_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig = px.scatter(df, x='x', y='y', color='category',
                 color_discrete_sequence=colorblind_colors,
                 title="Colorblind-Friendly Palette")

fig.show()
```

## Layout and Templates

### Basic Layout Customization

```python
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[10, 20, 15, 25, 30],
    mode='markers+lines',
    name='Data Series'
))

# Customize layout
fig.update_layout(
    title={
        'text': "Customized Plot Title",
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'color': 'darkblue'}
    },
    xaxis_title="X Axis Label",
    yaxis_title="Y Axis Label",
    width=800,
    height=600,
    showlegend=True,
    plot_bgcolor='white',
    paper_bgcolor='lightgray'
)

fig.show()
```

### Layout Templates

```python
# Available templates
templates = [
    'plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn',
    'simple_white', 'none'
]

# Use a template
fig = px.scatter(df, x='x', y='y', color='category',
                 template='plotly_white',
                 title="Using plotly_white Template")

fig.show()
```

### Custom Templates

```python
import plotly.io as pio

# Create custom template
custom_template = dict(
    layout=dict(
        font=dict(family="Arial, sans-serif", size=12, color="black"),
        title=dict(font=dict(size=16, color="darkblue")),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            gridcolor="lightgray",
            zerolinecolor="black",
            zerolinewidth=2
        ),
        yaxis=dict(
            gridcolor="lightgray",
            zerolinecolor="black",
            zerolinewidth=2
        )
    )
)

# Register template
pio.templates["custom"] = custom_template

# Use custom template
fig = px.scatter(df, x='x', y='y', color='category',
                 template="custom",
                 title="Custom Template")

fig.show()
```

## Annotations and Shapes

### Text Annotations

```python
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[10, 20, 15, 25, 30],
    mode='markers+lines'
))

# Add text annotations
fig.add_annotation(
    x=3,
    y=15,
    text="Peak Point",
    showarrow=True,
    arrowhead=2,
    arrowsize=1,
    arrowwidth=2,
    arrowcolor="red",
    ax=40,
    ay=-40,
    font=dict(size=14, color="red")
)

fig.add_annotation(
    x=1,
    y=10,
    text="Starting Point",
    showarrow=False,
    font=dict(size=12, color="blue"),
    bgcolor="lightblue",
    bordercolor="blue",
    borderwidth=1
)

fig.update_layout(title="Text Annotations")
fig.show()
```

### Shapes and Lines

```python
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[10, 20, 15, 25, 30],
    mode='markers+lines'
))

# Add shapes
fig.add_shape(
    type="rect",
    x0=2, y0=10, x1=4, y1=25,
    fillcolor="lightblue",
    opacity=0.3,
    layer="below",
    line_width=0
)

fig.add_shape(
    type="line",
    x0=1, y0=10, x1=5, y1=30,
    line=dict(color="red", width=2, dash="dash")
)

fig.add_shape(
    type="circle",
    x0=2.5, y0=15, x1=3.5, y1=20,
    fillcolor="yellow",
    opacity=0.5,
    line=dict(color="orange", width=2)
)

fig.update_layout(title="Shapes and Lines")
fig.show()
```

### Advanced Annotations

```python
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[10, 20, 15, 25, 30],
    mode='markers+lines'
))

# Multiple annotations with different styles
annotations = [
    dict(
        x=2, y=20,
        text="<b>Important Point</b><br>Value: 20",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="red",
        ax=40, ay=-40,
        font=dict(size=12, color="red"),
        bgcolor="white",
        bordercolor="red",
        borderwidth=1
    ),
    dict(
        x=4, y=25,
        text="Trend Line",
        showarrow=True,
        arrowhead=1,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="blue",
        ax=-40, ay=40,
        font=dict(size=10, color="blue")
    )
]

fig.update_layout(
    title="Advanced Annotations",
    annotations=annotations
)

fig.show()
```

## Axes and Grid Customization

### Axis Customization

```python
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[10, 20, 15, 25, 30],
    mode='markers+lines'
))

# Customize axes
fig.update_xaxes(
    title="X Axis Title",
    titlefont=dict(size=14, color="darkblue"),
    tickfont=dict(size=12, color="black"),
    tickmode="linear",
    tick0=1,
    dtick=1,
    gridcolor="lightgray",
    gridwidth=1,
    zeroline=True,
    zerolinecolor="black",
    zerolinewidth=2,
    showline=True,
    linecolor="black",
    linewidth=2
)

fig.update_yaxes(
    title="Y Axis Title",
    titlefont=dict(size=14, color="darkblue"),
    tickfont=dict(size=12, color="black"),
    gridcolor="lightgray",
    gridwidth=1,
    zeroline=True,
    zerolinecolor="black",
    zerolinewidth=2,
    showline=True,
    linecolor="black",
    linewidth=2
)

fig.update_layout(title="Customized Axes")
fig.show()
```

### Grid Customization

```python
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[10, 20, 15, 25, 30],
    mode='markers+lines'
))

# Customize grid
fig.update_xaxes(
    gridcolor="lightblue",
    gridwidth=1,
    griddash="dot",
    showgrid=True
)

fig.update_yaxes(
    gridcolor="lightgreen",
    gridwidth=1,
    griddash="dash",
    showgrid=True
)

fig.update_layout(
    title="Custom Grid",
    plot_bgcolor="white"
)

fig.show()
```

### Range and Scale Customization

```python
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[10, 20, 15, 25, 30],
    mode='markers+lines'
))

# Set axis ranges
fig.update_xaxes(range=[0, 6])
fig.update_yaxes(range=[0, 35])

# Log scale (if appropriate)
# fig.update_yaxes(type="log")

fig.update_layout(title="Custom Ranges")
fig.show()
```

## Legend Customization

### Basic Legend Customization

```python
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[10, 20, 15, 25, 30],
    mode='markers+lines',
    name='Series 1'
))

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[15, 25, 20, 30, 35],
    mode='markers+lines',
    name='Series 2'
))

# Customize legend
fig.update_layout(
    legend=dict(
        title="Legend Title",
        titlefont=dict(size=14, color="darkblue"),
        font=dict(size=12, color="black"),
        bgcolor="lightgray",
        bordercolor="black",
        borderwidth=1,
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top"
    )
)

fig.show()
```

### Advanced Legend Features

```python
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[10, 20, 15, 25, 30],
    mode='markers+lines',
    name='Data Series',
    legendgroup="group1",
    legendgrouptitle_text="Group 1"
))

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[15, 25, 20, 30, 35],
    mode='markers+lines',
    name='Another Series',
    legendgroup="group1"
))

fig.update_layout(
    legend=dict(
        groupclick="toggleitem",  # 'toggleitem' or 'togglegroup'
        itemsizing="constant",
        itemwidth=30
    )
)

fig.show()
```

## Fonts and Typography

### Font Customization

```python
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[10, 20, 15, 25, 30],
    mode='markers+lines'
))

# Global font settings
fig.update_layout(
    font=dict(
        family="Arial, sans-serif",
        size=12,
        color="black"
    ),
    title=dict(
        text="Custom Font Title",
        font=dict(
            family="Times New Roman, serif",
            size=18,
            color="darkblue"
        )
    ),
    xaxis_title="X Axis with Custom Font",
    yaxis_title="Y Axis with Custom Font"
)

fig.show()
```

### Typography Best Practices

```python
# Font hierarchy
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[10, 20, 15, 25, 30],
    mode='markers+lines'
))

fig.update_layout(
    title=dict(
        text="Main Title (Largest)",
        font=dict(size=20, family="Arial, sans-serif", color="black")
    ),
    xaxis_title=dict(
        text="Axis Title (Medium)",
        font=dict(size=14, family="Arial, sans-serif", color="darkgray")
    ),
    yaxis_title=dict(
        text="Axis Title (Medium)",
        font=dict(size=14, family="Arial, sans-serif", color="darkgray")
    ),
    font=dict(
        size=12,  # Default text size
        family="Arial, sans-serif",
        color="black"
    )
)

fig.show()
```

## Themes and Templates

### Built-in Themes

```python
# Available themes
themes = [
    'plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn',
    'simple_white', 'none'
]

# Example with different themes
fig = px.scatter(df, x='x', y='y', color='category')

for theme in ['plotly', 'plotly_white', 'ggplot2']:
    fig_temp = fig.update_layout(template=theme)
    fig_temp.update_layout(title=f"Theme: {theme}")
    fig_temp.show()
```

### Custom Theme Creation

```python
import plotly.io as pio

# Define custom theme
custom_theme = dict(
    layout=dict(
        # Colors
        plot_bgcolor='white',
        paper_bgcolor='white',
        
        # Fonts
        font=dict(
            family='Arial, sans-serif',
            size=12,
            color='black'
        ),
        
        # Title
        title=dict(
            font=dict(size=18, color='darkblue'),
            x=0.5,
            xanchor='center'
        ),
        
        # Axes
        xaxis=dict(
            title=dict(font=dict(size=14, color='darkblue')),
            tickfont=dict(size=11, color='black'),
            gridcolor='lightgray',
            zerolinecolor='black',
            zerolinewidth=1,
            showline=True,
            linecolor='black',
            linewidth=1
        ),
        
        yaxis=dict(
            title=dict(font=dict(size=14, color='darkblue')),
            tickfont=dict(size=11, color='black'),
            gridcolor='lightgray',
            zerolinecolor='black',
            zerolinewidth=1,
            showline=True,
            linecolor='black',
            linewidth=1
        ),
        
        # Legend
        legend=dict(
            font=dict(size=11, color='black'),
            bgcolor='white',
            bordercolor='black',
            borderwidth=1
        )
    )
)

# Register theme
pio.templates["professional"] = custom_theme

# Use theme
fig = px.scatter(df, x='x', y='y', color='category',
                 template="professional",
                 title="Professional Theme")

fig.show()
```

## Advanced Styling

### Conditional Styling

```python
import plotly.graph_objects as go

# Create data with conditions
x = [1, 2, 3, 4, 5]
y = [10, 20, 15, 25, 30]
colors = ['red' if val > 20 else 'blue' for val in y]
sizes = [20 if val > 20 else 10 for val in y]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(
        color=colors,
        size=sizes,
        line=dict(color='black', width=1)
    ),
    text=[f"Value: {val}" for val in y],
    hovertemplate="<b>Point %{pointNumber}</b><br>" +
                  "X: %{x}<br>" +
                  "Y: %{y}<br>" +
                  "Text: %{text}<extra></extra>"
))

fig.update_layout(title="Conditional Styling")
fig.show()
```

### Gradient and Pattern Fills

```python
fig = go.Figure()

# Create area plot with gradient
fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[10, 20, 15, 25, 30],
    fill='tonexty',  # Fill to y=0
    fillcolor='rgba(0,100,80,0.2)',
    line=dict(color='rgb(0,100,80)'),
    name='Area with Gradient'
))

fig.update_layout(
    title="Gradient and Pattern Fills",
    plot_bgcolor='white'
)

fig.show()
```

### Custom Markers and Symbols

```python
fig = go.Figure()

# Different marker symbols
symbols = ['circle', 'square', 'diamond', 'triangle-up', 'star']

for i, symbol in enumerate(symbols):
    fig.add_trace(go.Scatter(
        x=[i+1],
        y=[10 + i*5],
        mode='markers',
        marker=dict(
            symbol=symbol,
            size=20,
            color=f'rgb({50+i*40},{100+i*30},{150+i*20})',
            line=dict(color='black', width=2)
        ),
        name=f'Symbol: {symbol}'
    ))

fig.update_layout(
    title="Custom Markers and Symbols",
    xaxis=dict(range=[0, 6]),
    yaxis=dict(range=[0, 35])
)

fig.show()
```

## Best Practices

### 1. Color Consistency

```python
# Define brand colors
brand_colors = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'accent': '#2ca02c',
    'neutral': '#7f7f7f'
}

# Use consistently across plots
fig = px.scatter(df, x='x', y='y', color='category',
                 color_discrete_sequence=[brand_colors['primary'], 
                                        brand_colors['secondary'],
                                        brand_colors['accent']])

fig.update_layout(
    title=dict(font=dict(color=brand_colors['primary'])),
    xaxis_title=dict(font=dict(color=brand_colors['neutral'])),
    yaxis_title=dict(font=dict(color=brand_colors['neutral']))
)

fig.show()
```

### 2. Accessibility

```python
# High contrast colors
high_contrast_colors = ['#000000', '#FFFFFF', '#FF0000', '#00FF00', '#0000FF']

# Large fonts for readability
fig = px.scatter(df, x='x', y='y', color='category',
                 color_discrete_sequence=high_contrast_colors)

fig.update_layout(
    font=dict(size=14),  # Larger base font
    title=dict(font=dict(size=18)),
    xaxis_title=dict(font=dict(size=16)),
    yaxis_title=dict(font=dict(size=16))
)

fig.show()
```

### 3. Mobile Responsiveness

```python
# Responsive design
fig = px.scatter(df, x='x', y='y', color='category')

fig.update_layout(
    autosize=True,
    margin=dict(l=50, r=50, t=50, b=50),
    font=dict(size=12),  # Readable on mobile
    legend=dict(
        orientation="h",  # Horizontal legend for mobile
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

fig.show()
```

### 4. Performance Optimization

```python
# Efficient styling updates
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[10, 20, 15, 25, 30],
    mode='markers+lines'
))

# Batch layout updates
fig.update_layout(
    title="Efficient Styling",
    xaxis_title="X Axis",
    yaxis_title="Y Axis",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(size=12, family="Arial")
)

# Batch trace updates
fig.update_traces(
    marker=dict(size=10, color="blue"),
    line=dict(width=2, color="blue")
)

fig.show()
```

## Summary

Plotly customization provides powerful tools for creating professional visualizations:

- **Colors**: Built-in palettes, custom colors, and accessibility considerations
- **Layout**: Templates, themes, and responsive design
- **Annotations**: Text, shapes, and interactive elements
- **Typography**: Font families, sizes, and hierarchy
- **Best Practices**: Consistency, accessibility, and performance

Master these customization techniques to create stunning, professional data visualizations that effectively communicate your insights.

## Next Steps

- Explore [Plotly Express](https://plotly.com/python/plotly-express/) for quick customization
- Learn [Graph Objects](https://plotly.com/python/graph-objects/) for advanced styling
- Study [Templates](https://plotly.com/python/templates/) for consistent themes
- Practice [Accessibility](https://plotly.com/python/accessibility/) guidelines

---

**Happy Customizing!** 🎨✨ 