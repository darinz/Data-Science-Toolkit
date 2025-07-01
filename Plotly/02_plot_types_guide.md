# Plotly Plot Types: Complete Guide

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Plotly](https://img.shields.io/badge/Plotly-5.0+-purple.svg)](https://plotly.com/python/)
[![Dash](https://img.shields.io/badge/Dash-2.0+-blue.svg)](https://dash.plotly.com/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

## Table of Contents
1. [Introduction](#introduction)
2. [Line Plots](#line-plots)
3. [Scatter Plots](#scatter-plots)
4. [Bar Charts](#bar-charts)
5. [Histograms](#histograms)
6. [Pie Charts](#pie-charts)
7. [Area Plots](#area-plots)
8. [Box Plots](#box-plots)
9. [Violin Plots](#violin-plots)
10. [Heatmaps](#heatmaps)
11. [Contour Plots](#contour-plots)
12. [3D Plots](#3d-plots)
13. [Polar Plots](#polar-plots)
14. [Bubble Charts](#bubble-charts)
15. [Funnel Charts](#funnel-charts)
16. [Waterfall Charts](#waterfall-charts)
17. [Best Practices](#best-practices)

## Introduction

Plotly offers a wide variety of plot types for different data visualization needs. This guide covers the most commonly used plot types with practical examples and customization options.

### Plot Categories
- **Basic Plots**: Line, scatter, bar, histogram, pie
- **Statistical Plots**: Box, violin, histogram, density
- **Specialized Plots**: Heatmap, contour, 3D, polar
- **Business Plots**: Funnel, waterfall, bubble

## Line Plots

Line plots are ideal for showing trends over time or continuous relationships.

### Basic Line Plot
```python
import plotly.express as px
import numpy as np

# Create data
x = np.linspace(0, 10, 100)
y = np.sin(x)

fig = px.line(x=x, y=y, title='Basic Line Plot')
fig.show()
```

### Multiple Lines
```python
import plotly.express as px
import numpy as np

x = np.linspace(0, 4*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x/2)

fig = px.line(x=x, y=[y1, y2, y3], 
               title='Multiple Lines',
               labels={'x': 'Angle (radians)', 'y': 'Value', 'variable': 'Function'})
fig.show()
```

### Line Plot with Markers
```python
import plotly.graph_objects as go
import numpy as np

x = np.linspace(0, 10, 20)
y = np.sin(x)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=x, y=y,
    mode='lines+markers',
    name='sin(x)',
    line=dict(color='red', width=2),
    marker=dict(size=8, color='red')
))

fig.update_layout(
    title='Line Plot with Markers',
    xaxis_title='x',
    yaxis_title='sin(x)'
)
fig.show()
```

### Area Line Plot
```python
import plotly.express as px
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig = px.area(x=x, y=y, 
               title='Area Line Plot',
               labels={'x': 'x', 'y': 'sin(x)'})
fig.show()
```

## Scatter Plots

Scatter plots are perfect for showing relationships between two variables.

### Basic Scatter Plot
```python
import plotly.express as px
import numpy as np

np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)

fig = px.scatter(x=x, y=y, 
                  title='Basic Scatter Plot',
                  labels={'x': 'X', 'y': 'Y'})
fig.show()
```

### Scatter Plot with Color Mapping
```python
import plotly.express as px
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'size': np.random.randint(10, 50, 100)
})

fig = px.scatter(df, x='x', y='y', 
                 color='category',
                 size='size',
                 title='Scatter Plot with Color and Size',
                 labels={'x': 'X', 'y': 'Y', 'category': 'Category'})
fig.show()
```

### Scatter Plot with Trend Line
```python
import plotly.express as px
import numpy as np

np.random.seed(42)
x = np.linspace(0, 10, 50)
y = 2*x + 1 + np.random.normal(0, 1, 50)

fig = px.scatter(x=x, y=y, 
                  title='Scatter Plot with Trend Line',
                  trendline='ols',
                  labels={'x': 'X', 'y': 'Y'})
fig.show()
```

### Bubble Chart
```python
import plotly.express as px
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.randn(50),
    'y': np.random.randn(50),
    'size': np.random.randint(10, 100, 50),
    'category': np.random.choice(['A', 'B', 'C'], 50)
})

fig = px.scatter(df, x='x', y='y', 
                 size='size',
                 color='category',
                 title='Bubble Chart',
                 labels={'x': 'X', 'y': 'Y', 'size': 'Size'})
fig.show()
```

## Bar Charts

Bar charts are excellent for comparing categories or showing discrete data.

### Basic Bar Chart
```python
import plotly.express as px

categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

fig = px.bar(x=categories, y=values,
             title='Basic Bar Chart',
             labels={'x': 'Category', 'y': 'Value'})
fig.show()
```

### Horizontal Bar Chart
```python
import plotly.express as px

categories = ['Category A', 'Category B', 'Category C', 'Category D']
values = [23, 45, 56, 78]

fig = px.bar(x=values, y=categories,
             orientation='h',
             title='Horizontal Bar Chart',
             labels={'x': 'Value', 'y': 'Category'})
fig.show()
```

### Grouped Bar Chart
```python
import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'C', 'C'],
    'group': ['Group 1', 'Group 2', 'Group 1', 'Group 2', 'Group 1', 'Group 2'],
    'value': [10, 15, 20, 25, 30, 35]
})

fig = px.bar(df, x='category', y='value',
             color='group',
             title='Grouped Bar Chart',
             barmode='group')
fig.show()
```

### Stacked Bar Chart
```python
import plotly.express as px
import pandas as pd

df = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'C', 'C'],
    'group': ['Group 1', 'Group 2', 'Group 1', 'Group 2', 'Group 1', 'Group 2'],
    'value': [10, 15, 20, 25, 30, 35]
})

fig = px.bar(df, x='category', y='value',
             color='group',
             title='Stacked Bar Chart',
             barmode='stack')
fig.show()
```

## Histograms

Histograms show the distribution of data.

### Basic Histogram
```python
import plotly.express as px
import numpy as np

data = np.random.normal(0, 1, 1000)

fig = px.histogram(data, 
                   title='Histogram of Normal Distribution',
                   labels={'value': 'Value', 'count': 'Frequency'})
fig.show()
```

### Histogram with Multiple Distributions
```python
import plotly.express as px
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'value': np.concatenate([
        np.random.normal(0, 1, 500),
        np.random.normal(3, 1, 500)
    ]),
    'group': ['Group A'] * 500 + ['Group B'] * 500
})

fig = px.histogram(df, x='value',
                   color='group',
                   title='Histogram with Multiple Groups',
                   opacity=0.7,
                   barmode='overlay')
fig.show()
```

### Density Histogram
```python
import plotly.express as px
import numpy as np

data = np.random.normal(0, 1, 1000)

fig = px.histogram(data, 
                   title='Density Histogram',
                   histnorm='density',
                   labels={'value': 'Value', 'density': 'Density'})
fig.show()
```

## Pie Charts

Pie charts show proportions of a whole.

### Basic Pie Chart
```python
import plotly.express as px

labels = ['A', 'B', 'C', 'D']
values = [30, 25, 20, 25]

fig = px.pie(values=values, names=labels,
             title='Basic Pie Chart')
fig.show()
```

### Donut Chart
```python
import plotly.express as px

labels = ['A', 'B', 'C', 'D']
values = [30, 25, 20, 25]

fig = px.pie(values=values, names=labels,
             title='Donut Chart',
             hole=0.4)
fig.show()
```

### Pie Chart with Custom Colors
```python
import plotly.express as px

labels = ['A', 'B', 'C', 'D']
values = [30, 25, 20, 25]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

fig = px.pie(values=values, names=labels,
             title='Pie Chart with Custom Colors',
             color_discrete_sequence=colors)
fig.show()
```

## Area Plots

Area plots show cumulative data or filled regions.

### Basic Area Plot
```python
import plotly.express as px
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

fig = px.area(x=x, y=y,
               title='Basic Area Plot',
               labels={'x': 'x', 'y': 'sin(x)'})
fig.show()
```

### Stacked Area Plot
```python
import plotly.express as px
import pandas as pd
import numpy as np

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

df = pd.DataFrame({
    'x': np.concatenate([x, x]),
    'y': np.concatenate([y1, y2]),
    'function': ['sin(x)'] * 100 + ['cos(x)'] * 100
})

fig = px.area(df, x='x', y='y',
              color='function',
              title='Stacked Area Plot')
fig.show()
```

## Box Plots

Box plots show the distribution of data through quartiles.

### Basic Box Plot
```python
import plotly.express as px
import numpy as np

np.random.seed(42)
data = np.random.normal(0, 1, 1000)

fig = px.box(y=data,
             title='Basic Box Plot',
             labels={'y': 'Value'})
fig.show()
```

### Grouped Box Plot
```python
import plotly.express as px
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'value': np.concatenate([
        np.random.normal(0, 1, 200),
        np.random.normal(2, 1, 200),
        np.random.normal(4, 1, 200)
    ]),
    'group': ['A'] * 200 + ['B'] * 200 + ['C'] * 200
})

fig = px.box(df, x='group', y='value',
             title='Grouped Box Plot',
             labels={'group': 'Group', 'value': 'Value'})
fig.show()
```

### Box Plot with Points
```python
import plotly.express as px
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'value': np.random.normal(0, 1, 100),
    'group': np.random.choice(['A', 'B'], 100)
})

fig = px.box(df, x='group', y='value',
             title='Box Plot with Points',
             points='all')
fig.show()
```

## Violin Plots

Violin plots show the full distribution of data.

### Basic Violin Plot
```python
import plotly.express as px
import numpy as np

np.random.seed(42)
data = np.random.normal(0, 1, 1000)

fig = px.violin(y=data,
                title='Basic Violin Plot',
                labels={'y': 'Value'})
fig.show()
```

### Grouped Violin Plot
```python
import plotly.express as px
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'value': np.concatenate([
        np.random.normal(0, 1, 200),
        np.random.normal(2, 1, 200),
        np.random.normal(4, 1, 200)
    ]),
    'group': ['A'] * 200 + ['B'] * 200 + ['C'] * 200
})

fig = px.violin(df, x='group', y='value',
                title='Grouped Violin Plot',
                labels={'group': 'Group', 'value': 'Value'})
fig.show()
```

## Heatmaps

Heatmaps show relationships between two variables using color intensity.

### Basic Heatmap
```python
import plotly.express as px
import numpy as np

# Create correlation matrix
np.random.seed(42)
data = np.random.randn(100, 5)
corr_matrix = np.corrcoef(data.T)

fig = px.imshow(corr_matrix,
                title='Correlation Heatmap',
                color_continuous_scale='RdBu')
fig.show()
```

### Heatmap with Custom Labels
```python
import plotly.express as px
import numpy as np

# Create sample data
data = np.random.rand(5, 5)
row_labels = ['A', 'B', 'C', 'D', 'E']
col_labels = ['X', 'Y', 'Z', 'W', 'V']

fig = px.imshow(data,
                x=col_labels,
                y=row_labels,
                title='Custom Labeled Heatmap',
                color_continuous_scale='Viridis')
fig.show()
```

## Contour Plots

Contour plots show 3D data in 2D using contour lines.

### Basic Contour Plot
```python
import plotly.graph_objects as go
import numpy as np

# Create data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = go.Figure(data=go.Contour(z=Z, x=x, y=y))
fig.update_layout(title='Basic Contour Plot')
fig.show()
```

### Filled Contour Plot
```python
import plotly.graph_objects as go
import numpy as np

x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = go.Figure(data=go.Contour(z=Z, x=x, y=y, 
                                contours_coloring='heatmap'))
fig.update_layout(title='Filled Contour Plot')
fig.show()
```

## 3D Plots

3D plots add an extra dimension to data visualization.

### 3D Scatter Plot
```python
import plotly.express as px
import numpy as np

np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)

fig = px.scatter_3d(x=x, y=y, z=z,
                    title='3D Scatter Plot',
                    labels={'x': 'X', 'y': 'Y', 'z': 'Z'})
fig.show()
```

### 3D Surface Plot
```python
import plotly.graph_objects as go
import numpy as np

x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = go.Figure(data=go.Surface(z=Z, x=x, y=y))
fig.update_layout(title='3D Surface Plot')
fig.show()
```

## Polar Plots

Polar plots show data in polar coordinates.

### Basic Polar Plot
```python
import plotly.graph_objects as go
import numpy as np

theta = np.linspace(0, 2*np.pi, 100)
r = 1 + 0.5*np.sin(3*theta)

fig = go.Figure(data=go.Scatterpolar(r=r, theta=theta, mode='lines'))
fig.update_layout(title='Polar Plot')
fig.show()
```

### Polar Scatter Plot
```python
import plotly.express as px
import numpy as np

np.random.seed(42)
theta = np.random.uniform(0, 2*np.pi, 50)
r = np.random.uniform(0, 5, 50)

fig = px.scatter_polar(r=r, theta=theta,
                       title='Polar Scatter Plot')
fig.show()
```

## Bubble Charts

Bubble charts are scatter plots with a third dimension represented by bubble size.

### Basic Bubble Chart
```python
import plotly.express as px
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.randn(50),
    'y': np.random.randn(50),
    'size': np.random.randint(10, 100, 50),
    'category': np.random.choice(['A', 'B', 'C'], 50)
})

fig = px.scatter(df, x='x', y='y',
                 size='size',
                 color='category',
                 title='Bubble Chart',
                 labels={'x': 'X', 'y': 'Y', 'size': 'Size'})
fig.show()
```

## Funnel Charts

Funnel charts show stages in a process.

### Basic Funnel Chart
```python
import plotly.graph_objects as go

stages = ['Website Visit', 'Product View', 'Add to Cart', 'Purchase']
values = [1000, 800, 600, 400]

fig = go.Figure(go.Funnel(y=stages, x=values))
fig.update_layout(title='Sales Funnel')
fig.show()
```

## Waterfall Charts

Waterfall charts show cumulative effects of positive and negative values.

### Basic Waterfall Chart
```python
import plotly.graph_objects as go

x = ['Sales', 'Consulting', 'Net Revenue', 'Purchases', 'Other Expenses', 'Profit Before Tax']
y = [60, 80, 0, -40, -20, 0]

fig = go.Figure(go.Waterfall(
    name="2020",
    orientation="h",
    measure=["relative", "relative", "total", "relative", "relative", "total"],
    x=y,
    textposition="outside",
    text=y,
    y=x,
    connector={"line": {"color": "rgb(63, 63, 63)"}},
))

fig.update_layout(title="Profit and Loss Statement 2020")
fig.show()
```

## Best Practices

### 1. Choose the Right Plot Type
- **Line plots** for trends over time
- **Scatter plots** for relationships between variables
- **Bar charts** for categorical comparisons
- **Histograms** for data distributions
- **Heatmaps** for correlation matrices

### 2. Color and Styling
```python
# Use consistent color schemes
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

fig = px.scatter(df, x='x', y='y', color='category',
                 color_discrete_sequence=colors)
```

### 3. Labels and Titles
```python
fig.update_layout(
    title='Descriptive Title',
    xaxis_title='Clear X-axis Label',
    yaxis_title='Clear Y-axis Label'
)
```

### 4. Interactive Features
```python
# Add hover information
fig.update_traces(
    hovertemplate='<b>%{x}</b><br>Value: %{y}<extra></extra>'
)
```

### 5. Performance
```python
# For large datasets, use downsampling
if len(df) > 1000:
    df = df.sample(n=1000)
```

## Resources

- [Plotly Express Reference](https://plotly.com/python-api-reference/plotly.express.html)
- [Graph Objects Reference](https://plotly.com/python-api-reference/plotly.graph_objects.html)
- [Plotly Gallery](https://plotly.com/python/plotly-fundamentals/)
- [Plotly Community Forum](https://community.plotly.com/)

---

**Happy Plotting!** 📊✨ 