# Matplotlib Subplots and Layout: Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Basic Subplots](#basic-subplots)
3. [Subplot Grids](#subplot-grids)
4. [GridSpec for Complex Layouts](#gridspec-for-complex-layouts)
5. [Subplot2grid](#subplot2grid)
6. [Figure and Axes Management](#figure-and-axes-management)
7. [Layout Optimization](#layout-optimization)
8. [Advanced Layout Techniques](#advanced-layout-techniques)
9. [Best Practices](#best-practices)

## Introduction

Matplotlib provides powerful tools for creating complex multi-panel visualizations. This guide covers subplot creation, layout management, and advanced techniques for professional multi-panel plots.

## Basic Subplots

### Simple Subplot Creation
```python
import matplotlib.pyplot as plt
import numpy as np

# Create a figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Generate sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = np.exp(-x/3)

# Plot on each subplot
axes[0, 0].plot(x, y1)
axes[0, 0].set_title('sin(x)')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(x, y2)
axes[0, 1].set_title('cos(x)')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(x, y3)
axes[1, 0].set_title('tan(x)')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(x, y4)
axes[1, 1].set_title('exp(-x/3)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Subplot with Different Sizes
```python
# Create subplots with different sizes
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plot on each subplot with different styles
for i in range(2):
    for j in range(3):
        ax = axes[i, j]
        ax.plot(x, y + i*0.5 + j*0.2)
        ax.set_title(f'Subplot ({i}, {j})')
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Individual Subplot Creation
```python
# Create subplots individually
fig = plt.figure(figsize=(12, 8))

# Create subplots at specific positions
ax1 = plt.subplot(2, 2, 1)  # 2 rows, 2 columns, position 1
ax2 = plt.subplot(2, 2, 2)  # position 2
ax3 = plt.subplot(2, 2, 3)  # position 3
ax4 = plt.subplot(2, 2, 4)  # position 4

# Plot data
x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x))
ax1.set_title('Subplot 1')

ax2.plot(x, np.cos(x))
ax2.set_title('Subplot 2')

ax3.plot(x, np.tan(x))
ax3.set_title('Subplot 3')

ax4.plot(x, np.exp(-x/3))
ax4.set_title('Subplot 4')

plt.tight_layout()
plt.show()
```

## Subplot Grids

### Different Grid Configurations
```python
# Various grid configurations
configurations = [
    (1, 2, "1 row, 2 columns"),
    (2, 1, "2 rows, 1 column"),
    (2, 2, "2 rows, 2 columns"),
    (2, 3, "2 rows, 3 columns"),
    (3, 2, "3 rows, 2 columns"),
    (3, 3, "3 rows, 3 columns")
]

for rows, cols, title in configurations:
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Flatten axes if needed
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot on each subplot
    for i, ax in enumerate(axes):
        x = np.linspace(0, 10, 100)
        ax.plot(x, np.sin(x + i*0.5))
        ax.set_title(f'Plot {i+1}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

### Subplot with Shared Axes
```python
# Create subplots with shared x and y axes
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = np.exp(-x/3)

# Plot data
axes[0, 0].plot(x, y1)
axes[0, 0].set_title('sin(x)')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(x, y2)
axes[0, 1].set_title('cos(x)')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(x, y3)
axes[1, 0].set_title('tan(x)')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(x, y4)
axes[1, 1].set_title('exp(-x/3)')
axes[1, 1].grid(True, alpha=0.3)

# Add labels to the entire figure
fig.text(0.5, 0.02, 'x', ha='center', fontsize=12)
fig.text(0.02, 0.5, 'y', va='center', rotation='vertical', fontsize=12)

plt.tight_layout()
plt.show()
```

## GridSpec for Complex Layouts

### Basic GridSpec
```python
from matplotlib.gridspec import GridSpec

# Create figure with GridSpec
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(3, 3, figure=fig)

# Create subplots using GridSpec
ax1 = fig.add_subplot(gs[0, :2])  # Top left, spans 2 columns
ax2 = fig.add_subplot(gs[0, 2])   # Top right
ax3 = fig.add_subplot(gs[1, :])   # Middle, spans all columns
ax4 = fig.add_subplot(gs[2, 0])   # Bottom left
ax5 = fig.add_subplot(gs[2, 1:])  # Bottom right, spans 2 columns

# Generate data
x = np.linspace(0, 10, 100)

# Plot on each subplot
ax1.plot(x, np.sin(x))
ax1.set_title('Top Left (spans 2 columns)')
ax1.grid(True, alpha=0.3)

ax2.plot(x, np.cos(x))
ax2.set_title('Top Right')
ax2.grid(True, alpha=0.3)

ax3.plot(x, np.tan(x))
ax3.set_title('Middle (spans all columns)')
ax3.grid(True, alpha=0.3)

ax4.scatter(np.random.rand(20), np.random.rand(20))
ax4.set_title('Bottom Left')
ax4.grid(True, alpha=0.3)

ax5.hist(np.random.randn(1000), bins=30)
ax5.set_title('Bottom Right (spans 2 columns)')
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### GridSpec with Custom Spacing
```python
# Create GridSpec with custom spacing
fig = plt.figure(figsize=(12, 8))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.4)

# Create subplots
ax1 = fig.add_subplot(gs[0, :2])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1:, :])

# Generate data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plot data
ax1.plot(x, y, 'b-', linewidth=2)
ax1.set_title('Line Plot')
ax1.grid(True, alpha=0.3)

ax2.scatter(np.random.rand(50), np.random.rand(50), alpha=0.6)
ax2.set_title('Scatter Plot')
ax2.grid(True, alpha=0.3)

ax3.hist(np.random.randn(1000), bins=30, alpha=0.7, color='green')
ax3.set_title('Histogram')
ax3.grid(True, alpha=0.3)

plt.suptitle('GridSpec with Custom Spacing', fontsize=16, fontweight='bold')
plt.show()
```

### Complex GridSpec Layout
```python
# Create a complex layout
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

# Define subplot positions
ax1 = fig.add_subplot(gs[0, :2])      # Top left, 2 columns
ax2 = fig.add_subplot(gs[0, 2:])      # Top right, 2 columns
ax3 = fig.add_subplot(gs[1, :])       # Middle, all columns
ax4 = fig.add_subplot(gs[2:, 0])      # Bottom left, 2 rows
ax5 = fig.add_subplot(gs[2:, 1:])     # Bottom right, 2 rows, 3 columns

# Generate data
x = np.linspace(0, 10, 100)
t = np.linspace(0, 4*np.pi, 100)

# Plot 1: Line plot
ax1.plot(x, np.sin(x), 'b-', linewidth=2)
ax1.set_title('Sine Wave')
ax1.grid(True, alpha=0.3)

# Plot 2: Scatter plot
np.random.seed(42)
ax2.scatter(np.random.randn(100), np.random.randn(100), alpha=0.6)
ax2.set_title('Random Scatter')
ax2.grid(True, alpha=0.3)

# Plot 3: Multiple lines
ax3.plot(t, np.sin(t), label='sin(t)', linewidth=2)
ax3.plot(t, np.cos(t), label='cos(t)', linewidth=2)
ax3.set_title('Trigonometric Functions')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Bar chart
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
ax4.bar(categories, values, color='skyblue')
ax4.set_title('Bar Chart')
ax4.set_ylabel('Values')

# Plot 5: Histogram
data = np.random.normal(0, 1, 1000)
ax5.hist(data, bins=30, alpha=0.7, color='lightcoral')
ax5.set_title('Normal Distribution')
ax5.set_xlabel('Value')
ax5.set_ylabel('Frequency')

plt.suptitle('Complex GridSpec Layout', fontsize=16, fontweight='bold')
plt.show()
```

## Subplot2grid

### Basic Subplot2grid
```python
from matplotlib.gridspec import GridSpec

# Create figure
fig = plt.figure(figsize=(12, 8))

# Create GridSpec
gs = GridSpec(3, 3, figure=fig)

# Create subplots using subplot2grid
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)  # Top left, spans 2 columns
ax2 = plt.subplot2grid((3, 3), (0, 2))             # Top right
ax3 = plt.subplot2grid((3, 3), (1, 0), colspan=3)  # Middle, spans all columns
ax4 = plt.subplot2grid((3, 3), (2, 0))             # Bottom left
ax5 = plt.subplot2grid((3, 3), (2, 1), colspan=2)  # Bottom right, spans 2 columns

# Generate data
x = np.linspace(0, 10, 100)

# Plot data
ax1.plot(x, np.sin(x))
ax1.set_title('Top Left (colspan=2)')
ax1.grid(True, alpha=0.3)

ax2.plot(x, np.cos(x))
ax2.set_title('Top Right')
ax2.grid(True, alpha=0.3)

ax3.plot(x, np.tan(x))
ax3.set_title('Middle (colspan=3)')
ax3.grid(True, alpha=0.3)

ax4.scatter(np.random.rand(20), np.random.rand(20))
ax4.set_title('Bottom Left')
ax4.grid(True, alpha=0.3)

ax5.hist(np.random.randn(1000), bins=30)
ax5.set_title('Bottom Right (colspan=2)')
ax5.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Subplot2grid with Rowspan
```python
# Create figure
fig = plt.figure(figsize=(12, 8))

# Create subplots with rowspan
ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2)  # Left, spans 2 rows
ax2 = plt.subplot2grid((3, 3), (0, 1), colspan=2)  # Top right, spans 2 columns
ax3 = plt.subplot2grid((3, 3), (1, 1))             # Middle right
ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)  # Bottom, spans all columns

# Generate data
x = np.linspace(0, 10, 100)

# Plot data
ax1.plot(x, np.sin(x), linewidth=2)
ax1.set_title('Left (rowspan=2)')
ax1.grid(True, alpha=0.3)

ax2.scatter(np.random.randn(50), np.random.randn(50), alpha=0.6)
ax2.set_title('Top Right (colspan=2)')
ax2.grid(True, alpha=0.3)

ax3.plot(x, np.cos(x), linewidth=2)
ax3.set_title('Middle Right')
ax3.grid(True, alpha=0.3)

ax4.hist(np.random.randn(1000), bins=30, alpha=0.7)
ax4.set_title('Bottom (colspan=3)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Figure and Axes Management

### Figure Properties
```python
# Create figure with specific properties
fig = plt.figure(
    figsize=(12, 8),
    dpi=100,
    facecolor='white',
    edgecolor='black',
    linewidth=2
)

# Create subplots
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)

# Plot data
x = np.linspace(0, 10, 100)
ax1.plot(x, np.sin(x))
ax1.set_title('Subplot 1')

ax2.plot(x, np.cos(x))
ax2.set_title('Subplot 2')

ax3.plot(x, np.tan(x))
ax3.set_title('Subplot 3')

ax4.plot(x, np.exp(-x/3))
ax4.set_title('Subplot 4')

# Set figure title
fig.suptitle('Figure with Custom Properties', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.show()
```

### Axes Properties and Methods
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

x = np.linspace(0, 10, 100)
y = np.sin(x)

# Customize each axes
for i, ax in enumerate(axes.flat):
    ax.plot(x, y + i*0.5)
    ax.set_title(f'Axes {i+1}')
    
    # Set different properties for each axes
    if i == 0:
        ax.set_facecolor('lightblue')
        ax.grid(True, alpha=0.5)
    elif i == 1:
        ax.set_xlim(0, 5)
        ax.set_ylim(-1, 2)
    elif i == 2:
        ax.set_aspect('equal')
    else:
        ax.invert_xaxis()
        ax.invert_yaxis()

plt.tight_layout()
plt.show()
```

## Layout Optimization

### Tight Layout
```python
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

x = np.linspace(0, 10, 100)

# Plot on each subplot
for i, ax in enumerate(axes.flat):
    ax.plot(x, np.sin(x + i*0.5))
    ax.set_title(f'Subplot {i+1}')
    ax.grid(True, alpha=0.3)

# Apply tight layout
plt.tight_layout()
plt.show()
```

### Constrained Layout
```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

x = np.linspace(0, 10, 100)

# Plot data
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Subplot 1')

axes[0, 1].plot(x, np.cos(x))
axes[0, 1].set_title('Subplot 2')

axes[1, 0].plot(x, np.tan(x))
axes[1, 0].set_title('Subplot 3')

axes[1, 1].plot(x, np.exp(-x/3))
axes[1, 1].set_title('Subplot 4')

# Add a main title
fig.suptitle('Constrained Layout Example', fontsize=16, fontweight='bold')

plt.show()
```

### Manual Layout Adjustment
```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

x = np.linspace(0, 10, 100)

# Plot data
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Subplot 1')

axes[0, 1].plot(x, np.cos(x))
axes[0, 1].set_title('Subplot 2')

axes[1, 0].plot(x, np.tan(x))
axes[1, 0].set_title('Subplot 3')

axes[1, 1].plot(x, np.exp(-x/3))
axes[1, 1].set_title('Subplot 4')

# Manual layout adjustment
plt.subplots_adjust(
    left=0.1,    # Left margin
    bottom=0.1,  # Bottom margin
    right=0.9,   # Right margin
    top=0.9,     # Top margin
    wspace=0.3,  # Width space between subplots
    hspace=0.3   # Height space between subplots
)

plt.show()
```

## Advanced Layout Techniques

### Nested GridSpec
```python
# Create nested GridSpec
fig = plt.figure(figsize=(15, 10))
outer_gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# Create inner GridSpec for top-left subplot
inner_gs = GridSpec(2, 2, figure=fig, hspace=0.1, wspace=0.1)
inner_gs.update(left=0.05, right=0.48, top=0.95, bottom=0.55)

# Create subplots
ax1a = fig.add_subplot(inner_gs[0, 0])
ax1b = fig.add_subplot(inner_gs[0, 1])
ax1c = fig.add_subplot(inner_gs[1, 0])
ax1d = fig.add_subplot(inner_gs[1, 1])

ax2 = fig.add_subplot(outer_gs[0, 1])
ax3 = fig.add_subplot(outer_gs[1, 0])
ax4 = fig.add_subplot(outer_gs[1, 1])

# Generate data
x = np.linspace(0, 10, 100)

# Plot on nested subplots
ax1a.plot(x, np.sin(x))
ax1a.set_title('1a')

ax1b.plot(x, np.cos(x))
ax1b.set_title('1b')

ax1c.plot(x, np.tan(x))
ax1c.set_title('1c')

ax1d.plot(x, np.exp(-x/3))
ax1d.set_title('1d')

# Plot on main subplots
ax2.scatter(np.random.randn(100), np.random.randn(100), alpha=0.6)
ax2.set_title('Scatter Plot')

ax3.hist(np.random.randn(1000), bins=30, alpha=0.7)
ax3.set_title('Histogram')

ax4.plot(x, np.sin(x), linewidth=2)
ax4.plot(x, np.cos(x), linewidth=2)
ax4.set_title('Multiple Lines')
ax4.legend(['sin(x)', 'cos(x)'])

plt.suptitle('Nested GridSpec Layout', fontsize=16, fontweight='bold')
plt.show()
```

### Dynamic Subplot Creation
```python
def create_dashboard(data_dict, figsize=(15, 10)):
    """Create a dynamic dashboard based on data dictionary."""
    n_plots = len(data_dict)
    cols = int(np.ceil(np.sqrt(n_plots)))
    rows = int(np.ceil(n_plots / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot each dataset
    for i, (title, data) in enumerate(data_dict.items()):
        if i < len(axes):
            ax = axes[i]
            if isinstance(data, dict):
                if data['type'] == 'line':
                    ax.plot(data['x'], data['y'])
                elif data['type'] == 'scatter':
                    ax.scatter(data['x'], data['y'], alpha=0.6)
                elif data['type'] == 'hist':
                    ax.hist(data['values'], bins=30, alpha=0.7)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig, axes

# Example usage
x = np.linspace(0, 10, 100)
data_dict = {
    'Sine Wave': {'type': 'line', 'x': x, 'y': np.sin(x)},
    'Cosine Wave': {'type': 'line', 'x': x, 'y': np.cos(x)},
    'Random Scatter': {'type': 'scatter', 'x': np.random.randn(100), 'y': np.random.randn(100)},
    'Normal Distribution': {'type': 'hist', 'values': np.random.randn(1000)},
    'Exponential Decay': {'type': 'line', 'x': x, 'y': np.exp(-x/3)}
}

fig, axes = create_dashboard(data_dict)
plt.suptitle('Dynamic Dashboard', fontsize=16, fontweight='bold')
plt.show()
```

## Best Practices

1. **Plan your layout**: Sketch out your desired layout before coding
2. **Use appropriate figure sizes**: Consider your publication or display requirements
3. **Maintain consistency**: Use consistent styling across subplots
4. **Optimize spacing**: Use `tight_layout()` or `constrained_layout=True`
5. **Consider readability**: Ensure text and plots are readable at your target size
6. **Test different sizes**: Verify your layout works at different figure sizes
7. **Use descriptive titles**: Make each subplot's purpose clear
8. **Share axes when appropriate**: Use `sharex=True` and `sharey=True` for related plots

## Resources

- [Matplotlib Subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html)
- [GridSpec Documentation](https://matplotlib.org/stable/api/gridspec_api.html)
- [Layout Management](https://matplotlib.org/stable/tutorials/intermediate/tight_layout_guide.html)

---

**Master complex multi-panel visualizations!**