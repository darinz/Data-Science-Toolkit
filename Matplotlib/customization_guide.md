# Matplotlib Customization: Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Colors and Color Maps](#colors-and-color-maps)
3. [Styles and Themes](#styles-and-themes)
4. [Fonts and Typography](#fonts-and-typography)
5. [Annotations and Text](#annotations-and-text)
6. [Legends](#legends)
7. [Grids and Axes](#grids-and-axes)
8. [Backgrounds and Layout](#backgrounds-and-layout)
9. [Advanced Customization](#advanced-customization)
10. [Best Practices](#best-practices)

## Introduction

Matplotlib provides extensive customization options to create professional, publication-ready visualizations. This guide covers color schemes, styles, fonts, annotations, and advanced styling techniques.

## Colors and Color Maps

### Named Colors
```python
import matplotlib.pyplot as plt
import numpy as np

# Basic named colors
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
x = np.linspace(0, 2*np.pi, 100)

for i, color in enumerate(colors):
    plt.plot(x, np.sin(x + i*0.5), color=color, label=color, linewidth=2)

plt.title('Named Colors')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### RGB and Hex Colors
```python
# RGB colors (0-1 scale)
rgb_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]

# Hex colors
hex_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

x = np.linspace(0, 10, 100)
for i, (rgb, hex_color) in enumerate(zip(rgb_colors, hex_colors)):
    plt.plot(x, np.sin(x + i), color=hex_color, linewidth=2, 
             label=f'Color {i+1}')

plt.title('RGB and Hex Colors')
plt.legend()
plt.show()
```

### Color Maps
```python
# Sequential colormaps
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo']

for ax, cmap in zip(axes.flat, colormaps):
    data = np.random.rand(10, 10)
    im = ax.imshow(data, cmap=cmap)
    ax.set_title(cmap)
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.show()
```

### Diverging and Qualitative Colormaps
```python
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Diverging colormaps
diverging = ['RdBu', 'RdYlBu', 'coolwarm']
for i, cmap in enumerate(diverging):
    data = np.random.randn(10, 10)
    im = axes[0, i].imshow(data, cmap=cmap)
    axes[0, i].set_title(f'Diverging: {cmap}')
    plt.colorbar(im, ax=axes[0, i])

# Qualitative colormaps
qualitative = ['Set1', 'Set2', 'tab10']
for i, cmap in enumerate(qualitative):
    data = np.random.randint(0, 8, (10, 10))
    im = axes[1, i].imshow(data, cmap=cmap)
    axes[1, i].set_title(f'Qualitative: {cmap}')
    plt.colorbar(im, ax=axes[1, i])

plt.tight_layout()
plt.show()
```

### Custom Color Maps
```python
from matplotlib.colors import LinearSegmentedColormap

# Create custom colormap
colors = ['darkblue', 'blue', 'lightblue', 'white', 'lightcoral', 'red', 'darkred']
n_bins = 100
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)

# Use custom colormap
data = np.random.randn(20, 20)
plt.imshow(data, cmap=custom_cmap)
plt.colorbar(label='Value')
plt.title('Custom Color Map')
plt.show()
```

## Styles and Themes

### Built-in Styles
```python
# Available styles
print(plt.style.available)

# Apply different styles
styles = ['default', 'classic', 'bmh', 'ggplot', 'fivethirtyeight']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
x = np.linspace(0, 10, 100)

for i, style in enumerate(styles):
    plt.style.use(style)
    ax = axes[i//3, i%3]
    ax.plot(x, np.sin(x), label='sin(x)')
    ax.plot(x, np.cos(x), label='cos(x)')
    ax.set_title(f'Style: {style}')
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
```

### Custom Style Parameters
```python
# Set custom style parameters
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
})

x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x), 'o-', label='sin(x)')
plt.plot(x, np.cos(x), 's-', label='cos(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Custom Style Parameters')
plt.legend()
plt.show()
```

### Context Managers for Styles
```python
with plt.style.context('ggplot'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    x = np.linspace(0, 10, 100)
    ax1.plot(x, np.sin(x))
    ax1.set_title('ggplot style')
    
    ax2.plot(x, np.cos(x))
    ax2.set_title('ggplot style')

plt.tight_layout()
plt.show()
```

## Fonts and Typography

### Font Families
```python
# Available font families
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm

# List available fonts
fonts = [f.name for f in fm.fontManager.ttflist]
print("Available fonts:", len(fonts))
print("Sample fonts:", fonts[:10])

# Use different font families
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
font_families = ['serif', 'sans-serif', 'monospace', 'cursive']

for ax, family in zip(axes.flat, font_families):
    ax.plot([1, 2, 3], [1, 4, 2])
    ax.set_title(f'Font Family: {family}', fontfamily=family, fontsize=14)
    ax.set_xlabel('X Axis', fontfamily=family)
    ax.set_ylabel('Y Axis', fontfamily=family)

plt.tight_layout()
plt.show()
```

### Font Properties
```python
# Custom font properties
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), linewidth=3)

# Title with custom font
ax.set_title('Custom Typography Example', 
             fontsize=20, fontweight='bold', fontfamily='serif',
             color='darkblue', pad=20)

# Axis labels with custom font
ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='semibold',
              fontfamily='sans-serif', color='darkred')
ax.set_ylabel('Amplitude', fontsize=14, fontweight='semibold',
              fontfamily='sans-serif', color='darkred')

# Tick labels
ax.tick_params(axis='both', labelsize=12, labelcolor='darkgreen')

plt.show()
```

### Mathematical Text
```python
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 4*np.pi, 100)
ax.plot(x, np.sin(x), label=r'$\sin(x)$')
ax.plot(x, np.cos(x), label=r'$\cos(x)$')
ax.plot(x, np.exp(-x/5), label=r'$e^{-x/5}$')

ax.set_title(r'Mathematical Functions: $\sin(x)$, $\cos(x)$, $e^{-x/5}$', 
             fontsize=16, fontweight='bold')
ax.set_xlabel(r'$x$ (radians)', fontsize=14)
ax.set_ylabel(r'$f(x)$', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.show()
```

## Annotations and Text

### Basic Annotations
```python
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y)

# Point annotation
ax.annotate('Maximum', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 1.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, fontweight='bold')

# Text annotation
ax.text(2, 0.5, 'This is a text annotation', fontsize=12, 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

ax.set_title('Annotations Example')
ax.grid(True, alpha=0.3)
plt.show()
```

### Advanced Annotations
```python
fig, ax = plt.subplots(figsize=(12, 8))

# Create some data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

ax.plot(x, y1, 'b-', linewidth=2, label='sin(x)')
ax.plot(x, y2, 'r-', linewidth=2, label='cos(x)')

# Multiple annotations
annotations = [
    {'pos': (np.pi/2, 1), 'text': 'sin(π/2) = 1', 'color': 'blue'},
    {'pos': (0, 1), 'text': 'cos(0) = 1', 'color': 'red'},
    {'pos': (np.pi, -1), 'text': 'cos(π) = -1', 'color': 'red'},
]

for ann in annotations:
    ax.annotate(ann['text'], xy=ann['pos'], xytext=(ann['pos'][0] + 1, ann['pos'][1] + 0.5),
                arrowprops=dict(arrowstyle='->', color=ann['color'], lw=2),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax.set_title('Advanced Annotations', fontsize=16, fontweight='bold')
ax.set_xlabel('x (radians)', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.show()
```

### Text with Special Formatting
```python
fig, ax = plt.subplots(figsize=(10, 6))

# Sample data
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]

bars = ax.bar(categories, values, color='skyblue', edgecolor='black')

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{value}', ha='center', va='bottom', fontweight='bold')

ax.set_title('Bar Chart with Value Labels', fontsize=14, fontweight='bold')
ax.set_xlabel('Categories', fontsize=12)
ax.set_ylabel('Values', fontsize=12)

plt.show()
```

## Legends

### Basic Legend
```python
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin(x)', linewidth=2)
ax.plot(x, np.cos(x), label='cos(x)', linewidth=2)
ax.plot(x, np.tan(x), label='tan(x)', linewidth=2)

ax.legend()
ax.set_title('Basic Legend')
ax.grid(True, alpha=0.3)
plt.show()
```

### Customized Legend
```python
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
line1, = ax.plot(x, np.sin(x), 'b-', linewidth=2, label='sin(x)')
line2, = ax.plot(x, np.cos(x), 'r--', linewidth=2, label='cos(x)')
line3, = ax.plot(x, np.exp(-x/3), 'g:', linewidth=2, label='exp(-x/3)')

# Custom legend
legend = ax.legend(loc='upper right', fontsize=12, frameon=True,
                   fancybox=True, shadow=True, framealpha=0.8)
legend.get_frame().set_facecolor('lightgray')

ax.set_title('Customized Legend', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.show()
```

### Legend Outside Plot
```python
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), label='sin(x)', linewidth=2)
ax.plot(x, np.cos(x), label='cos(x)', linewidth=2)
ax.plot(x, np.tan(x), label='tan(x)', linewidth=2)

# Legend outside the plot
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

ax.set_title('Legend Outside Plot')
ax.grid(True, alpha=0.3)

# Adjust layout to prevent legend cutoff
plt.tight_layout()
plt.show()
```

## Grids and Axes

### Grid Customization
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

x = np.linspace(0, 10, 100)
y = np.sin(x)

# No grid
axes[0, 0].plot(x, y)
axes[0, 0].set_title('No Grid')

# Basic grid
axes[0, 1].plot(x, y)
axes[0, 1].grid(True)
axes[0, 1].set_title('Basic Grid')

# Custom grid
axes[1, 0].plot(x, y)
axes[1, 0].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
axes[1, 0].set_title('Custom Grid')

# Major and minor grid
axes[1, 1].plot(x, y)
axes[1, 1].grid(True, which='major', alpha=0.5, color='red')
axes[1, 1].grid(True, which='minor', alpha=0.2, color='blue')
axes[1, 1].minorticks_on()
axes[1, 1].set_title('Major/Minor Grid')

plt.tight_layout()
plt.show()
```

### Axis Customization
```python
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y, linewidth=2)

# Customize axes
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax.set_ylabel('Amplitude', fontsize=12, fontweight='bold')

# Custom tick marks
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_xticklabels(['0s', '2s', '4s', '6s', '8s', '10s'], rotation=45)
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0'])

# Customize tick appearance
ax.tick_params(axis='both', which='major', labelsize=10, width=2, length=6)
ax.tick_params(axis='both', which='minor', width=1, length=3)

ax.set_title('Customized Axes', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.show()
```

### Twin Axes
```python
fig, ax1 = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.exp(-x/3)

# Primary axis
color = 'tab:blue'
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('sin(x)', color=color, fontsize=12)
line1 = ax1.plot(x, y1, color=color, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color)

# Secondary axis
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('exp(-x/3)', color=color, fontsize=12)
line2 = ax2.plot(x, y2, color=color, linewidth=2)
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_title('Twin Axes Example', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

plt.show()
```

## Backgrounds and Layout

### Figure Background
```python
fig, ax = plt.subplots(figsize=(10, 6))

# Set figure background
fig.patch.set_facecolor('lightgray')

x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y, linewidth=3, color='darkblue')

# Set axes background
ax.set_facecolor('white')

ax.set_title('Custom Backgrounds', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.show()
```

### Subplot Layout
```python
# Create figure with custom layout
fig = plt.figure(figsize=(12, 8))

# Define grid layout
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Create subplots
ax1 = fig.add_subplot(gs[0, :2])  # Top left, spans 2 columns
ax2 = fig.add_subplot(gs[0, 2])   # Top right
ax3 = fig.add_subplot(gs[1, :])   # Middle, spans all columns
ax4 = fig.add_subplot(gs[2, 0])   # Bottom left
ax5 = fig.add_subplot(gs[2, 1:])  # Bottom right, spans 2 columns

# Add content to each subplot
x = np.linspace(0, 10, 100)

ax1.plot(x, np.sin(x))
ax1.set_title('Subplot 1')

ax2.plot(x, np.cos(x))
ax2.set_title('Subplot 2')

ax3.plot(x, np.tan(x))
ax3.set_title('Subplot 3')

ax4.scatter(np.random.rand(20), np.random.rand(20))
ax4.set_title('Subplot 4')

ax5.hist(np.random.randn(1000), bins=30)
ax5.set_title('Subplot 5')

plt.suptitle('Custom Subplot Layout', fontsize=16, fontweight='bold')
plt.show()
```

## Advanced Customization

### Custom Plot Styles
```python
# Define custom style
def custom_style():
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'axes.edgecolor': '#dee2e6',
        'axes.linewidth': 1.5,
        'axes.grid': True,
        'grid.color': '#e9ecef',
        'grid.alpha': 0.8,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'font.size': 11,
        'font.family': 'sans-serif',
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
    })

# Apply custom style
custom_style()

fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
ax.plot(x, np.sin(x), 'o-', label='sin(x)', markersize=6)
ax.plot(x, np.cos(x), 's-', label='cos(x)', markersize=6)

ax.set_title('Custom Style Example', fontsize=14, fontweight='bold')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.legend(fontsize=11)

plt.show()
```

### Animated Customization
```python
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)
ax.grid(True, alpha=0.3)

line, = ax.plot([], [], 'b-', linewidth=2)
ax.set_title('Animated Sine Wave', fontsize=14, fontweight='bold')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('sin(x)', fontsize=12)

def animate(frame):
    x = np.linspace(0, 10, 100)
    y = np.sin(x + frame * 0.1)
    line.set_data(x, y)
    return line,

ani = FuncAnimation(fig, animate, frames=100, interval=50, blit=True)
plt.show()
```

### Publication Quality Plots
```python
# Publication quality settings
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
})

fig, ax = plt.subplots(figsize=(8, 6))

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

ax.plot(x, y1, 'b-', label='sin(x)', linewidth=1.5)
ax.plot(x, y2, 'r--', label='cos(x)', linewidth=1.5)

ax.set_xlabel('x (radians)', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title('Publication Quality Plot', fontsize=14, fontweight='bold')
ax.legend(frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3)

# Save with high quality
plt.savefig('publication_quality.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Best Practices

1. **Consistency**: Use consistent colors, fonts, and styles across related plots
2. **Accessibility**: Choose color combinations that work for colorblind viewers
3. **Clarity**: Ensure all elements are clearly labeled and readable
4. **Simplicity**: Avoid unnecessary decorative elements
5. **Context**: Consider your audience and publication requirements
6. **Testing**: Test your plots at different sizes and resolutions

## Resources

- [Matplotlib Customization](https://matplotlib.org/stable/tutorials/introductory/customizing.html)
- [Matplotlib Style Sheets](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html)
- [Color Maps in Matplotlib](https://matplotlib.org/stable/tutorials/colors/colormaps.html)

---

**Create stunning, professional visualizations!** 🎨 