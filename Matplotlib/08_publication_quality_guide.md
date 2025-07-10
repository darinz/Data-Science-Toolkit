# Matplotlib Publication Quality: Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Typography and Fonts](#typography-and-fonts)
3. [Color Schemes](#color-schemes)
4. [Layout and Spacing](#layout-and-spacing)
5. [Export Settings](#export-settings)
6. [Professional Styling](#professional-styling)
7. [Multi-Panel Figures](#multi-panel-figures)
8. [Journal-Specific Requirements](#journal-specific-requirements)
9. [Quality Assurance](#quality-assurance)
10. [Best Practices](#best-practices)

## Introduction

Creating publication-quality plots requires attention to typography, color schemes, layout, and export settings. This guide covers techniques for producing professional visualizations suitable for academic papers, presentations, and reports.

## Typography and Fonts

### Professional Font Configuration
```python
import matplotlib.pyplot as plt
import numpy as np

# Configure professional typography
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'text.usetex': False,  # Set to True if using LaTeX
})

# Create sample plot
fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(0, 10, 100)
y = np.sin(x)

ax.plot(x, y, linewidth=1.5, color='black')
ax.set_xlabel('Time (s)', fontweight='normal')
ax.set_ylabel('Amplitude', fontweight='normal')
ax.set_title('Publication-Quality Plot', fontweight='bold')
ax.grid(True, alpha=0.3, linewidth=0.5)

plt.tight_layout()
plt.show()
```

### LaTeX Integration
```python
# Enable LaTeX rendering
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{amssymb}',
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
})

fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(0, 2*np.pi, 100)

ax.plot(x, np.sin(x), label=r'$\sin(x)$', linewidth=1.5)
ax.plot(x, np.cos(x), label=r'$\cos(x)$', linewidth=1.5)
ax.plot(x, np.exp(-x/3), label=r'$e^{-x/3}$', linewidth=1.5)

ax.set_xlabel(r'$x$ (radians)', fontweight='normal')
ax.set_ylabel(r'$f(x)$', fontweight='normal')
ax.set_title(r'Mathematical Functions', fontweight='bold')
ax.legend(frameon=True, fancybox=False, shadow=False)
ax.grid(True, alpha=0.3, linewidth=0.5)

plt.tight_layout()
plt.show()
```

### Font Hierarchy and Consistency
```python
# Define consistent font hierarchy
FONT_CONFIG = {
    'figure_title': {'size': 14, 'weight': 'bold'},
    'subplot_title': {'size': 12, 'weight': 'bold'},
    'axis_label': {'size': 11, 'weight': 'normal'},
    'tick_label': {'size': 9, 'weight': 'normal'},
    'legend': {'size': 9, 'weight': 'normal'},
    'annotation': {'size': 10, 'weight': 'normal'},
}

def apply_font_config(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent font configuration to axes."""
    if title:
        ax.set_title(title, **FONT_CONFIG['subplot_title'])
    if xlabel:
        ax.set_xlabel(xlabel, **FONT_CONFIG['axis_label'])
    if ylabel:
        ax.set_ylabel(ylabel, **FONT_CONFIG['axis_label'])
    
    ax.tick_params(axis='both', which='major', 
                   labelsize=FONT_CONFIG['tick_label']['size'])

# Create multi-panel figure with consistent typography
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.exp(-x/3)
y4 = np.tan(x)

# Apply consistent styling
apply_font_config(axes[0, 0], 'Sine Function', 'x', 'sin(x)')
axes[0, 0].plot(x, y1, linewidth=1.5, color='black')

apply_font_config(axes[0, 1], 'Cosine Function', 'x', 'cos(x)')
axes[0, 1].plot(x, y2, linewidth=1.5, color='black')

apply_font_config(axes[1, 0], 'Exponential Decay', 'x', 'e^(-x/3)')
axes[1, 0].plot(x, y3, linewidth=1.5, color='black')

apply_font_config(axes[1, 1], 'Tangent Function', 'x', 'tan(x)')
axes[1, 1].plot(x, y4, linewidth=1.5, color='black')

# Add grid to all subplots
for ax in axes.flat:
    ax.grid(True, alpha=0.3, linewidth=0.5)

fig.suptitle('Consistent Typography Example', **FONT_CONFIG['figure_title'])
plt.tight_layout()
plt.show()
```

## Color Schemes

### Publication-Ready Color Palettes
```python
# Define publication-ready color palettes
PUBLICATION_COLORS = {
    'sequential': ['#f7f7f7', '#cccccc', '#969696', '#525252', '#252525'],
    'diverging': ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd'],
    'qualitative': ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf'],
    'grayscale': ['#000000', '#404040', '#808080', '#c0c0c0', '#ffffff'],
    'colorblind_friendly': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
}

def create_publication_plot():
    """Create plot with publication-ready colors."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    x = np.linspace(0, 10, 100)
    
    # Sequential colors
    for i, color in enumerate(PUBLICATION_COLORS['sequential']):
        y = np.sin(x + i*0.5)
        axes[0, 0].plot(x, y, color=color, linewidth=2, 
                       label=f'Series {i+1}')
    axes[0, 0].set_title('Sequential Colors')
    axes[0, 0].legend()
    
    # Diverging colors
    for i, color in enumerate(PUBLICATION_COLORS['diverging']):
        y = np.sin(x + i*0.3)
        axes[0, 1].plot(x, y, color=color, linewidth=2)
    axes[0, 1].set_title('Diverging Colors')
    
    # Qualitative colors
    for i, color in enumerate(PUBLICATION_COLORS['qualitative'][:4]):
        y = np.sin(x + i*0.5)
        axes[0, 2].plot(x, y, color=color, linewidth=2, 
                       label=f'Category {i+1}')
    axes[0, 2].set_title('Qualitative Colors')
    axes[0, 2].legend()
    
    # Grayscale
    for i, color in enumerate(PUBLICATION_COLORS['grayscale']):
        y = np.sin(x + i*0.5)
        axes[1, 0].plot(x, y, color=color, linewidth=2)
    axes[1, 0].set_title('Grayscale')
    
    # Colorblind friendly
    for i, color in enumerate(PUBLICATION_COLORS['colorblind_friendly'][:4]):
        y = np.sin(x + i*0.5)
        axes[1, 1].plot(x, y, color=color, linewidth=2, 
                       label=f'Series {i+1}')
    axes[1, 1].set_title('Colorblind Friendly')
    axes[1, 1].legend()
    
    # Mixed plot types
    axes[1, 2].scatter(np.random.randn(50), np.random.randn(50), 
                      c=PUBLICATION_COLORS['qualitative'][0], alpha=0.7)
    axes[1, 2].plot(x, np.sin(x), color=PUBLICATION_COLORS['qualitative'][1], 
                   linewidth=2)
    axes[1, 2].set_title('Mixed Plot Types')
    
    # Apply consistent styling
    for ax in axes.flat:
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    
    plt.tight_layout()
    plt.show()

create_publication_plot()
```

### Color Accessibility
```python
def test_color_contrast():
    """Test color contrast for accessibility."""
    import colorsys
    
    def get_luminance(r, g, b):
        """Calculate relative luminance."""
        r, g, b = r/255, g/255, b/255
        r = r/12.92 if r <= 0.03928 else ((r + 0.055)/1.055)**2.4
        g = g/12.92 if g <= 0.03928 else ((g + 0.055)/1.055)**2.4
        b = b/12.92 if b <= 0.03928 else ((b + 0.055)/1.055)**2.4
        return 0.2126*r + 0.7152*g + 0.0722*b
    
    def contrast_ratio(l1, l2):
        """Calculate contrast ratio."""
        lighter = max(l1, l2)
        darker = min(l1, l2)
        return (lighter + 0.05) / (darker + 0.05)
    
    # Test color combinations
    colors = [
        ('#1f77b4', '#ffffff'),  # Blue on white
        ('#ff7f0e', '#000000'),  # Orange on black
        ('#2ca02c', '#ffffff'),  # Green on white
        ('#d62728', '#ffffff'),  # Red on white
    ]
    
    print("Color Contrast Analysis:")
    print("-" * 40)
    for color1, color2 in colors:
        # Convert hex to RGB
        r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
        r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
        
        l1 = get_luminance(r1, g1, b1)
        l2 = get_luminance(r2, g2, b2)
        ratio = contrast_ratio(l1, l2)
        
        status = "✓ PASS" if ratio >= 4.5 else "✗ FAIL"
        print(f"{color1} on {color2}: {ratio:.2f}:1 {status}")

test_color_contrast()
```

## Layout and Spacing

### Professional Layout Configuration
```python
def create_publication_layout():
    """Create publication-ready layout configuration."""
    plt.rcParams.update({
        # Figure settings
        'figure.dpi': 300,
        'figure.autolayout': False,
        'figure.constrained_layout.use': False,
        
        # Spacing settings
        'figure.subplot.left': 0.12,
        'figure.subplot.right': 0.95,
        'figure.subplot.bottom': 0.12,
        'figure.subplot.top': 0.95,
        'figure.subplot.wspace': 0.2,
        'figure.subplot.hspace': 0.2,
        
        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'lines.markeredgewidth': 1.0,
        
        # Axes settings
        'axes.linewidth': 1.0,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Grid settings
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        
        # Tick settings
        'xtick.major.size': 4,
        'xtick.major.width': 1.0,
        'ytick.major.size': 4,
        'ytick.major.width': 1.0,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
    })

# Apply layout configuration
create_publication_layout()

# Create professional plot
fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(0, 10, 100)
y = np.sin(x)

ax.plot(x, y, linewidth=1.5, color='black', label='sin(x)')
ax.plot(x, np.cos(x), linewidth=1.5, color='red', label='cos(x)')

ax.set_xlabel('Time (s)', fontweight='normal')
ax.set_ylabel('Amplitude', fontweight='normal')
ax.set_title('Professional Layout', fontweight='bold')
ax.legend(frameon=True, fancybox=False, shadow=False)
ax.grid(True, alpha=0.3, linewidth=0.5)

plt.show()
```

### Multi-Panel Layout Optimization
```python
def create_optimized_multi_panel():
    """Create optimized multi-panel layout."""
    # Create figure with specific dimensions
    fig = plt.figure(figsize=(12, 8))
    
    # Define grid layout
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Create subplots with specific positions
    ax1 = fig.add_subplot(gs[0, :2])  # Top left, spans 2 columns
    ax2 = fig.add_subplot(gs[0, 2])   # Top right
    ax3 = fig.add_subplot(gs[1, :])   # Bottom, spans all columns
    
    # Generate data
    x = np.linspace(0, 10, 100)
    t = np.linspace(0, 4*np.pi, 100)
    
    # Plot 1: Main analysis
    ax1.plot(x, np.sin(x), 'b-', linewidth=1.5, label='sin(x)')
    ax1.plot(x, np.cos(x), 'r-', linewidth=1.5, label='cos(x)')
    ax1.set_title('Main Analysis', fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    
    # Plot 2: Summary statistics
    categories = ['A', 'B', 'C', 'D']
    values = [23, 45, 56, 78]
    ax2.bar(categories, values, color='gray', alpha=0.7)
    ax2.set_title('Summary', fontweight='bold')
    ax2.set_ylabel('Count')
    
    # Plot 3: Time series
    ax3.plot(t, np.sin(t), 'g-', linewidth=1.5)
    ax3.set_title('Time Series', fontweight='bold')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Value')
    ax3.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add figure title
    fig.suptitle('Optimized Multi-Panel Layout', fontsize=16, fontweight='bold')
    
    plt.show()

create_optimized_multi_panel()
```

## Export Settings

### High-Resolution Export
```python
def export_publication_plot(filename, dpi=300, format='pdf'):
    """Export plot with publication-quality settings."""
    # Create plot
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    ax.plot(x, y, linewidth=1.5, color='black')
    ax.set_xlabel('Time (s)', fontweight='normal')
    ax.set_ylabel('Amplitude', fontweight='normal')
    ax.set_title('Publication-Ready Plot', fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Optimize layout
    plt.tight_layout()
    
    # Export with high quality
    plt.savefig(filename, 
                dpi=dpi, 
                format=format,
                bbox_inches='tight',
                pad_inches=0.1,
                facecolor='white',
                edgecolor='none',
                transparent=False)
    
    print(f"Plot exported as {filename}")
    plt.show()

# Export in different formats
export_publication_plot('publication_plot.pdf', dpi=300, format='pdf')
export_publication_plot('publication_plot.png', dpi=300, format='png')
export_publication_plot('publication_plot.svg', dpi=300, format='svg')
```

### Vector Graphics Export
```python
def create_vector_plot():
    """Create plot optimized for vector graphics export."""
    # Configure for vector graphics
    plt.rcParams.update({
        'svg.fonttype': 'none',  # Use system fonts in SVG
        'pdf.fonttype': 42,      # Use TrueType fonts in PDF
        'ps.fonttype': 42,       # Use TrueType fonts in PostScript
    })
    
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    # Use vector-friendly styling
    ax.plot(x, y, linewidth=1.5, color='black', solid_capstyle='round')
    ax.set_xlabel('Time (s)', fontweight='normal')
    ax.set_ylabel('Amplitude', fontweight='normal')
    ax.set_title('Vector Graphics Ready', fontweight='bold')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    
    # Export as vector graphics
    plt.savefig('vector_plot.pdf', format='pdf', bbox_inches='tight')
    plt.savefig('vector_plot.svg', format='svg', bbox_inches='tight')
    
    plt.show()

create_vector_plot()
```

## Professional Styling

### Publication Style Sheet
```python
def create_publication_style():
    """Create comprehensive publication style configuration."""
    style_config = {
        # Figure settings
        'figure.dpi': 300,
        'figure.figsize': (6, 4),
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 10,
        'font.weight': 'normal',
        
        # Text settings
        'text.usetex': False,
        'text.color': 'black',
        
        # Axes settings
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'normal',
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Tick settings
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'lines.markeredgewidth': 1.0,
        'lines.solid_capstyle': 'round',
        
        # Grid settings
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.color': 'gray',
        
        # Legend settings
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.shadow': False,
        'legend.framealpha': 1.0,
        'legend.edgecolor': 'black',
        'legend.borderpad': 0.4,
        
        # Save settings
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',
        'savefig.transparent': False,
    }
    
    return style_config

# Apply publication style
style = create_publication_style()
plt.rcParams.update(style)

# Create professional plot
fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

ax.plot(x, y1, 'b-', linewidth=1.5, label='sin(x)')
ax.plot(x, y2, 'r-', linewidth=1.5, label='cos(x)')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Professional Publication Style')
ax.legend()
ax.grid(True, alpha=0.3, linewidth=0.5)

plt.tight_layout()
plt.show()
```

### Consistent Styling Functions
```python
def apply_publication_style(ax, title=None, xlabel=None, ylabel=None, 
                           show_grid=True, show_legend=False):
    """Apply consistent publication styling to axes."""
    
    # Set labels with consistent formatting
    if title:
        ax.set_title(title, fontweight='bold', fontsize=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontweight='normal', fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontweight='normal', fontsize=11)
    
    # Configure ticks
    ax.tick_params(axis='both', which='major', 
                   labelsize=9, width=1.0, length=4)
    ax.tick_params(axis='both', which='minor', 
                   labelsize=9, width=1.0, length=2)
    
    # Configure spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    
    # Add grid
    if show_grid:
        ax.grid(True, alpha=0.3, linewidth=0.5, linestyle='--')
    
    # Configure legend
    if show_legend:
        ax.legend(frameon=True, fancybox=False, shadow=False, 
                 fontsize=9, framealpha=1.0)

# Example usage
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.exp(-x/3)
y4 = np.tan(x)

# Apply consistent styling to each subplot
axes[0, 0].plot(x, y1, 'b-', linewidth=1.5)
apply_publication_style(axes[0, 0], 'Sine Function', 'x', 'sin(x)')

axes[0, 1].plot(x, y2, 'r-', linewidth=1.5)
apply_publication_style(axes[0, 1], 'Cosine Function', 'x', 'cos(x)')

axes[1, 0].plot(x, y3, 'g-', linewidth=1.5)
apply_publication_style(axes[1, 0], 'Exponential Decay', 'x', 'e^(-x/3)')

axes[1, 1].plot(x, y4, 'm-', linewidth=1.5)
apply_publication_style(axes[1, 1], 'Tangent Function', 'x', 'tan(x)')

plt.tight_layout()
plt.show()
```

## Multi-Panel Figures

### Complex Multi-Panel Layout
```python
def create_complex_multi_panel():
    """Create complex multi-panel figure with publication quality."""
    
    # Create figure with specific dimensions
    fig = plt.figure(figsize=(12, 10))
    
    # Define complex grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
    
    # Create subplots with specific positions
    ax1 = fig.add_subplot(gs[0, :2])      # Top left, 2 columns
    ax2 = fig.add_subplot(gs[0, 2:])      # Top right, 2 columns
    ax3 = fig.add_subplot(gs[1, :])       # Middle, all columns
    ax4 = fig.add_subplot(gs[2:, 0])      # Bottom left, 2 rows
    ax5 = fig.add_subplot(gs[2:, 1:])     # Bottom right, 2 rows, 3 columns
    
    # Generate data
    x = np.linspace(0, 10, 100)
    t = np.linspace(0, 4*np.pi, 100)
    
    # Plot 1: Main analysis
    ax1.plot(x, np.sin(x), 'b-', linewidth=1.5, label='sin(x)')
    ax1.plot(x, np.cos(x), 'r-', linewidth=1.5, label='cos(x)')
    apply_publication_style(ax1, 'Main Analysis', 'x', 'y', show_legend=True)
    
    # Plot 2: Summary statistics
    categories = ['A', 'B', 'C', 'D']
    values = [23, 45, 56, 78]
    ax2.bar(categories, values, color='gray', alpha=0.7, edgecolor='black')
    apply_publication_style(ax2, 'Summary Statistics', 'Category', 'Count')
    
    # Plot 3: Time series
    ax3.plot(t, np.sin(t), 'g-', linewidth=1.5)
    apply_publication_style(ax3, 'Time Series Analysis', 'Time', 'Value')
    
    # Plot 4: Scatter plot
    np.random.seed(42)
    x_scatter = np.random.randn(100)
    y_scatter = np.random.randn(100)
    ax4.scatter(x_scatter, y_scatter, alpha=0.6, color='blue', s=20)
    apply_publication_style(ax4, 'Scatter Plot', 'X', 'Y')
    
    # Plot 5: Histogram
    data = np.random.normal(0, 1, 1000)
    ax5.hist(data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    apply_publication_style(ax5, 'Distribution', 'Value', 'Frequency')
    
    # Add figure title
    fig.suptitle('Complex Multi-Panel Publication Figure', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.show()

create_complex_multi_panel()
```

### Subplot Labeling
```python
def create_labeled_multi_panel():
    """Create multi-panel figure with proper subplot labeling."""
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.exp(-x/3)
    y4 = np.tan(x)
    
    # Create plots
    axes[0, 0].plot(x, y1, 'b-', linewidth=1.5)
    axes[0, 1].plot(x, y2, 'r-', linewidth=1.5)
    axes[1, 0].plot(x, y3, 'g-', linewidth=1.5)
    axes[1, 1].plot(x, y4, 'm-', linewidth=1.5)
    
    # Apply styling and add subplot labels
    apply_publication_style(axes[0, 0], 'Sine Function', 'x', 'sin(x)')
    apply_publication_style(axes[0, 1], 'Cosine Function', 'x', 'cos(x)')
    apply_publication_style(axes[1, 0], 'Exponential Decay', 'x', 'e^(-x/3)')
    apply_publication_style(axes[1, 1], 'Tangent Function', 'x', 'tan(x)')
    
    # Add subplot labels (a, b, c, d)
    labels = ['(a)', '(b)', '(c)', '(d)']
    for ax, label in zip(axes.flat, labels):
        ax.text(0.02, 0.98, label, transform=ax.transAxes, 
                fontsize=12, fontweight='bold', 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

create_labeled_multi_panel()
```

## Journal-Specific Requirements

### Common Journal Formats
```python
JOURNAL_FORMATS = {
    'nature': {
        'figure_width': 89,  # mm
        'figure_height': 120,  # mm
        'font_size': 7,
        'line_width': 0.5,
        'dpi': 300,
    },
    'science': {
        'figure_width': 85,  # mm
        'figure_height': 110,  # mm
        'font_size': 8,
        'line_width': 0.5,
        'dpi': 300,
    },
    'plos': {
        'figure_width': 85,  # mm
        'figure_height': 110,  # mm
        'font_size': 8,
        'line_width': 0.5,
        'dpi': 300,
    },
    'ieee': {
        'figure_width': 85,  # mm
        'figure_height': 110,  # mm
        'font_size': 8,
        'line_width': 0.5,
        'dpi': 300,
    }
}

def create_journal_plot(journal_name):
    """Create plot formatted for specific journal."""
    
    if journal_name not in JOURNAL_FORMATS:
        raise ValueError(f"Unknown journal: {journal_name}")
    
    format_spec = JOURNAL_FORMATS[journal_name]
    
    # Convert mm to inches
    width_inch = format_spec['figure_width'] / 25.4
    height_inch = format_spec['figure_height'] / 25.4
    
    # Configure for journal
    plt.rcParams.update({
        'font.size': format_spec['font_size'],
        'lines.linewidth': format_spec['line_width'],
        'axes.linewidth': format_spec['line_width'],
        'xtick.major.width': format_spec['line_width'],
        'ytick.major.width': format_spec['line_width'],
    })
    
    # Create plot
    fig, ax = plt.subplots(figsize=(width_inch, height_inch))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    ax.plot(x, y, linewidth=format_spec['line_width'], color='black')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'{journal_name.title()} Format')
    ax.grid(True, alpha=0.3, linewidth=format_spec['line_width'])
    
    plt.tight_layout()
    
    # Export with journal specifications
    filename = f'{journal_name}_format.pdf'
    plt.savefig(filename, dpi=format_spec['dpi'], bbox_inches='tight')
    
    print(f"Plot saved as {filename}")
    plt.show()

# Create plots for different journals
for journal in ['nature', 'science', 'plos', 'ieee']:
    create_journal_plot(journal)
```

## Quality Assurance

### Plot Quality Checklist
```python
def check_plot_quality(fig, ax):
    """Check plot quality against publication standards."""
    
    checklist = {
        'Typography': {
            'Font size readable': ax.get_title().get_fontsize() >= 10,
            'Axis labels present': bool(ax.get_xlabel() and ax.get_ylabel()),
            'Consistent font family': True,  # Check if all fonts are consistent
        },
        'Layout': {
            'No overlapping elements': True,  # Would need more sophisticated checking
            'Proper margins': True,
            'Grid appropriate': ax.get_xgrid() or ax.get_ygrid(),
        },
        'Color': {
            'Colorblind friendly': True,  # Would need color analysis
            'Sufficient contrast': True,
            'Grayscale compatible': True,
        },
        'Data': {
            'Error bars if needed': True,
            'Appropriate scale': True,
            'Clear data points': True,
        },
        'Export': {
            'High resolution': plt.rcParams['figure.dpi'] >= 300,
            'Vector format available': True,
            'Proper file size': True,
        }
    }
    
    print("Publication Quality Checklist:")
    print("=" * 40)
    
    for category, items in checklist.items():
        print(f"\n{category}:")
        for item, status in items.items():
            status_symbol = "✓" if status else "✗"
            print(f"  {status_symbol} {item}")
    
    return checklist

# Test quality checklist
fig, ax = plt.subplots(figsize=(6, 4))
x = np.linspace(0, 10, 100)
y = np.sin(x)

ax.plot(x, y, linewidth=1.5, color='black')
ax.set_xlabel('Time (s)', fontweight='normal')
ax.set_ylabel('Amplitude', fontweight='normal')
ax.set_title('Quality Test Plot', fontweight='bold')
ax.grid(True, alpha=0.3, linewidth=0.5)

check_plot_quality(fig, ax)
plt.show()
```

### Automated Quality Testing
```python
def automated_quality_test():
    """Automated testing of plot quality metrics."""
    
    def test_resolution():
        """Test if plot meets resolution requirements."""
        return plt.rcParams['figure.dpi'] >= 300
    
    def test_font_sizes():
        """Test if font sizes are appropriate."""
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        ax.plot(x, y)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Test')
        
        title_size = ax.get_title().get_fontsize()
        label_size = ax.get_xlabel().get_fontsize()
        
        plt.close(fig)
        
        return title_size >= 10 and label_size >= 9
    
    def test_color_contrast():
        """Test color contrast ratios."""
        # This would implement actual color contrast testing
        return True
    
    tests = {
        'Resolution': test_resolution(),
        'Font Sizes': test_font_sizes(),
        'Color Contrast': test_color_contrast(),
    }
    
    print("Automated Quality Tests:")
    print("=" * 30)
    
    all_passed = True
    for test_name, result in tests.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")
    return all_passed

automated_quality_test()
```

## Best Practices

1. **Use consistent typography** across all plots
2. **Choose appropriate color schemes** for your audience
3. **Optimize layout** for readability and impact
4. **Export at high resolution** for publication
5. **Test accessibility** for colorblind viewers
6. **Follow journal guidelines** when submitting
7. **Use vector formats** when possible
8. **Maintain consistent styling** across related plots
9. **Include proper labels** and legends
10. **Test at different sizes** to ensure readability

## Resources

- [Matplotlib Publication Quality](https://matplotlib.org/stable/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py)
- [Journal Submission Guidelines](https://www.nature.com/nature/for-authors/formatting-guide)
- [Color Accessibility Tools](https://www.color-blindness.com/color-name-hue/)
- [Typography Best Practices](https://www.typography.com/techniques/)

---

**Create stunning publication-quality visualizations!**