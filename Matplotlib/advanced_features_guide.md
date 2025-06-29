# Matplotlib Advanced Features: Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Event Handling](#event-handling)
3. [Custom Projections](#custom-projections)
4. [Backend Management](#backend-management)
5. [Performance Optimization](#performance-optimization)
6. [Advanced Customization](#advanced-customization)
7. [Interactive Features](#interactive-features)
8. [Custom Artists](#custom-artists)
9. [Advanced Layout](#advanced-layout)
10. [Best Practices](#best-practices)

## Introduction

Matplotlib's advanced features enable sophisticated customizations, interactive visualizations, and high-performance plotting. This guide covers event handling, custom projections, backend management, performance optimization, and advanced techniques.

## Event Handling

### Basic Event Handling
```python
import matplotlib.pyplot as plt
import numpy as np

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
line, = ax.plot(x, y)

# Event handler for mouse clicks
def on_click(event):
    if event.inaxes == ax:
        print(f'Mouse clicked at ({event.xdata:.2f}, {event.ydata:.2f})')
        # Add a marker at click location
        ax.plot(event.xdata, event.ydata, 'ro', markersize=10)
        fig.canvas.draw()

# Connect event handler
fig.canvas.mpl_connect('button_press_event', on_click)

ax.set_title('Click anywhere on the plot!')
ax.grid(True, alpha=0.3)
plt.show()
```

### Keyboard Event Handling
```python
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
line, = ax.plot(x, y, linewidth=2)

# Global variables to track state
current_phase = 0
current_amplitude = 1

def on_key(event):
    global current_phase, current_amplitude
    
    if event.key == 'left':
        current_phase -= 0.5
    elif event.key == 'right':
        current_phase += 0.5
    elif event.key == 'up':
        current_amplitude += 0.1
    elif event.key == 'down':
        current_amplitude -= 0.1
    
    # Update the plot
    y_new = current_amplitude * np.sin(x + current_phase)
    line.set_ydata(y_new)
    ax.set_title(f'Amplitude: {current_amplitude:.1f}, Phase: {current_phase:.1f}')
    fig.canvas.draw()

# Connect keyboard event
fig.canvas.mpl_connect('key_press_event', on_key)

ax.set_title('Use arrow keys to adjust amplitude and phase')
ax.grid(True, alpha=0.3)
plt.show()
```

### Mouse Motion Events
```python
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
line, = ax.plot(x, y, linewidth=2)

# Create text annotation
text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def on_mouse_move(event):
    if event.inaxes == ax:
        # Update text with current mouse position
        text.set_text(f'Mouse: ({event.xdata:.2f}, {event.ydata:.2f})')
        fig.canvas.draw_idle()

# Connect mouse motion event
fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

ax.set_title('Move mouse over the plot')
ax.grid(True, alpha=0.3)
plt.show()
```

### Custom Event Classes
```python
from matplotlib.backend_bases import MouseButton

class PlotInteractor:
    def __init__(self, ax):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.points = []
        self.lines = []
        
        # Connect events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('key_press_event', self.on_key)
        
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        if event.button == MouseButton.LEFT:
            # Add point
            point, = self.ax.plot(event.xdata, event.ydata, 'ro', markersize=8)
            self.points.append((event.xdata, event.ydata, point))
            
            # Connect points with lines
            if len(self.points) > 1:
                prev_x, prev_y, _ = self.points[-2]
                line, = self.ax.plot([prev_x, event.xdata], [prev_y, event.ydata], 'b-')
                self.lines.append(line)
                
        elif event.button == MouseButton.RIGHT:
            # Remove last point
            if self.points:
                x, y, point = self.points.pop()
                point.remove()
                if self.lines:
                    line = self.lines.pop()
                    line.remove()
        
        self.canvas.draw()
    
    def on_key(self, event):
        if event.key == 'c':
            # Clear all points
            for x, y, point in self.points:
                point.remove()
            for line in self.lines:
                line.remove()
            self.points.clear()
            self.lines.clear()
            self.canvas.draw()

# Create interactive plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(-2, 2)
ax.grid(True, alpha=0.3)
ax.set_title('Left click: Add point, Right click: Remove point, C: Clear all')

interactor = PlotInteractor(ax)
plt.show()
```

## Custom Projections

### Basic Custom Projection
```python
import matplotlib.projections as proj
import matplotlib.axes as maxes
from matplotlib.transforms import Affine2D

class PolarProjection(proj.ProjectionBase):
    name = 'custom_polar'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _set_lim_and_transforms(self):
        # Set up transforms
        self.transData = Affine2D().scale(1, 1)
        self.transAxes = Affine2D().scale(1, 1)
        self.transProjection = Affine2D().scale(1, 1)
        self.transProjectionAffine = Affine2D().scale(1, 1)

# Register the projection
proj.register_projection(PolarProjection)

# Use custom projection
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='custom_polar')

# Plot some data
theta = np.linspace(0, 2*np.pi, 100)
r = 1 + 0.5*np.sin(3*theta)
ax.plot(theta, r)

ax.set_title('Custom Polar Projection')
plt.show()
```

### Advanced Custom Projection
```python
class LogPolarProjection(proj.ProjectionBase):
    name = 'log_polar'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _set_lim_and_transforms(self):
        # Create transforms for log-polar coordinates
        self.transData = Affine2D().scale(1, 1)
        self.transAxes = Affine2D().scale(1, 1)
        self.transProjection = Affine2D().scale(1, 1)
        self.transProjectionAffine = Affine2D().scale(1, 1)
    
    def transform(self, tr):
        # Transform from data coordinates to display coordinates
        return tr
    
    def inverted(self):
        return LogPolarProjectionInverted()

class LogPolarProjectionInverted(proj.ProjectionBase):
    name = 'log_polar_inverted'
    
    def transform(self, tr):
        # Inverse transform
        return tr

# Register projection
proj.register_projection(LogPolarProjection)

# Use the projection
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='log_polar')

# Generate spiral data
theta = np.linspace(0, 4*np.pi, 200)
r = np.exp(0.1 * theta)
ax.plot(theta, r)

ax.set_title('Log-Polar Projection')
plt.show()
```

## Backend Management

### Backend Selection
```python
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt

# Available backends
print("Available backends:")
for backend in matplotlib.backend_bases.Backend.__subclasses__():
    print(f"  {backend.__name__}")

# Interactive backends
interactive_backends = ['TkAgg', 'Qt5Agg', 'Qt4Agg', 'GTK3Agg', 'MacOSX']
print(f"\nInteractive backends: {interactive_backends}")
```

### Backend-Specific Features
```python
# Check current backend
current_backend = matplotlib.get_backend()
print(f"Current backend: {current_backend}")

# Test backend capabilities
fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y)

# Backend-specific features
if 'TkAgg' in current_backend:
    print("Using TkAgg backend - good for interactive plots")
elif 'Qt' in current_backend:
    print("Using Qt backend - good for complex GUIs")
elif 'Agg' in current_backend:
    print("Using Agg backend - good for non-interactive plots")

plt.show()
```

### Backend Performance Comparison
```python
import time

def benchmark_backend(backend_name):
    """Benchmark plotting performance for a given backend."""
    matplotlib.use(backend_name)
    import matplotlib.pyplot as plt
    
    start_time = time.time()
    
    # Create complex plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, ax in enumerate(axes.flat):
        x = np.linspace(0, 10, 1000)
        y = np.sin(x + i*0.5)
        ax.plot(x, y)
        ax.scatter(x[::10], y[::10], alpha=0.6)
        ax.set_title(f'Subplot {i+1}')
    
    plt.tight_layout()
    
    # Save to memory (simulate rendering)
    fig.canvas.draw()
    
    end_time = time.time()
    plt.close(fig)
    
    return end_time - start_time

# Test different backends
backends = ['Agg', 'TkAgg', 'Qt5Agg']
results = {}

for backend in backends:
    try:
        time_taken = benchmark_backend(backend)
        results[backend] = time_taken
        print(f"{backend}: {time_taken:.3f} seconds")
    except Exception as e:
        print(f"{backend}: Error - {e}")

# Display results
if results:
    fastest = min(results, key=results.get)
    print(f"\nFastest backend: {fastest} ({results[fastest]:.3f}s)")
```

## Performance Optimization

### Efficient Plotting Techniques
```python
import matplotlib.pyplot as plt
import numpy as np
import time

# Generate large dataset
n_points = 100000
x = np.random.randn(n_points)
y = np.random.randn(n_points)

# Method 1: Standard plotting (slow)
start_time = time.time()
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(x, y, alpha=0.6, s=1)
ax.set_title('Standard Scatter Plot')
plt.show()
standard_time = time.time() - start_time

# Method 2: Efficient plotting with downsampling
start_time = time.time()
fig, ax = plt.subplots(figsize=(10, 6))

# Downsample for display
sample_size = 10000
indices = np.random.choice(n_points, sample_size, replace=False)
ax.scatter(x[indices], y[indices], alpha=0.6, s=1)
ax.set_title('Downsampled Scatter Plot')
plt.show()
efficient_time = time.time() - start_time

print(f"Standard plotting time: {standard_time:.3f}s")
print(f"Efficient plotting time: {efficient_time:.3f}s")
print(f"Speedup: {standard_time/efficient_time:.1f}x")
```

### Memory-Efficient Plotting
```python
# Memory-efficient approach for large datasets
def create_memory_efficient_plot(data_generator, n_batches=10):
    """Create plot using data generator to minimize memory usage."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i in range(n_batches):
        # Generate data in batches
        x_batch, y_batch = data_generator()
        
        # Plot batch
        ax.scatter(x_batch, y_batch, alpha=0.6, s=1, 
                  c=f'C{i}', label=f'Batch {i+1}')
    
    ax.set_title('Memory-Efficient Plotting')
    ax.legend()
    plt.show()

# Data generator function
def generate_batch():
    n_points = 10000
    return np.random.randn(n_points), np.random.randn(n_points)

# Create memory-efficient plot
create_memory_efficient_plot(generate_batch)
```

### Optimized Rendering
```python
# Optimize rendering settings
plt.rcParams['figure.dpi'] = 100  # Lower DPI for faster rendering
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.max_open_warning'] = 0  # Suppress warnings

# Use efficient rendering
fig, ax = plt.subplots(figsize=(10, 6))

# Generate data
x = np.linspace(0, 10, 1000)
y = np.sin(x)

# Optimized plotting
ax.plot(x, y, linewidth=1, alpha=0.8)  # Reduce line width and alpha
ax.grid(True, alpha=0.3, linewidth=0.5)  # Light grid

ax.set_title('Optimized Rendering')
plt.show()
```

## Advanced Customization

### Custom Color Maps
```python
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# Create custom sequential colormap
colors = ['darkblue', 'blue', 'lightblue', 'white', 'lightcoral', 'red', 'darkred']
custom_seq = LinearSegmentedColormap.from_list('custom_sequential', colors, N=256)

# Create custom diverging colormap
colors_div = ['darkred', 'red', 'lightcoral', 'white', 'lightblue', 'blue', 'darkblue']
custom_div = LinearSegmentedColormap.from_list('custom_diverging', colors_div, N=256)

# Create custom categorical colormap
categorical_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
custom_cat = ListedColormap(categorical_colors)

# Test colormaps
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Sequential
data = np.random.rand(10, 10)
im1 = axes[0].imshow(data, cmap=custom_seq)
axes[0].set_title('Custom Sequential')
plt.colorbar(im1, ax=axes[0])

# Diverging
data_div = np.random.randn(10, 10)
im2 = axes[1].imshow(data_div, cmap=custom_div)
axes[1].set_title('Custom Diverging')
plt.colorbar(im2, ax=axes[1])

# Categorical
data_cat = np.random.randint(0, 5, (10, 10))
im3 = axes[2].imshow(data_cat, cmap=custom_cat)
axes[2].set_title('Custom Categorical')
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.show()
```

### Custom Plot Styles
```python
# Define custom style
custom_style = {
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
}

# Apply custom style
plt.rcParams.update(custom_style)

# Create plot with custom style
fig, ax = plt.subplots(figsize=(10, 6))
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

ax.plot(x, y1, 'o-', label='sin(x)', markersize=6)
ax.plot(x, y2, 's-', label='cos(x)', markersize=6)

ax.set_title('Custom Style Example', fontsize=14, fontweight='bold')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.legend(fontsize=11)

plt.show()
```

### Advanced Text Rendering
```python
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.text import TextPath
from matplotlib.transforms import Affine2D

fig, ax = plt.subplots(figsize=(12, 8))

# Create advanced text elements
x = np.linspace(0, 10, 100)
y = np.sin(x)
ax.plot(x, y, linewidth=2)

# Fancy text box
fancy_box = FancyBboxPatch((2, 0.5), 3, 0.8, 
                          boxstyle="round,pad=0.1", 
                          facecolor='lightblue', 
                          edgecolor='blue', 
                          linewidth=2)
ax.add_patch(fancy_box)
ax.text(3.5, 0.9, 'Fancy Text Box', ha='center', va='center', 
        fontsize=12, fontweight='bold')

# Text with background
ax.text(7, 0.5, 'Text with Background', 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
        fontsize=12, ha='center')

# Annotated point
ax.annotate('Peak Point', xy=(np.pi/2, 1), xytext=(np.pi/2 + 1, 1.5),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax.set_title('Advanced Text Rendering', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.show()
```

## Interactive Features

### Interactive Plot with Widgets
```python
from matplotlib.widgets import Slider, Button, RadioButtons

# Create figure with subplots
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.25, bottom=0.25)

# Generate initial data
x = np.linspace(0, 10, 1000)
y = np.sin(x)
line, = ax.plot(x, y, linewidth=2)

# Create sliders
ax_freq = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_amp = plt.axes([0.25, 0.15, 0.65, 0.03])

freq_slider = Slider(ax_freq, 'Frequency', 0.1, 3.0, valinit=1.0)
amp_slider = Slider(ax_amp, 'Amplitude', 0.1, 2.0, valinit=1.0)

# Create buttons
ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
button_reset = Button(ax_reset, 'Reset')

# Create radio buttons
ax_radio = plt.axes([0.025, 0.5, 0.15, 0.15])
radio = RadioButtons(ax_radio, ('sin', 'cos', 'tan'))

def update(val):
    freq = freq_slider.val
    amp = amp_slider.val
    func = radio.value_selected
    
    if func == 'sin':
        y_new = amp * np.sin(freq * x)
    elif func == 'cos':
        y_new = amp * np.cos(freq * x)
    else:  # tan
        y_new = amp * np.tan(freq * x)
    
    line.set_ydata(y_new)
    fig.canvas.draw_idle()

def reset(event):
    freq_slider.reset()
    amp_slider.reset()

# Connect events
freq_slider.on_changed(update)
amp_slider.on_changed(update)
button_reset.on_clicked(reset)
radio.on_clicked(update)

ax.set_title('Interactive Plot with Widgets')
ax.grid(True, alpha=0.3)
plt.show()
```

### Interactive Data Selection
```python
class DataSelector:
    def __init__(self, ax):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.selected_points = []
        self.selection_rect = None
        self.start_point = None
        
        # Connect events
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.start_point = (event.xdata, event.ydata)
        
    def on_motion(self, event):
        if self.start_point is None or event.inaxes != self.ax:
            return
        
        # Update selection rectangle
        if self.selection_rect:
            self.selection_rect.remove()
        
        x0, y0 = self.start_point
        x1, y1 = event.xdata, event.ydata
        
        self.selection_rect = plt.Rectangle((min(x0, x1), min(y0, y1)), 
                                          abs(x1-x0), abs(y1-y0), 
                                          fill=False, color='red', linewidth=2)
        self.ax.add_patch(self.selection_rect)
        self.canvas.draw_idle()
        
    def on_release(self, event):
        if self.start_point is None or event.inaxes != self.ax:
            return
        
        # Select points in rectangle
        x0, y0 = self.start_point
        x1, y1 = event.xdata, event.ydata
        
        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)
        
        # Find points in selection
        mask = ((self.x >= x_min) & (self.x <= x_max) & 
                (self.y >= y_min) & (self.y <= y_max))
        
        # Highlight selected points
        self.ax.scatter(self.x[mask], self.y[mask], 
                       c='red', s=100, alpha=0.7, zorder=5)
        
        self.start_point = None
        if self.selection_rect:
            self.selection_rect.remove()
            self.selection_rect = None
        
        self.canvas.draw()

# Create interactive plot
fig, ax = plt.subplots(figsize=(10, 6))

# Generate data
np.random.seed(42)
x = np.random.randn(100)
y = np.random.randn(100)
ax.scatter(x, y, alpha=0.6)

# Add selector
selector = DataSelector(ax)
selector.x = x
selector.y = y

ax.set_title('Click and drag to select points')
ax.grid(True, alpha=0.3)
plt.show()
```

## Custom Artists

### Custom Shape Artist
```python
from matplotlib.artist import Artist
from matplotlib.patches import Polygon

class CustomStar(Artist):
    def __init__(self, xy, size=1.0, **kwargs):
        super().__init__()
        self.xy = xy
        self.size = size
        self.kwargs = kwargs
        
    def draw(self, renderer):
        if not self.get_visible():
            return
        
        # Create star shape
        angles = np.linspace(0, 2*np.pi, 11)[:-1]
        outer_radius = self.size
        inner_radius = self.size * 0.4
        
        x_coords = []
        y_coords = []
        
        for i, angle in enumerate(angles):
            radius = outer_radius if i % 2 == 0 else inner_radius
            x_coords.append(self.xy[0] + radius * np.cos(angle))
            y_coords.append(self.xy[1] + radius * np.sin(angle))
        
        # Create polygon
        star = Polygon(list(zip(x_coords, y_coords)), **self.kwargs)
        star.draw(renderer)

# Use custom artist
fig, ax = plt.subplots(figsize=(8, 8))

# Add custom stars
star1 = CustomStar((0, 0), size=2, facecolor='gold', edgecolor='orange', linewidth=2)
star2 = CustomStar((3, 3), size=1.5, facecolor='red', edgecolor='darkred', linewidth=2)
star3 = CustomStar((-2, 2), size=1, facecolor='blue', edgecolor='darkblue', linewidth=2)

ax.add_artist(star1)
ax.add_artist(star2)
ax.add_artist(star3)

ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.set_title('Custom Star Artists')
ax.grid(True, alpha=0.3)
plt.show()
```

### Custom Line Artist
```python
class DashedLine(Artist):
    def __init__(self, x, y, dash_length=0.5, **kwargs):
        super().__init__()
        self.x = x
        self.y = y
        self.dash_length = dash_length
        self.kwargs = kwargs
        
    def draw(self, renderer):
        if not self.get_visible():
            return
        
        # Create dashed line segments
        for i in range(len(self.x) - 1):
            if i % 2 == 0:  # Draw dash
                line, = plt.plot([self.x[i], self.x[i+1]], 
                               [self.y[i], self.y[i+1]], **self.kwargs)
                line.draw(renderer)

# Use custom line artist
fig, ax = plt.subplots(figsize=(10, 6))

x = np.linspace(0, 10, 100)
y = np.sin(x)

# Regular line
ax.plot(x, y, 'b-', linewidth=2, label='Regular Line')

# Custom dashed line
dashed_line = DashedLine(x, y + 1, dash_length=0.5, 
                        color='red', linewidth=2, label='Custom Dashed')
ax.add_artist(dashed_line)

ax.set_title('Custom Line Artist')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()
```

## Advanced Layout

### Complex Grid Layout
```python
from matplotlib.gridspec import GridSpec

# Create complex layout
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

plt.suptitle('Complex Grid Layout', fontsize=16, fontweight='bold')
plt.show()
```

### Dynamic Layout Management
```python
class DynamicLayout:
    def __init__(self, n_plots):
        self.n_plots = n_plots
        self.fig = None
        self.axes = []
        
    def create_layout(self):
        # Calculate optimal grid
        cols = int(np.ceil(np.sqrt(self.n_plots)))
        rows = int(np.ceil(self.n_plots / cols))
        
        # Create figure and subplots
        self.fig, self.axes = plt.subplots(rows, cols, 
                                          figsize=(4*cols, 3*rows))
        
        # Flatten axes if needed
        if rows == 1 and cols == 1:
            self.axes = [self.axes]
        elif rows == 1 or cols == 1:
            self.axes = self.axes.flatten()
        else:
            self.axes = self.axes.flatten()
        
        # Hide unused subplots
        for i in range(self.n_plots, len(self.axes)):
            self.axes[i].set_visible(False)
        
        return self.fig, self.axes
    
    def add_plot(self, plot_func, index, **kwargs):
        """Add a plot to a specific subplot."""
        if index < len(self.axes):
            plot_func(self.axes[index], **kwargs)

# Example usage
def line_plot(ax, **kwargs):
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y, **kwargs)
    ax.set_title('Line Plot')
    ax.grid(True, alpha=0.3)

def scatter_plot(ax, **kwargs):
    np.random.seed(42)
    x = np.random.randn(100)
    y = np.random.randn(100)
    ax.scatter(x, y, **kwargs)
    ax.set_title('Scatter Plot')
    ax.grid(True, alpha=0.3)

def histogram_plot(ax, **kwargs):
    data = np.random.normal(0, 1, 1000)
    ax.hist(data, bins=30, **kwargs)
    ax.set_title('Histogram')
    ax.grid(True, alpha=0.3)

# Create dynamic layout
layout = DynamicLayout(3)
fig, axes = layout.create_layout()

# Add plots
layout.add_plot(line_plot, 0, linewidth=2, color='blue')
layout.add_plot(scatter_plot, 1, alpha=0.6, color='red')
layout.add_plot(histogram_plot, 2, alpha=0.7, color='green')

plt.suptitle('Dynamic Layout Management', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Best Practices

1. **Choose appropriate backends** for your use case
2. **Optimize performance** for large datasets
3. **Use event handling** for interactive features
4. **Create reusable components** with custom artists
5. **Plan layouts carefully** for complex visualizations
6. **Test across different environments** and backends
7. **Document custom features** for maintainability
8. **Consider memory usage** for large-scale applications

## Resources

- [Matplotlib Advanced Features](https://matplotlib.org/stable/tutorials/advanced/index.html)
- [Event Handling Guide](https://matplotlib.org/stable/users/event_handling.html)
- [Custom Artists](https://matplotlib.org/stable/tutorials/advanced/artists.html)
- [Backend Management](https://matplotlib.org/stable/tutorials/introductory/usage.html#backends)

---

**Master advanced Matplotlib techniques!** 🚀 