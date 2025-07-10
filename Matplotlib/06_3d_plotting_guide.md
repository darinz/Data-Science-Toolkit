# Matplotlib 3D Plotting: Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [3D Line Plots](#3d-line-plots)
3. [3D Scatter Plots](#3d-scatter-plots)
4. [3D Surface Plots](#3d-surface-plots)
5. [3D Wireframe Plots](#3d-wireframe-plots)
6. [3D Contour Plots](#3d-contour-plots)
7. [3D Bar Plots](#3d-bar-plots)
8. [Advanced 3D Techniques](#advanced-3d-techniques)
9. [3D Animation](#3d-animation)
10. [Best Practices](#best-practices)

## Introduction

Matplotlib's 3D plotting capabilities allow you to create sophisticated three-dimensional visualizations. This guide covers 3D line plots, scatter plots, surface plots, wireframes, contour plots, and advanced 3D techniques.

## 3D Line Plots

### Basic 3D Line Plot
```python
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Create 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate data
t = np.linspace(0, 10, 100)
x = np.sin(t)
y = np.cos(t)
z = t

# Create 3D line plot
ax.plot(x, y, z, linewidth=2, label='3D Line')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Basic 3D Line Plot')
ax.legend()
plt.show()
```

### Multiple 3D Lines
```python
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate multiple 3D curves
t = np.linspace(0, 4*np.pi, 100)

# Helix 1
x1 = np.cos(t)
y1 = np.sin(t)
z1 = t

# Helix 2 (different radius and pitch)
x2 = 2 * np.cos(t)
y2 = 2 * np.sin(t)
z2 = t * 0.5

# Helix 3 (different phase)
x3 = np.cos(t + np.pi/2)
y3 = np.sin(t + np.pi/2)
z3 = t

# Plot all curves
ax.plot(x1, y1, z1, 'b-', linewidth=2, label='Helix 1')
ax.plot(x2, y2, z2, 'r-', linewidth=2, label='Helix 2')
ax.plot(x3, y3, z3, 'g-', linewidth=2, label='Helix 3')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Multiple 3D Lines')
ax.legend()
plt.show()
```

### Parametric 3D Curve
```python
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate parametric curve data
t = np.linspace(0, 2*np.pi, 200)
x = np.cos(t) * (1 + np.cos(2*t))
y = np.sin(t) * (1 + np.cos(2*t))
z = np.sin(2*t)

# Create 3D parametric plot
ax.plot(x, y, z, linewidth=3, color='purple', label='Parametric Curve')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Parametric 3D Curve')
ax.legend()
plt.show()
```

## 3D Scatter Plots

### Basic 3D Scatter Plot
```python
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate random 3D data
np.random.seed(42)
n_points = 100
x = np.random.randn(n_points)
y = np.random.randn(n_points)
z = np.random.randn(n_points)

# Create 3D scatter plot
scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.6)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Basic 3D Scatter Plot')

# Add colorbar
plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
plt.show()
```

### 3D Scatter with Different Colors and Sizes
```python
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate data with different properties
np.random.seed(42)
n_points = 200
x = np.random.randn(n_points)
y = np.random.randn(n_points)
z = np.random.randn(n_points)

# Create color and size arrays
colors = np.random.rand(n_points)
sizes = 100 * np.random.rand(n_points)

# Create 3D scatter plot
scatter = ax.scatter(x, y, z, c=colors, s=sizes, cmap='plasma', alpha=0.7)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot with Variable Colors and Sizes')

# Add colorbar
plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
plt.show()
```

### 3D Scatter with Categories
```python
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate data for different categories
np.random.seed(42)
categories = ['A', 'B', 'C']
colors = ['red', 'blue', 'green']

for i, (cat, color) in enumerate(zip(categories, colors)):
    # Generate data for each category
    n_points = 50
    x = np.random.normal(i, 0.5, n_points)
    y = np.random.normal(0, 1, n_points)
    z = np.random.normal(0, 1, n_points)
    
    ax.scatter(x, y, z, c=color, s=50, alpha=0.7, label=f'Category {cat}')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot with Categories')
ax.legend()
plt.show()
```

## 3D Surface Plots

### Basic 3D Surface Plot
```python
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate grid data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Create surface function
Z = np.sin(np.sqrt(X**2 + Y**2))

# Create 3D surface plot
surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface Plot')

# Add colorbar
plt.colorbar(surface, ax=ax, shrink=0.5, aspect=20)
plt.show()
```

### Multiple 3D Surfaces
```python
fig = plt.figure(figsize=(15, 5))

# Create different surface functions
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)

# Surface 1: Gaussian
Z1 = np.exp(-(X**2 + Y**2) / 2)

# Surface 2: Saddle
Z2 = X**2 - Y**2

# Surface 3: Ripple
Z3 = np.sin(X) * np.cos(Y)

surfaces = [Z1, Z2, Z3]
titles = ['Gaussian Surface', 'Saddle Surface', 'Ripple Surface']
cmaps = ['viridis', 'plasma', 'coolwarm']

for i, (Z, title, cmap) in enumerate(zip(surfaces, titles, cmaps)):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    surface = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.colorbar(surface, ax=ax, shrink=0.5, aspect=20)

plt.tight_layout()
plt.show()
```

### 3D Surface with Custom Colormap
```python
from matplotlib.colors import LinearSegmentedColormap

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

# Create complex surface
Z = np.sin(X) * np.cos(Y) * np.exp(-(X**2 + Y**2) / 10)

# Create custom colormap
colors = ['darkblue', 'blue', 'lightblue', 'white', 'lightcoral', 'red', 'darkred']
custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)

# Create surface plot
surface = ax.plot_surface(X, Y, Z, cmap=custom_cmap, alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface with Custom Colormap')

plt.colorbar(surface, ax=ax, shrink=0.5, aspect=20)
plt.show()
```

## 3D Wireframe Plots

### Basic 3D Wireframe
```python
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate grid data
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)

# Create surface function
Z = np.sin(np.sqrt(X**2 + Y**2))

# Create 3D wireframe
wireframe = ax.plot_wireframe(X, Y, Z, color='blue', linewidth=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Wireframe Plot')
plt.show()
```

### 3D Wireframe with Surface
```python
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate data
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Create surface plot
surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3)

# Add wireframe overlay
wireframe = ax.plot_wireframe(X, Y, Z, color='black', linewidth=0.5, alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface with Wireframe Overlay')

plt.colorbar(surface, ax=ax, shrink=0.5, aspect=20)
plt.show()
```

## 3D Contour Plots

### 3D Contour Plot
```python
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate data
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Create 3D contour plot
contour = ax.contour3D(X, Y, Z, 50, cmap='viridis')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Contour Plot')

plt.colorbar(contour, ax=ax, shrink=0.5, aspect=20)
plt.show()
```

### 3D Contour with Surface
```python
fig = plt.figure(figsize=(15, 5))

# Generate data
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Surface plot
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
surface = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_title('Surface Plot')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Contour plot
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
contour = ax2.contour3D(X, Y, Z, 30, cmap='plasma')
ax2.set_title('3D Contour Plot')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

# Combined plot
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
surface_combined = ax3.plot_surface(X, Y, Z, cmap='viridis', alpha=0.3)
contour_combined = ax3.contour3D(X, Y, Z, 20, cmap='plasma')
ax3.set_title('Surface + Contour')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

plt.tight_layout()
plt.show()
```

## 3D Bar Plots

### Basic 3D Bar Plot
```python
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate data
x = np.arange(5)
y = np.arange(5)
X, Y = np.meshgrid(x, y)
Z = np.random.rand(5, 5)

# Create 3D bar plot
dx = dy = 0.8
ax.bar3d(X.flatten(), Y.flatten(), np.zeros_like(Z.flatten()), 
         dx, dy, Z.flatten(), alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Bar Plot')
plt.show()
```

### 3D Bar Plot with Colors
```python
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate data
x = np.arange(8)
y = np.arange(8)
X, Y = np.meshgrid(x, y)
Z = np.random.rand(8, 8)

# Create color array
colors = plt.cm.viridis(Z.flatten())

# Create 3D bar plot
dx = dy = 0.8
bars = ax.bar3d(X.flatten(), Y.flatten(), np.zeros_like(Z.flatten()), 
                dx, dy, Z.flatten(), color=colors, alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Bar Plot with Colors')

# Add colorbar
norm = plt.Normalize(Z.min(), Z.max())
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
plt.show()
```

## Advanced 3D Techniques

### 3D Vector Field
```python
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate grid
x = np.linspace(-2, 2, 8)
y = np.linspace(-2, 2, 8)
z = np.linspace(-2, 2, 8)
X, Y, Z = np.meshgrid(x, y, z)

# Create vector field
U = Y
V = -X
W = Z * 0.5

# Create 3D quiver plot
ax.quiver(X, Y, Z, U, V, W, length=0.3, normalize=True, alpha=0.7)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Vector Field')
plt.show()
```

### 3D Scatter with Trajectories
```python
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate trajectory data
t = np.linspace(0, 10, 100)
x = np.cos(t)
y = np.sin(t)
z = t * 0.5

# Plot trajectory
ax.plot(x, y, z, 'b-', linewidth=2, label='Trajectory')

# Add scatter points at specific times
times = [0, 2, 4, 6, 8, 10]
for time in times:
    idx = int(time * 10)
    ax.scatter(x[idx], y[idx], z[idx], s=100, c='red', alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Trajectory with Points')
ax.legend()
plt.show()
```

### 3D Surface with Projections
```python
fig = plt.figure(figsize=(15, 10))

# Generate data
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

# Main 3D plot
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
surface = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_title('3D Surface')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# XY projection (contour)
ax2 = fig.add_subplot(2, 2, 2)
contour_xy = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
ax2.set_title('XY Projection (Contour)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
plt.colorbar(contour_xy, ax=ax2)

# XZ projection
ax3 = fig.add_subplot(2, 2, 3)
contour_xz = ax3.contour(X, Z, Y, levels=20, cmap='plasma')
ax3.set_title('XZ Projection')
ax3.set_xlabel('X')
ax3.set_ylabel('Z')
plt.colorbar(contour_xz, ax=ax3)

# YZ projection
ax4 = fig.add_subplot(2, 2, 4)
contour_yz = ax4.contour(Y, Z, X, levels=20, cmap='coolwarm')
ax4.set_title('YZ Projection')
ax4.set_xlabel('Y')
ax4.set_ylabel('Z')
plt.colorbar(contour_yz, ax=ax4)

plt.tight_layout()
plt.show()
```

## 3D Animation

### Animated 3D Surface
```python
from matplotlib.animation import FuncAnimation

# Create figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate data
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)

def animate(frame):
    ax.clear()
    
    # Create animated surface
    Z = np.sin(X + frame * 0.1) * np.cos(Y + frame * 0.1)
    
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Animated 3D Surface (Frame {frame})')
    
    return surface,

# Create animation
ani = FuncAnimation(fig, animate, frames=50, interval=100, blit=False)
plt.show()
```

### Animated 3D Scatter
```python
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate initial data
np.random.seed(42)
n_points = 100
x = np.random.randn(n_points)
y = np.random.randn(n_points)
z = np.random.randn(n_points)

scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=50, alpha=0.6)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Animated 3D Scatter')

def animate(frame):
    # Update positions
    x_new = x + 0.1 * np.sin(frame * 0.1)
    y_new = y + 0.1 * np.cos(frame * 0.1)
    z_new = z + 0.05 * np.sin(frame * 0.2)
    
    scatter._offsets3d = (x_new, y_new, z_new)
    return scatter,

ani = FuncAnimation(fig, animate, frames=100, interval=50, blit=True)
plt.show()
```

## Best Practices

1. **Choose appropriate plot types** for your data and analysis goals
2. **Use proper aspect ratios** to avoid distortion
3. **Include clear labels** for all axes
4. **Consider viewing angles** that best show your data
5. **Use color effectively** to enhance understanding
6. **Test different perspectives** to find the best view
7. **Consider performance** for large datasets
8. **Use transparency** to show overlapping elements

## Resources

- [Matplotlib 3D Plotting](https://matplotlib.org/stable/tutorials/toolkits/mplot3d.html)
- [3D Plotting Examples](https://matplotlib.org/stable/gallery/mplot3d/index.html)
- [Advanced 3D Techniques](https://matplotlib.org/stable/api/toolkits/mplot3d.html)

---

**Create stunning 3D visualizations!**