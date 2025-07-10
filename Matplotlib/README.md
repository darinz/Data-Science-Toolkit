# Matplotlib Tutorials

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-blue.svg)](https://matplotlib.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Conda](https://img.shields.io/badge/Conda-Environment-green.svg)](https://docs.conda.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

A comprehensive guide to Matplotlib, the most popular plotting library for Python, covering data visualization fundamentals and advanced techniques.

## What's Included

### Matplotlib Fundamentals
- **Basic Plotting** - Line plots, scatter plots, bar charts, and histograms
- **Figure and Axes** - Understanding the Matplotlib architecture
- **Customization** - Colors, styles, labels, and legends
- **Subplots** - Creating multi-panel visualizations
- **Saving and Exporting** - High-quality image formats and settings

### Advanced Visualization
- **Statistical Plots** - Box plots, violin plots, and heatmaps
- **3D Plotting** - Surface plots, scatter plots, and wireframes
- **Geographic Plots** - Maps and spatial data visualization
- **Interactive Plots** - Dynamic visualizations with user interaction
- **Animation** - Creating animated plots and visualizations

### Data Science Applications
- **Exploratory Data Analysis** - Visualizing data distributions and relationships
- **Model Evaluation** - Plotting model performance and results
- **Time Series Visualization** - Temporal data analysis and forecasting
- **Scientific Plotting** - Publication-ready scientific figures
- **Dashboard Creation** - Building comprehensive data dashboards

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Basic understanding of Python programming
- Familiarity with NumPy (covered in NumPy tutorials)

### Installation

**Using Conda (Recommended):**
```bash
cd Matplotlib
conda env create -f environment.yml
conda activate matplotlib-tutorials
```

**Using pip:**
```bash
cd Matplotlib
pip install -r requirements.txt
```

### Verify Installation
```python
import matplotlib
import matplotlib.pyplot as plt
print(f"Matplotlib version: {matplotlib.__version__}")
```

## File Structure

```
Matplotlib/
├── 01_matplotlib_basics_guide.md     # Comprehensive basics tutorial
├── 02_plot_types_guide.md            # Different plot types and examples
├── 03_customization_guide.md         # Colors, styles, and customization
├── 04_subplots_layout_guide.md       # Multi-panel figures and layouts
├── 05_statistical_plots_guide.md     # Statistical visualization
├── 06_3d_plotting_guide.md           # 3D plotting and visualization
├── 07_advanced_features_guide.md     # Advanced features and techniques
├── 08_publication_quality_guide.md   # Publication-ready figures
├── environment.yml                   # Conda environment configuration
├── requirements.txt                  # pip dependencies
└── README.md                         # This file
```

## Tutorial Structure

### 1. Matplotlib Basics (`01_matplotlib_basics_guide.md`)
- Introduction to Matplotlib architecture
- Basic plotting functions and syntax
- Figure and axes objects
- Customizing plot appearance
- **Read:** `01_matplotlib_basics_guide.md`

### 2. Plot Types (`02_plot_types_guide.md`)
- Line plots and scatter plots
- Bar charts and histograms
- Pie charts and area plots
- Error bars and confidence intervals
- Statistical plots and distributions

### 3. Customization (`03_customization_guide.md`)
- Colors, markers, and line styles
- Labels, titles, and legends
- Grids and spines
- Text annotations and arrows
- Themes and style sheets

### 4. Subplots and Layout (`04_subplots_layout_guide.md`)
- Creating subplots
- GridSpec for complex layouts
- Sharing axes and ranges
- Figure size and DPI settings
- Multi-panel dashboards

### 5. Statistical Visualization (`05_statistical_plots_guide.md`)
- Box plots and violin plots
- Histograms and density plots
- Heatmaps and correlation matrices
- Statistical annotations
- Distribution analysis

### 6. 3D Plotting (`06_3d_plotting_guide.md`)
- 3D scatter plots
- Surface plots and wireframes
- Contour plots
- 3D customization and interaction
- Scientific 3D plotting

### 7. Advanced Features (`07_advanced_features_guide.md`)
- Interactive plots
- Animations and transitions
- Custom projections
- Geographic plotting
- Performance optimization

### 8. Publication Quality (`08_publication_quality_guide.md`)
- High-resolution output
- LaTeX integration
- Style sheets and themes
- Export to various formats
- Professional figure preparation

## Running the Tutorials

### Reading the Guides
```bash
# Open guides in your preferred markdown viewer
# Or use a text editor to read the markdown files
```

### Sequential Learning
```bash
# Read guides in order for comprehensive learning
# Start with 01_matplotlib_basics_guide.md
# Then proceed through each guide sequentially
```

## Environment Management

### Conda Commands

```bash
# Create new environment
conda env create -f environment.yml

# Activate environment
conda activate matplotlib-tutorials

# Update environment (when dependencies change)
conda env update -f environment.yml --prune

# Remove environment
conda remove --name matplotlib-tutorials --all

# List all environments
conda env list

# Deactivate current environment
conda deactivate
```

### pip Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Check installed packages
pip list

# Export current environment
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
```

## Learning Path

### Beginner Path
1. **Start with Matplotlib Basics** - Read `01_matplotlib_basics_guide.md` to understand the fundamentals
2. **Learn Plot Types** - Master different visualization types with `02_plot_types_guide.md`
3. **Customize Your Plots** - Make them look professional with `03_customization_guide.md`
4. **Create Complex Layouts** - Build multi-panel figures with `04_subplots_layout_guide.md`

### Intermediate Path
1. **Explore Statistical Plots** - Visualize data distributions with `05_statistical_plots_guide.md`
2. **Dive into 3D Plotting** - Add dimensionality to your visualizations with `06_3d_plotting_guide.md`
3. **Master Advanced Features** - Create interactive and animated plots with `07_advanced_features_guide.md`
4. **Produce Publication Quality** - Create figures for papers and presentations with `08_publication_quality_guide.md`

### Advanced Path
1. **Custom Animations** - Create dynamic visualizations
2. **Geographic Plots** - Work with spatial data
3. **Performance Optimization** - Handle large datasets efficiently
4. **Integration Projects** - Combine with other libraries

## Key Concepts Covered

### Matplotlib Architecture
- **Figure** - The top-level container for all plot elements
- **Axes** - The area where plots are drawn
- **Artist** - The basic drawing primitive
- **Backend** - The rendering engine

### Plotting Fundamentals
- **Line Plots** - Basic line and scatter plots
- **Bar Charts** - Categorical data visualization
- **Histograms** - Distribution visualization
- **Subplots** - Multi-panel figures

### Customization
- **Colors and Styles** - Visual appearance control
- **Labels and Annotations** - Text and information display
- **Legends and Grids** - Plot organization and clarity
- **Themes** - Consistent styling across plots

### Advanced Features
- **3D Plotting** - Three-dimensional visualizations
- **Animations** - Dynamic plot updates
- **Interactive Elements** - User interaction capabilities
- **Export Options** - Multiple output formats

## Common Use Cases

### Data Exploration
```python
import matplotlib.pyplot as plt
import numpy as np

# Create sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Basic line plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Wave')
plt.legend()
plt.grid(True)
plt.show()
```

### Statistical Visualization
```python
import matplotlib.pyplot as plt
import numpy as np

# Create sample data
data = np.random.normal(0, 1, 1000)

# Histogram with KDE
plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, density=True, alpha=0.7, color='skyblue')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Normal Distribution')
plt.show()
```

### Multi-panel Figure
```python
import matplotlib.pyplot as plt
import numpy as np

# Create data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# First subplot
ax1.plot(x, y1, 'b-', label='sin(x)')
ax1.set_title('Sine Function')
ax1.legend()
ax1.grid(True)

# Second subplot
ax2.plot(x, y2, 'r-', label='cos(x)')
ax2.set_title('Cosine Function')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## Integration with Other Libraries

### NumPy Integration
```python
import matplotlib.pyplot as plt
import numpy as np

# Create data with NumPy
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Plot with Matplotlib
plt.plot(x, y)
plt.show()
```

### pandas Integration
```python
import matplotlib.pyplot as plt
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'x': range(10),
    'y': range(10)
})

# Plot directly from pandas
df.plot(x='x', y='y', kind='scatter')
plt.show()
```

### Seaborn Integration
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")

# Create plot
plt.figure(figsize=(10, 6))
sns.histplot(data=data, bins=30)
plt.title('Distribution with Seaborn Style')
plt.show()
```

## Additional Resources

### Official Documentation
- [Matplotlib Official Documentation](https://matplotlib.org/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Matplotlib Examples](https://matplotlib.org/stable/gallery/index.html)
- [Matplotlib Cheat Sheet](https://matplotlib.org/cheatsheets/)

### Learning Resources
- [Matplotlib GitHub Repository](https://github.com/matplotlib/matplotlib)
- [Matplotlib Community](https://matplotlib.org/community/)
- [Matplotlib Style Gallery](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html)

### Recommended Books
- "Python Data Science Handbook" by Jake VanderPlas
- "Matplotlib for Python Developers" by Sandro Tosi
- "Effective Matplotlib" by Benjamin Root

## Contributing

Found an error or have a suggestion? Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Happy Plotting!** 