# Matplotlib Tutorials

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

## Tutorial Structure

### 1. Matplotlib Basics (`matplotlib_basics.py`)
- Introduction to Matplotlib architecture
- Basic plotting functions
- Figure and axes objects
- Customizing plot appearance
- **Run with:** `python matplotlib_basics.py`

### 2. Plot Types (`plot_types.ipynb`)
- Line plots and scatter plots
- Bar charts and histograms
- Pie charts and area plots
- Error bars and confidence intervals

### 3. Customization (`customization.ipynb`)
- Colors, markers, and line styles
- Labels, titles, and legends
- Grids and spines
- Text annotations and arrows

### 4. Subplots and Layout (`subplots_layout.ipynb`)
- Creating subplots
- GridSpec for complex layouts
- Sharing axes and ranges
- Figure size and DPI settings

### 5. Statistical Visualization (`statistical_plots.ipynb`)
- Box plots and violin plots
- Histograms and density plots
- Heatmaps and correlation matrices
- Statistical annotations

### 6. 3D Plotting (`3d_plotting.ipynb`)
- 3D scatter plots
- Surface plots and wireframes
- Contour plots
- 3D customization

### 7. Advanced Features (`advanced_features.ipynb`)
- Interactive plots
- Animations
- Custom projections
- Geographic plotting

### 8. Publication Quality (`publication_quality.ipynb`)
- High-resolution output
- LaTeX integration
- Style sheets and themes
- Export to various formats

## Learning Path

1. **Start with Matplotlib Basics** - Run the Python script to understand the fundamentals
2. **Learn Plot Types** - Master different visualization types
3. **Customize Your Plots** - Make them look professional
4. **Create Complex Layouts** - Build multi-panel figures
5. **Explore Statistical Plots** - Visualize data distributions
6. **Dive into 3D Plotting** - Add dimensionality to your visualizations
7. **Master Advanced Features** - Create interactive and animated plots
8. **Produce Publication Quality** - Create figures for papers and presentations

## Key Features

- **Comprehensive Plotting** - Support for all major plot types
- **High-Quality Output** - Publication-ready figures
- **Customizable** - Complete control over appearance
- **Integration** - Works seamlessly with NumPy, pandas, and other libraries
- **Cross-Platform** - Consistent output across different systems

## Style and Themes

- **Default Style** - Clean, professional appearance
- **Seaborn Integration** - Statistical plotting styles
- **Custom Themes** - Create your own visual identity
- **Publication Styles** - Pre-configured for academic papers

## Output Formats

- **PNG** - Raster format for web and presentations
- **PDF** - Vector format for publications
- **SVG** - Scalable vector graphics
- **EPS** - Encapsulated PostScript
- **Interactive HTML** - Web-based visualizations

## Running the Tutorials

### Python Script (Recommended for Basics)
```bash
# Run the comprehensive basics tutorial
python matplotlib_basics.py

# This will create an 'output' directory with all generated plots
```

### Jupyter Notebooks (For Interactive Learning)
```bash
# Start Jupyter Lab
jupyter lab

# Or start Jupyter Notebook
jupyter notebook
```

## Resources

- [Matplotlib Official Documentation](https://matplotlib.org/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Matplotlib Examples](https://matplotlib.org/stable/gallery/index.html)
- [Matplotlib Cheat Sheet](https://matplotlib.org/cheatsheets/)

## Support

For issues and questions:
- Check the individual tutorial README files
- Refer to Matplotlib documentation
- Open an issue on GitHub

---

**Happy Plotting!** 