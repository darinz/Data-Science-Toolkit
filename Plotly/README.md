# Plotly Tutorials

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Plotly](https://img.shields.io/badge/Plotly-5.0+-purple.svg)](https://plotly.com/python/)
[![Dash](https://img.shields.io/badge/Dash-2.0+-blue.svg)](https://dash.plotly.com/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Conda](https://img.shields.io/badge/Conda-Environment-green.svg)](https://docs.conda.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

A comprehensive guide to Plotly, the leading interactive plotting library for Python, covering data visualization fundamentals and advanced techniques for creating stunning interactive charts and dashboards.

## What's Included

### Plotly Fundamentals
- **Basic Plotting** - Line plots, scatter plots, bar charts, and histograms
- **Interactive Features** - Zoom, pan, hover, and selection tools
- **Customization** - Colors, styles, layouts, and themes
- **Subplots** - Creating multi-panel interactive visualizations
- **Export and Sharing** - HTML, PNG, SVG, and PDF formats

### Advanced Visualization
- **Statistical Plots** - Box plots, violin plots, and heatmaps
- **3D Plotting** - Surface plots, scatter plots, and wireframes
- **Geographic Plots** - Maps, choropleth, and scattergeo
- **Financial Plots** - Candlestick, OHLC, and technical indicators
- **Scientific Plots** - Contour plots, polar plots, and ternary diagrams

### Interactive Dashboards
- **Dash Framework** - Building web applications with Plotly
- **Real-time Updates** - Live data visualization
- **User Interactions** - Callbacks and event handling
- **Responsive Design** - Mobile-friendly dashboards
- **Deployment** - Hosting and sharing dashboards

### Data Science Applications
- **Exploratory Data Analysis** - Interactive data exploration
- **Model Evaluation** - Interactive model performance visualization
- **Time Series Analysis** - Temporal data with interactive features
- **Machine Learning Visualization** - Model results and feature importance
- **Big Data Visualization** - Efficient plotting of large datasets

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Basic understanding of Python programming
- Familiarity with NumPy and pandas (covered in respective tutorials)

### Installation

**Using Conda (Recommended):**
```bash
cd Plotly
conda env create -f environment.yml
conda activate plotly-tutorials
```

**Using pip:**
```bash
cd Plotly
pip install -r requirements.txt
```

### Verify Installation
```python
import plotly
import plotly.express as px
import plotly.graph_objects as go
print(f"Plotly version: {plotly.__version__}")
```

## Tutorial Structure

### 1. Plotly Basics (`01_plotly_basics_guide.md`)
- Introduction to Plotly architecture
- Basic plotting with Plotly Express
- Graph Objects for advanced customization
- Interactive features and tools
- **Run with:** `python plotly_basics.py`

### 2. Plot Types (`02_plot_types_guide.md`)
- Line plots and scatter plots
- Bar charts and histograms
- Pie charts and area plots
- Error bars and confidence intervals
- Statistical plots and distributions

### 3. Interactive Features (`03_interactive_features_guide.md`)
- Zoom, pan, and selection tools
- Hover information and annotations
- Click events and callbacks
- Range sliders and buttons
- Animation and transitions

### 4. Customization (`04_customization_guide.md`)
- Colors, themes, and styles
- Layouts and templates
- Annotations and shapes
- Legends and axes customization
- Responsive design

### 5. Subplots and Layouts (`05_subplots_layout_guide.md`)
- Creating subplots
- Complex layouts with make_subplots
- Shared axes and ranges
- Figure composition
- Multi-panel dashboards

### 6. 3D Visualization (`06_3d_visualization_guide.md`)
- 3D scatter plots
- Surface plots and wireframes
- Contour plots and heatmaps
- 3D customization and interaction
- Scientific 3D plotting

### 7. Geographic Visualization (`07_geographic_visualization_guide.md`)
- Choropleth maps
- Scattergeo plots
- Map projections and styling
- Geographic data handling
- Interactive maps

### 8. Statistical Visualization (`08_statistical_visualization_guide.md`)
- Box plots and violin plots
- Histograms and density plots
- Correlation matrices
- Statistical annotations
- Distribution analysis

### 9. Advanced Features (`09_advanced_features_guide.md`)
- Custom traces and shapes
- Animation and transitions
- Real-time data updates
- Performance optimization
- Integration with other libraries

### 10. Dash Applications (`10_dash_applications_guide.md`)
- Building web applications
- Layout and components
- Callbacks and interactivity
- Styling and themes
- Deployment and hosting

## Learning Path

1. **Start with Plotly Basics** - Run the Python script to understand the fundamentals
2. **Learn Plot Types** - Master different visualization types with interactivity
3. **Explore Interactive Features** - Understand zoom, pan, hover, and selection
4. **Customize Your Plots** - Make them look professional and branded
5. **Create Complex Layouts** - Build multi-panel interactive figures
6. **Dive into 3D Visualization** - Add dimensionality to your visualizations
7. **Master Geographic Plots** - Create interactive maps and spatial visualizations
8. **Explore Statistical Plots** - Visualize data distributions and relationships
9. **Build Advanced Features** - Create custom traces and animations
10. **Develop Dash Applications** - Build complete web applications

## Key Features

- **Interactive by Default** - All plots are interactive with zoom, pan, hover
- **Multiple Output Formats** - HTML, PNG, SVG, PDF, and more
- **Web-Ready** - Perfect for web applications and dashboards
- **Rich Customization** - Complete control over appearance and behavior
- **Integration** - Works seamlessly with NumPy, pandas, and other libraries
- **Cross-Platform** - Consistent output across different systems and browsers

## Plotly Express vs Graph Objects

- **Plotly Express (px)** - High-level interface for quick plotting
- **Graph Objects (go)** - Low-level interface for complete customization
- **Choose px for** - Quick exploration and standard plots
- **Choose go for** - Complex layouts and custom features

## Interactive Features

- **Zoom and Pan** - Navigate through data with mouse and touch
- **Hover Information** - Display detailed data on hover
- **Selection Tools** - Select and highlight data points
- **Range Sliders** - Filter data by time or value ranges
- **Buttons and Dropdowns** - Control plot behavior and appearance

## Output Formats

- **HTML** - Interactive web-based visualizations
- **PNG/SVG** - Static images for publications
- **PDF** - Vector format for documents
- **Embedded** - Integrate into web pages and applications
- **Dashboards** - Complete web applications

## Running the Tutorials

### Python Script (Recommended for Basics)
```bash
# Run the comprehensive basics tutorial
python plotly_basics.py

# This will create an 'output' directory with all generated plots
```

### Jupyter Notebooks (For Interactive Learning)
```bash
# Start Jupyter Lab
jupyter lab

# Or start Jupyter Notebook
jupyter notebook
```

### Dash Applications
```bash
# Run a Dash application
python app.py

# Access at http://localhost:8050
```

## Resources

- [Plotly Python Documentation](https://plotly.com/python/)
- [Plotly Express Reference](https://plotly.com/python-api-reference/plotly.express.html)
- [Graph Objects Reference](https://plotly.com/python-api-reference/plotly.graph_objects.html)
- [Dash Documentation](https://dash.plotly.com/)
- [Plotly Community Forum](https://community.plotly.com/)

## Support

For issues and questions:
- Check the individual tutorial README files
- Refer to Plotly documentation
- Open an issue on GitHub

---

**Happy Interactive Plotting!** 📊✨ 