# Seaborn Tutorials

A comprehensive guide to Seaborn, the statistical data visualization library built on top of Matplotlib, covering beautiful and informative statistical graphics.

## What's Included

### Seaborn Fundamentals
- **Basic Plotting** - Statistical plots, distribution plots, and categorical plots
- **Figure Aesthetics** - Understanding Seaborn's beautiful default styles
- **Color Palettes** - Built-in color schemes and custom palettes
- **Statistical Relationships** - Regression plots and correlation analysis
- **Categorical Data** - Bar plots, box plots, and violin plots

### Advanced Visualization
- **Distribution Plots** - Histograms, KDE plots, and rug plots
- **Regression Analysis** - Linear and non-linear regression visualization
- **Categorical Analysis** - Count plots, point plots, and factor plots
- **Matrix Plots** - Heatmaps and clustermaps
- **Multi-Plot Grids** - FacetGrid and PairGrid for complex visualizations

### Data Science Applications
- **Exploratory Data Analysis** - Quick statistical insights and patterns
- **Model Evaluation** - Visualizing model performance and residuals
- **Statistical Testing** - Visual representation of statistical tests
- **Time Series Analysis** - Temporal data visualization
- **Correlation Analysis** - Relationship discovery and visualization

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Basic understanding of Python programming
- Familiarity with NumPy and pandas (covered in respective tutorials)
- Basic knowledge of Matplotlib (covered in Matplotlib tutorials)

### Installation

**Using Conda (Recommended):**
```bash
cd Seaborn
conda env create -f environment.yml
conda activate seaborn-tutorials
```

**Using pip:**
```bash
cd Seaborn
pip install -r requirements.txt
```

### Verify Installation
```python
import seaborn as sns
import matplotlib.pyplot as plt
print(f"Seaborn version: {sns.__version__}")
```

## Tutorial Structure

### 1. Seaborn Basics (`seaborn_basics_guide.md`)
- Introduction to Seaborn and its philosophy
- Basic plotting functions and syntax
- Figure aesthetics and styling
- Integration with pandas DataFrames
- **Run with:** `python seaborn_basics.py`

### 2. Statistical Plots (`statistical_plots_guide.md`)
- Distribution plots (histogram, KDE, rug)
- Regression plots and trend analysis
- Residual plots and model diagnostics
- Statistical annotations and significance

### 3. Categorical Plots (`categorical_plots_guide.md`)
- Bar plots and count plots
- Box plots and violin plots
- Point plots and factor plots
- Swarm plots and strip plots

### 4. Distribution Analysis (`distribution_analysis_guide.md`)
- Univariate and bivariate distributions
- Joint plots and pair plots
- Marginal distributions
- Distribution comparisons

### 5. Correlation Analysis (`correlation_analysis_guide.md`)
- Correlation matrices and heatmaps
- Pair plots for relationship discovery
- Clustermaps for hierarchical clustering
- Correlation significance testing

### 6. Multi-Plot Grids (`multi_plot_grids_guide.md`)
- FacetGrid for conditional plotting
- PairGrid for pairwise relationships
- JointGrid for joint distributions
- Complex multi-panel layouts

### 7. Advanced Features (`advanced_features_guide.md`)
- Custom color palettes and themes
- Statistical annotations and significance
- Integration with statistical libraries
- Publication-quality output

### 8. Real-World Applications (`real_world_applications_guide.md`)
- Data science workflow integration
- Model evaluation and comparison
- Business intelligence dashboards
- Scientific research visualization

### 9. Quick Tutorial (`seaborn_ultraquick_tutorial.ipynb`)
- Interactive Jupyter notebook
- Essential Seaborn functions
- Common use cases and examples
- Best practices and tips

## Learning Path

1. **Start with Seaborn Basics** - Understand the library philosophy and basic syntax
2. **Learn Statistical Plots** - Master distribution and regression visualization
3. **Explore Categorical Data** - Visualize categorical variables effectively
4. **Analyze Distributions** - Understand data distributions and relationships
5. **Discover Correlations** - Find patterns and relationships in data
6. **Create Multi-Plot Grids** - Build complex, informative visualizations
7. **Master Advanced Features** - Customize and enhance your plots
8. **Apply to Real Problems** - Use Seaborn in data science workflows

## Key Features

- **Statistical Focus** - Built for statistical data visualization
- **Beautiful Defaults** - Attractive plots with minimal configuration
- **Pandas Integration** - Seamless work with pandas DataFrames
- **Statistical Annotations** - Built-in statistical testing and annotations
- **Flexible Styling** - Easy customization and theming

## Style and Themes

- **Default Style** - Clean, modern statistical graphics
- **Color Palettes** - Built-in palettes for different data types
- **Context-Sensitive** - Automatic styling based on plot type
- **Publication Ready** - High-quality output for papers and presentations

## Plot Categories

### Distribution Plots
- `distplot()` - Histogram with KDE
- `kdeplot()` - Kernel density estimation
- `rugplot()` - Marginal distributions
- `jointplot()` - Joint and marginal distributions

### Categorical Plots
- `catplot()` - Categorical plotting interface
- `boxplot()` - Box and whisker plots
- `violinplot()` - Violin plots
- `stripplot()` - Strip plots
- `swarmplot()` - Swarm plots

### Regression Plots
- `regplot()` - Linear regression plots
- `lmplot()` - Faceted regression plots
- `residplot()` - Residual plots

### Matrix Plots
- `heatmap()` - Correlation heatmaps
- `clustermap()` - Hierarchical clustering heatmaps

## Running the Tutorials

### Python Script (Recommended for Basics)
```bash
# Run the comprehensive basics tutorial
python seaborn_basics.py

# This will create an 'output' directory with all generated plots
```

### Jupyter Notebooks (For Interactive Learning)
```bash
# Start Jupyter Lab
jupyter lab

# Or start Jupyter Notebook
jupyter notebook
```

## Best Practices

- **Use Appropriate Plot Types** - Choose plots that match your data and question
- **Leverage Default Styles** - Seaborn's defaults are often optimal
- **Consider Your Audience** - Adjust complexity based on who will view the plots
- **Document Your Choices** - Explain why you chose specific visualizations
- **Test Different Views** - Try multiple plot types to find the best representation

## Integration with Other Libraries

- **Pandas** - Direct DataFrame plotting
- **NumPy** - Array-based data visualization
- **Matplotlib** - Fine-grained customization
- **SciPy** - Statistical testing and analysis
- **Scikit-learn** - Model evaluation and comparison

## Resources

- [Seaborn Official Documentation](https://seaborn.pydata.org/)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)
- [Seaborn API Reference](https://seaborn.pydata.org/api.html)

## Support

For issues and questions:
- Check the individual tutorial README files
- Refer to Seaborn documentation
- Open an issue on GitHub

---

**Happy Statistical Visualization!** 