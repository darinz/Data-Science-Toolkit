#!/usr/bin/env python3
"""
Matplotlib Basics: A Comprehensive Introduction

Welcome to the Matplotlib basics tutorial! Matplotlib is the most popular plotting 
library for Python, providing a comprehensive set of tools for creating static, 
animated, and interactive visualizations.

This script covers:
- Understanding Matplotlib's architecture (Figure, Axes, Artists)
- Creating basic plots (line plots, scatter plots, bar charts)
- Customizing plot appearance (colors, styles, labels)
- Working with subplots and complex layouts
- Saving and exporting plots
- Best practices for data visualization

Prerequisites:
- Python 3.8 or higher
- Basic understanding of Python programming
- NumPy (covered in NumPy tutorials)
- Basic statistics and data concepts
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import os

# Configure Matplotlib for better output
plt.style.use('default')
mpl.rcParams['figure.figsize'] = (10, 6)
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3

# Set random seed for reproducibility
np.random.seed(42)

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_subsection_header(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")

def main():
    """Main function to run all tutorial sections."""
    
    print("Matplotlib Basics Tutorial")
    print("=" * 60)
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"Matplotlib version: {mpl.__version__}")
    print("All libraries imported successfully!")

    # Section 1: Understanding Matplotlib Architecture
    print_section_header("1. Understanding Matplotlib Architecture")
    
    print("""
Matplotlib has a hierarchical structure:

- Figure: The top-level container that holds everything
- Axes: The area where plots are drawn (contains the plot, labels, etc.)
- Artists: The objects that are drawn (lines, text, patches, etc.)
""")

    # Create a simple figure and axes
    fig, ax = plt.subplots()

    print(f"Figure object: {fig}")
    print(f"Axes object: {ax}")
    print(f"Figure size: {fig.get_size_inches()}")
    print(f"Number of axes: {len(fig.axes)}")

    # Add some data to the axes
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    ax.plot(x, y, label='sin(x)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Simple Sine Wave')
    ax.legend()

    plt.savefig('output/simple_sine_wave.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved simple_sine_wave.png")
    print(f"Figure children: {fig.get_children()}")
    print(f"Axes children: {ax.get_children()}")

    # Section 2: Basic Plotting Functions
    print_section_header("2. Basic Plotting Functions")
    
    print("""
Matplotlib provides several functions for creating different types of plots. 
Let's explore the most common ones.
""")

    # Generate sample data
    x = np.linspace(0, 10, 50)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.exp(-x/3)

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Line plot
    axes[0, 0].plot(x, y1, 'b-', linewidth=2, label='sin(x)')
    axes[0, 0].plot(x, y2, 'r--', linewidth=2, label='cos(x)')
    axes[0, 0].set_title('Line Plot')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Scatter plot
    noise = np.random.normal(0, 0.1, len(x))
    axes[0, 1].scatter(x, y1 + noise, alpha=0.6, s=30, label='sin(x) + noise')
    axes[0, 1].scatter(x, y2 + noise, alpha=0.6, s=30, label='cos(x) + noise')
    axes[0, 1].set_title('Scatter Plot')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].legend()

    # 3. Bar plot
    categories = ['A', 'B', 'C', 'D', 'E']
    values = np.random.randint(1, 20, len(categories))
    axes[1, 0].bar(categories, values, color=['red', 'blue', 'green', 'orange', 'purple'])
    axes[1, 0].set_title('Bar Plot')
    axes[1, 0].set_xlabel('Categories')
    axes[1, 0].set_ylabel('Values')

    # 4. Histogram
    data = np.random.normal(0, 1, 1000)
    axes[1, 1].hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].set_title('Histogram')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig('output/basic_plot_types.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved basic_plot_types.png")

    # Section 3: Customizing Plot Appearance
    print_section_header("3. Customizing Plot Appearance")
    
    print("""
Matplotlib offers extensive customization options. Let's explore some key aspects.
""")

    # Create data for demonstration
    x = np.linspace(0, 4*np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # Create a figure with custom styling
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot with custom styling
    ax.plot(x, y1, color='#FF6B6B', linewidth=3, linestyle='-', 
            marker='o', markersize=6, markerfacecolor='white', 
            markeredgecolor='#FF6B6B', markeredgewidth=2, 
            label='sin(x)', alpha=0.8)

    ax.plot(x, y2, color='#4ECDC4', linewidth=3, linestyle='--', 
            marker='s', markersize=6, markerfacecolor='white', 
            markeredgecolor='#4ECDC4', markeredgewidth=2, 
            label='cos(x)', alpha=0.8)

    # Customize axes
    ax.set_xlabel('x (radians)', fontsize=14, fontweight='bold')
    ax.set_ylabel('y', fontsize=14, fontweight='bold')
    ax.set_title('Customized Trigonometric Functions', fontsize=16, fontweight='bold', pad=20)

    # Customize grid
    ax.grid(True, linestyle='--', alpha=0.7, color='gray')

    # Customize legend
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, 
              loc='upper right', bbox_to_anchor=(1, 1))

    # Set axis limits
    ax.set_xlim(0, 4*np.pi)
    ax.set_ylim(-1.2, 1.2)

    # Add text annotation
    ax.text(np.pi/2, 0.8, 'Peak of sin(x)', fontsize=12, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    # Customize spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('darkblue')

    plt.tight_layout()
    plt.savefig('output/customized_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved customized_plot.png")

    print("\nAvailable colors:")
    print(plt.colormaps())

    print("\nAvailable line styles:")
    print(['-', '--', '-.', ':', 'None'])

    print("\nAvailable markers:")
    print(['o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h', 'H', '+', 'x', '|', '_'])

    # Section 4: Working with Subplots
    print_section_header("4. Working with Subplots")
    
    print("""
Subplots allow you to create multiple plots in a single figure. 
Let's explore different ways to create them.
""")

    # Method 1: Using plt.subplots()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Generate data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)
    y4 = np.exp(-x/5)
    y5 = np.log(x + 1)
    y6 = x**2

    # Plot 1: Sine function
    axes[0, 0].plot(x, y1, 'b-', linewidth=2)
    axes[0, 0].set_title('sin(x)')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')

    # Plot 2: Cosine function
    axes[0, 1].plot(x, y2, 'r-', linewidth=2)
    axes[0, 1].set_title('cos(x)')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')

    # Plot 3: Tangent function (with limits)
    axes[0, 2].plot(x, y3, 'g-', linewidth=2)
    axes[0, 2].set_title('tan(x)')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    axes[0, 2].set_ylim(-5, 5)

    # Plot 4: Exponential decay
    axes[1, 0].plot(x, y4, 'm-', linewidth=2)
    axes[1, 0].set_title('exp(-x/5)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')

    # Plot 5: Logarithm
    axes[1, 1].plot(x, y5, 'c-', linewidth=2)
    axes[1, 1].set_title('log(x+1)')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')

    # Plot 6: Quadratic function
    axes[1, 2].plot(x, y6, 'orange', linewidth=2)
    axes[1, 2].set_title('x²')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')

    plt.tight_layout()
    plt.savefig('output/subplots_grid.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved subplots_grid.png")

    # Method 2: Using GridSpec for more complex layouts
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(3, 3, figure=fig)

    # Large plot spanning 2x2
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.plot(x, y1, 'b-', linewidth=3, label='sin(x)')
    ax1.plot(x, y2, 'r--', linewidth=3, label='cos(x)')
    ax1.set_title('Main Plot: Trigonometric Functions', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Small plot in top right
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(x, y4, 'g-', linewidth=2)
    ax2.set_title('Decay')

    # Small plot in middle right
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(x, y5, 'm-', linewidth=2)
    ax3.set_title('Log')

    # Wide plot at bottom
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(x, y6, 'orange', linewidth=2)
    ax4.set_title('Quadratic Function')
    ax4.set_xlabel('x')

    plt.tight_layout()
    plt.savefig('output/gridspec_layout.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved gridspec_layout.png")

    # Section 5: Statistical Plots
    print_section_header("5. Statistical Plots")
    
    print("""
Matplotlib is excellent for creating statistical visualizations. 
Let's explore some common types.
""")

    # Generate sample data
    np.random.seed(42)
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(2, 1.5, 1000)
    data3 = np.random.normal(-1, 0.8, 1000)

    # Create statistical plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Histogram
    axes[0, 0].hist(data1, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Histogram')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')

    # 2. Box plot
    box_data = [data1, data2, data3]
    axes[0, 1].boxplot(box_data, labels=['Group 1', 'Group 2', 'Group 3'])
    axes[0, 1].set_title('Box Plot')
    axes[0, 1].set_ylabel('Value')

    # 3. Violin plot
    axes[0, 2].violinplot(box_data)
    axes[0, 2].set_title('Violin Plot')
    axes[0, 2].set_ylabel('Value')
    axes[0, 2].set_xticks([1, 2, 3])
    axes[0, 2].set_xticklabels(['Group 1', 'Group 2', 'Group 3'])

    # 4. Scatter plot with correlation
    x_scatter = np.random.normal(0, 1, 100)
    y_scatter = 0.7 * x_scatter + np.random.normal(0, 0.5, 100)
    axes[1, 0].scatter(x_scatter, y_scatter, alpha=0.6, s=50)
    axes[1, 0].set_title('Scatter Plot')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')

    # Add trend line
    z = np.polyfit(x_scatter, y_scatter, 1)
    p = np.poly1d(z)
    axes[1, 0].plot(x_scatter, p(x_scatter), "r--", alpha=0.8)

    # 5. Cumulative distribution
    axes[1, 1].hist(data1, bins=50, cumulative=True, density=True, 
                    alpha=0.7, color='green', edgecolor='black')
    axes[1, 1].set_title('Cumulative Distribution')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Cumulative Probability')

    # 6. Error bar plot
    categories = ['A', 'B', 'C', 'D']
    means = [2.5, 3.2, 1.8, 4.1]
    errors = [0.3, 0.4, 0.2, 0.5]
    axes[1, 2].errorbar(categories, means, yerr=errors, fmt='o', 
                       capsize=5, capthick=2, markersize=8)
    axes[1, 2].set_title('Error Bar Plot')
    axes[1, 2].set_ylabel('Value')

    plt.tight_layout()
    plt.savefig('output/statistical_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved statistical_plots.png")

    # Additional statistical plot: Heatmap
    correlation_data = np.random.rand(5, 5)
    correlation_data = (correlation_data + correlation_data.T) / 2  # Make symmetric
    np.fill_diagonal(correlation_data, 1)  # Diagonal should be 1

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(correlation_data, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(['Var1', 'Var2', 'Var3', 'Var4', 'Var5'])
    ax.set_yticklabels(['Var1', 'Var2', 'Var3', 'Var4', 'Var5'])

    # Add text annotations
    for i in range(5):
        for j in range(5):
            text = ax.text(j, i, f'{correlation_data[i, j]:.2f}',
                           ha="center", va="center", color="black")

    ax.set_title('Correlation Matrix Heatmap')
    plt.colorbar(im)
    plt.savefig('output/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved correlation_heatmap.png")

    # Section 6: Saving and Exporting Plots
    print_section_header("6. Saving and Exporting Plots")
    
    print("""
Matplotlib supports saving plots in various formats. Let's explore the options.
""")

    # Create a sample plot to save
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    ax.plot(x, y1, 'b-', linewidth=2, label='sin(x)')
    ax.plot(x, y2, 'r--', linewidth=2, label='cos(x)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Sample Plot for Export')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/sample_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saving plots in different formats...")

    # PNG format (raster, good for web)
    plt.savefig('output/sample_plot.png', dpi=300, bbox_inches='tight')
    print("Saved as sample_plot.png")

    # PDF format (vector, good for publications)
    plt.savefig('output/sample_plot.pdf', bbox_inches='tight')
    print("Saved as sample_plot.pdf")

    # SVG format (vector, good for web and editing)
    plt.savefig('output/sample_plot.svg', bbox_inches='tight')
    print("Saved as sample_plot.svg")

    # High-resolution PNG for printing
    plt.savefig('output/sample_plot_highres.png', dpi=600, bbox_inches='tight')
    print("Saved as sample_plot_highres.png")

    print("\nFile formats and their uses:")
    print("• PNG: Raster format, good for web and presentations")
    print("• PDF: Vector format, excellent for publications")
    print("• SVG: Vector format, good for web and further editing")
    print("• EPS: Encapsulated PostScript, for LaTeX documents")
    print("• JPG: Compressed raster format, smaller file sizes")

    # Section 7: Working with Real Data
    print_section_header("7. Working with Real Data")
    
    print("""
Let's work with some real data to demonstrate practical plotting.
""")

    # Create sample time series data
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    np.random.seed(42)

    # Generate realistic time series data
    trend = np.linspace(100, 120, 365)  # Upward trend
    seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365)  # Seasonal pattern
    noise = np.random.normal(0, 5, 365)  # Random noise
    values = trend + seasonal + noise

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'trend': trend,
        'seasonal': seasonal
    })

    print("Sample data:")
    print(df.head())
    print(f"\nData shape: {df.shape}")

    # Create comprehensive time series plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Time series plot
    axes[0, 0].plot(df['date'], df['value'], 'b-', alpha=0.7, linewidth=1, label='Actual')
    axes[0, 0].plot(df['date'], df['trend'], 'r--', linewidth=2, label='Trend')
    axes[0, 0].set_title('Time Series with Trend')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)

    # 2. Seasonal decomposition
    axes[0, 1].plot(df['date'], df['seasonal'], 'g-', linewidth=2)
    axes[0, 1].set_title('Seasonal Component')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Seasonal Value')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Distribution of values
    axes[1, 0].hist(df['value'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_title('Distribution of Values')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')

    # 4. Monthly averages
    monthly_avg = df.groupby(df['date'].dt.month)['value'].mean()
    axes[1, 1].bar(monthly_avg.index, monthly_avg.values, alpha=0.7, color='orange')
    axes[1, 1].set_title('Monthly Averages')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Average Value')
    axes[1, 1].set_xticks(range(1, 13))

    plt.tight_layout()
    plt.savefig('output/time_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved time_series_analysis.png")

    # Additional analysis: Rolling statistics
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Rolling mean and standard deviation
    rolling_mean = df['value'].rolling(window=30).mean()
    rolling_std = df['value'].rolling(window=30).std()

    axes[0].plot(df['date'], df['value'], 'b-', alpha=0.5, label='Original')
    axes[0].plot(df['date'], rolling_mean, 'r-', linewidth=2, label='30-day Rolling Mean')
    axes[0].fill_between(df['date'], 
                         rolling_mean - rolling_std, 
                         rolling_mean + rolling_std, 
                         alpha=0.3, color='red', label='±1 Std Dev')
    axes[0].set_title('Time Series with Rolling Statistics')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)

    # Rolling correlation with trend
    rolling_corr = df['value'].rolling(window=30).corr(pd.Series(trend))
    axes[1].plot(df['date'], rolling_corr, 'g-', linewidth=2)
    axes[1].set_title('Rolling Correlation with Trend')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Correlation')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('output/rolling_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved rolling_statistics.png")

    # Section 8: Best Practices and Tips
    print_section_header("8. Best Practices and Tips")
    
    print("""
Here are some best practices for creating effective visualizations.
""")

    # Demonstrate best practices
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Good practice: Clear labels and title
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    axes[0, 0].plot(x, y, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (seconds)', fontsize=12)
    axes[0, 0].set_ylabel('Amplitude', fontsize=12)
    axes[0, 0].set_title('Sine Wave Function', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Good practice: Appropriate color scheme
    categories = ['A', 'B', 'C', 'D']
    values = [25, 40, 30, 35]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    axes[0, 1].bar(categories, values, color=colors, alpha=0.8)
    axes[0, 1].set_xlabel('Categories', fontsize=12)
    axes[0, 1].set_ylabel('Values', fontsize=12)
    axes[0, 1].set_title('Bar Chart with Good Colors', fontsize=14, fontweight='bold')

    # Good practice: Multiple data series with legend
    x = np.linspace(0, 5, 100)
    y1 = np.exp(-x)
    y2 = np.exp(-x/2)
    y3 = np.exp(-x/3)
    axes[1, 0].plot(x, y1, 'r-', linewidth=2, label='τ=1')
    axes[1, 0].plot(x, y2, 'g--', linewidth=2, label='τ=2')
    axes[1, 0].plot(x, y3, 'b:', linewidth=2, label='τ=3')
    axes[1, 0].set_xlabel('Time', fontsize=12)
    axes[1, 0].set_ylabel('Amplitude', fontsize=12)
    axes[1, 0].set_title('Exponential Decay Functions', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Good practice: Error bars and confidence intervals
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2.1, 3.8, 7.2, 13.5, 26.0])
    yerr = np.array([0.3, 0.5, 0.8, 1.2, 2.0])
    axes[1, 1].errorbar(x, y, yerr=yerr, fmt='o-', capsize=5, capthick=2, 
                       markersize=8, linewidth=2, color='purple')
    axes[1, 1].set_xlabel('X Values', fontsize=12)
    axes[1, 1].set_ylabel('Y Values', fontsize=12)
    axes[1, 1].set_title('Data with Error Bars', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/best_practices.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved best_practices.png")

    print("\nBest Practices Summary:")
    print("1. Always include clear labels and titles")
    print("2. Use appropriate color schemes and avoid colorblind-unfriendly combinations")
    print("3. Include legends for multiple data series")
    print("4. Show uncertainty with error bars or confidence intervals")
    print("5. Choose appropriate plot types for your data")
    print("6. Use consistent formatting across related plots")
    print("7. Consider your audience and the message you want to convey")
    print("8. Test your plots in different formats (print, screen, etc.)")

    # Section 9: Summary and Next Steps
    print_section_header("9. Summary and Next Steps")
    
    print("""
Congratulations! You've completed the Matplotlib basics tutorial. Here's what you've learned:

Key Concepts Covered:
✅ Matplotlib Architecture: Understanding Figure, Axes, and Artists
✅ Basic Plotting: Line plots, scatter plots, bar charts, histograms
✅ Customization: Colors, styles, labels, legends, and grids
✅ Subplots: Creating multi-panel figures with different layouts
✅ Statistical Plots: Box plots, violin plots, heatmaps, and more
✅ Data Export: Saving plots in various formats
✅ Real Data: Working with time series and practical datasets
✅ Best Practices: Creating effective and professional visualizations

Next Steps:

1. Advanced Plot Types: Explore 3D plotting, geographic plots, and animations
2. Interactive Plots: Learn about Plotly, Bokeh, and interactive Matplotlib
3. Seaborn Integration: Combine Matplotlib with Seaborn for statistical plots
4. Publication Quality: Master techniques for creating publication-ready figures
5. Custom Styling: Create your own themes and color schemes

Additional Resources:
- Matplotlib Official Documentation: https://matplotlib.org/
- Matplotlib Tutorials: https://matplotlib.org/stable/tutorials/index.html
- Matplotlib Examples: https://matplotlib.org/stable/gallery/index.html
- Matplotlib Cheat Sheet: https://matplotlib.org/cheatsheets/

Practice Exercises:
1. Create a multi-panel figure showing different aspects of a dataset
2. Design a custom color scheme for a specific type of data
3. Create an animated plot showing data evolution over time
4. Build a dashboard-style visualization with multiple related plots

Happy plotting! 📊✨
""")

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('output', exist_ok=True)
    
    # Run the tutorial
    main()
    
    print("\n" + "="*60)
    print(" Tutorial completed successfully!")
    print(" Check the 'output' directory for generated plots.")
    print("="*60) 