# Advanced Features Guide

A comprehensive guide to advanced Seaborn features, including custom color palettes, statistical annotations, integration with other libraries, and publication-quality output.

## Table of Contents

1. [Custom Color Palettes](#custom-color-palettes)
2. [Statistical Annotations](#statistical-annotations)
3. [Integration with Statistical Libraries](#integration-with-statistical-libraries)
4. [Publication Quality Output](#publication-quality-output)
5. [Custom Plot Functions](#custom-plot-functions)
6. [Advanced Styling](#advanced-styling)

## Custom Color Palettes

### Creating Custom Palettes

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Set up the plotting style
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)

# Create custom color palettes
custom_palette_1 = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
custom_palette_2 = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#592E83"]
custom_palette_3 = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"]

# Generate sample data
np.random.seed(42)
data = []
for i, category in enumerate(['A', 'B', 'C', 'D', 'E']):
    for _ in range(50):
        data.append({
            'category': category,
            'value': np.random.normal(i, 1),
            'group': np.random.choice(['Group 1', 'Group 2'])
        })
df = pd.DataFrame(data)

# Create plots with custom palettes
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Custom palette 1
sns.boxplot(data=df, x='category', y='value', palette=custom_palette_1, ax=axes[0])
axes[0].set_title('Custom Palette 1')

# Plot 2: Custom palette 2
sns.boxplot(data=df, x='category', y='value', palette=custom_palette_2, ax=axes[1])
axes[1].set_title('Custom Palette 2')

# Plot 3: Custom palette 3
sns.boxplot(data=df, x='category', y='value', palette=custom_palette_3, ax=axes[2])
axes[2].set_title('Custom Palette 3')

plt.suptitle('Custom Color Palettes', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Diverging and Sequential Palettes

```python
# Create diverging and sequential palettes
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Diverging palettes
sns.color_palette("RdBu_r", 10)
sns.boxplot(data=df, x='category', y='value', palette="RdBu_r", ax=axes[0, 0])
axes[0, 0].set_title('Diverging: RdBu_r')

sns.color_palette("coolwarm", 10)
sns.boxplot(data=df, x='category', y='value', palette="coolwarm", ax=axes[0, 1])
axes[0, 1].set_title('Diverging: coolwarm')

sns.color_palette("PiYG", 10)
sns.boxplot(data=df, x='category', y='value', palette="PiYG", ax=axes[0, 2])
axes[0, 2].set_title('Diverging: PiYG')

# Sequential palettes
sns.color_palette("Blues", 10)
sns.boxplot(data=df, x='category', y='value', palette="Blues", ax=axes[1, 0])
axes[1, 0].set_title('Sequential: Blues')

sns.color_palette("Greens", 10)
sns.boxplot(data=df, x='category', y='value', palette="Greens", ax=axes[1, 1])
axes[1, 1].set_title('Sequential: Greens')

sns.color_palette("Oranges", 10)
sns.boxplot(data=df, x='category', y='value', palette="Oranges", ax=axes[1, 2])
axes[1, 2].set_title('Sequential: Oranges')

plt.suptitle('Diverging and Sequential Palettes', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Context-Sensitive Palettes

```python
# Create context-sensitive palettes
def create_context_palette(data, n_colors=5):
    """Create palette based on data characteristics"""
    if data.min() < 0 and data.max() > 0:
        # Data spans negative and positive values - use diverging
        return sns.color_palette("RdBu_r", n_colors)
    elif data.min() >= 0:
        # All positive data - use sequential
        return sns.color_palette("Blues", n_colors)
    else:
        # All negative data - use sequential
        return sns.color_palette("Reds_r", n_colors)

# Generate different types of data
np.random.seed(42)
positive_data = np.random.exponential(1, 100)
negative_data = -np.random.exponential(1, 100)
mixed_data = np.random.normal(0, 1, 100)

# Create plots with context-sensitive palettes
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Positive data
palette_pos = create_context_palette(positive_data)
sns.histplot(data=positive_data, kde=True, ax=axes[0], palette=palette_pos)
axes[0].set_title('Positive Data - Sequential Palette')

# Negative data
palette_neg = create_context_palette(negative_data)
sns.histplot(data=negative_data, kde=True, ax=axes[1], palette=palette_neg)
axes[1].set_title('Negative Data - Sequential Palette')

# Mixed data
palette_mix = create_context_palette(mixed_data)
sns.histplot(data=mixed_data, kde=True, ax=axes[2], palette=palette_mix)
axes[2].set_title('Mixed Data - Diverging Palette')

plt.suptitle('Context-Sensitive Palettes', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Statistical Annotations

### Basic Statistical Annotations

```python
# Create basic statistical annotations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Correlation with annotation
x = np.random.randn(100)
y = 0.7 * x + np.random.randn(100) * 0.5
corr = np.corrcoef(x, y)[0, 1]

sns.scatterplot(x=x, y=y, alpha=0.6, ax=axes[0, 0])
axes[0, 0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[0, 0].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
axes[0, 0].set_title('Correlation Annotation')

# Plot 2: Mean and standard deviation
data = np.random.normal(0, 1, 1000)
mean_val = np.mean(data)
std_val = np.std(data)

sns.histplot(data=data, kde=True, ax=axes[0, 1])
axes[0, 1].axvline(mean_val, color='red', linestyle='--', alpha=0.8)
axes[0, 1].text(0.05, 0.95, f'μ = {mean_val:.2f}\nσ = {std_val:.2f}', 
                transform=axes[0, 1].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
axes[0, 1].set_title('Mean and Standard Deviation')

# Plot 3: T-test results
group1 = np.random.normal(0, 1, 50)
group2 = np.random.normal(1, 1, 50)
t_stat, p_val = stats.ttest_ind(group1, group2)

sns.boxplot(data=pd.DataFrame({
    'value': np.concatenate([group1, group2]),
    'group': ['Group 1'] * 50 + ['Group 2'] * 50
}), x='group', y='value', ax=axes[1, 0])

if p_val < 0.001:
    sig_text = 'p < 0.001'
elif p_val < 0.01:
    sig_text = 'p < 0.01'
elif p_val < 0.05:
    sig_text = 'p < 0.05'
else:
    sig_text = f'p = {p_val:.3f}'

axes[1, 0].text(0.5, 0.95, f't-test: {sig_text}', 
                transform=axes[1, 0].transAxes, ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
axes[1, 0].set_title('T-Test Results')

# Plot 4: R-squared for regression
sns.regplot(x=x, y=y, scatter_kws={'alpha': 0.6}, ax=axes[1, 1])
r_squared = corr ** 2
axes[1, 1].text(0.05, 0.95, f'R² = {r_squared:.3f}', 
                transform=axes[1, 1].transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
axes[1, 1].set_title('R-Squared Annotation')

plt.suptitle('Basic Statistical Annotations', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Advanced Statistical Annotations

```python
# Create advanced statistical annotations
def add_statistical_annotations(ax, data_dict, test_type='t_test'):
    """Add comprehensive statistical annotations to plot"""
    
    if test_type == 't_test':
        # Perform t-test
        groups = list(data_dict.keys())
        group1_data = data_dict[groups[0]]
        group2_data = data_dict[groups[1]]
        
        t_stat, p_val = stats.ttest_ind(group1_data, group2_data)
        
        # Add significance stars
        if p_val < 0.001:
            stars = '***'
        elif p_val < 0.01:
            stars = '**'
        elif p_val < 0.05:
            stars = '*'
        else:
            stars = 'ns'
        
        # Add annotation
        ax.text(0.5, 0.95, f't({len(group1_data)+len(group2_data)-2}) = {t_stat:.2f}\np = {p_val:.3f} {stars}', 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    elif test_type == 'anova':
        # Perform one-way ANOVA
        groups = list(data_dict.keys())
        f_stat, p_val = stats.f_oneway(*data_dict.values())
        
        ax.text(0.5, 0.95, f'F({len(groups)-1},{sum(len(v) for v in data_dict.values())-len(groups)}) = {f_stat:.2f}\np = {p_val:.3f}', 
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))

# Generate data for different groups
np.random.seed(42)
group_data = {
    'Group A': np.random.normal(0, 1, 30),
    'Group B': np.random.normal(1, 1, 30),
    'Group C': np.random.normal(0.5, 1.2, 30)
}

# Create plots with advanced annotations
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# T-test between two groups
sns.boxplot(data=pd.DataFrame({
    'value': np.concatenate([group_data['Group A'], group_data['Group B']]),
    'group': ['Group A'] * 30 + ['Group B'] * 30
}), x='group', y='value', ax=axes[0])

add_statistical_annotations(axes[0], 
                           {'Group A': group_data['Group A'], 'Group B': group_data['Group B']}, 
                           't_test')
axes[0].set_title('T-Test Between Two Groups')

# ANOVA between three groups
sns.boxplot(data=pd.DataFrame({
    'value': np.concatenate([group_data['Group A'], group_data['Group B'], group_data['Group C']]),
    'group': ['Group A'] * 30 + ['Group B'] * 30 + ['Group C'] * 30
}), x='group', y='value', ax=axes[1])

add_statistical_annotations(axes[1], group_data, 'anova')
axes[1].set_title('ANOVA Between Three Groups')

plt.suptitle('Advanced Statistical Annotations', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Integration with Statistical Libraries

### Integration with SciPy

```python
# Integration with SciPy for advanced statistical analysis
from scipy.stats import pearsonr, spearmanr, kendalltau, mannwhitneyu, wilcoxon

# Generate sample data
np.random.seed(42)
x = np.random.randn(100)
y = 0.7 * x + np.random.randn(100) * 0.5

# Calculate different correlation coefficients
pearson_r, pearson_p = pearsonr(x, y)
spearman_r, spearman_p = spearmanr(x, y)
kendall_tau, kendall_p = kendalltau(x, y)

# Create correlation comparison plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Pearson correlation
sns.regplot(x=x, y=y, scatter_kws={'alpha': 0.6}, ax=axes[0])
axes[0].text(0.05, 0.95, f'Pearson: r = {pearson_r:.3f}\np = {pearson_p:.3f}', 
             transform=axes[0].transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
axes[0].set_title('Pearson Correlation')

# Spearman correlation
sns.regplot(x=x, y=y, scatter_kws={'alpha': 0.6}, ax=axes[1])
axes[1].text(0.05, 0.95, f'Spearman: ρ = {spearman_r:.3f}\np = {spearman_p:.3f}', 
             transform=axes[1].transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
axes[1].set_title('Spearman Correlation')

# Kendall correlation
sns.regplot(x=x, y=y, scatter_kws={'alpha': 0.6}, ax=axes[2])
axes[2].text(0.05, 0.95, f'Kendall: τ = {kendall_tau:.3f}\np = {kendall_p:.3f}', 
             transform=axes[2].transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
axes[2].set_title('Kendall Correlation')

plt.suptitle('Correlation Analysis with SciPy', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
```

### Integration with Statsmodels

```python
# Integration with Statsmodels for regression analysis
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Generate sample data
np.random.seed(42)
x1 = np.random.randn(100)
x2 = 0.5 * x1 + np.random.randn(100) * 0.5  # Correlated predictor
y = 2 * x1 + 1.5 * x2 + np.random.randn(100) * 0.5

# Create DataFrame
df_reg = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

# Fit multiple regression
X = sm.add_constant(df_reg[['x1', 'x2']])
model = sm.OLS(df_reg['y'], X).fit()

# Create diagnostic plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Residuals vs fitted
fitted_values = model.fittedvalues
residuals = model.resid

sns.scatterplot(x=fitted_values, y=residuals, alpha=0.6, ax=axes[0, 0])
axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
axes[0, 0].set_title('Residuals vs Fitted')
axes[0, 0].set_xlabel('Fitted Values')
axes[0, 0].set_ylabel('Residuals')

# Q-Q plot
from scipy.stats import probplot
probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot of Residuals')

# Residuals histogram
sns.histplot(data=residuals, kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Residuals Distribution')
axes[1, 0].set_xlabel('Residuals')

# Leverage plot
from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model, ax=axes[1, 1])
axes[1, 1].set_title('Influence Plot')

plt.suptitle('Regression Diagnostics with Statsmodels', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Print model summary
print("Regression Model Summary:")
print(model.summary())
```

## Publication Quality Output

### High-Resolution Output

```python
# Create publication-quality plots
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Generate publication-quality data
np.random.seed(42)
n_samples = 200

# Create comprehensive dataset
data = []
for treatment in ['Control', 'Treatment A', 'Treatment B']:
    for time in ['Baseline', 'Week 4', 'Week 8']:
        for _ in range(n_samples // 6):
            if treatment == 'Control':
                base_value = np.random.normal(50, 10)
            elif treatment == 'Treatment A':
                base_value = np.random.normal(55, 10)
            else:
                base_value = np.random.normal(60, 10)
            
            if time == 'Baseline':
                value = base_value
            elif time == 'Week 4':
                value = base_value + np.random.normal(5, 3)
            else:
                value = base_value + np.random.normal(10, 3)
            
            data.append({
                'treatment': treatment,
                'time': time,
                'value': value
            })

df_pub = pd.DataFrame(data)

# Create publication-quality figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Box plot with individual points
sns.boxplot(data=df_pub, x='treatment', y='value', ax=axes[0, 0])
sns.stripplot(data=df_pub, x='treatment', y='value', color='red', alpha=0.3, size=3, ax=axes[0, 0])
axes[0, 0].set_title('Treatment Effects')
axes[0, 0].set_xlabel('Treatment Group')
axes[0, 0].set_ylabel('Response Value')

# Plot 2: Time series by treatment
sns.lineplot(data=df_pub, x='time', y='value', hue='treatment', ax=axes[0, 1])
axes[0, 1].set_title('Time Course by Treatment')
axes[0, 1].set_xlabel('Time Point')
axes[0, 1].set_ylabel('Response Value')

# Plot 3: Violin plot
sns.violinplot(data=df_pub, x='treatment', y='value', inner='box', ax=axes[1, 0])
axes[1, 0].set_title('Distribution by Treatment')
axes[1, 0].set_xlabel('Treatment Group')
axes[1, 0].set_ylabel('Response Value')

# Plot 4: Heatmap of means
pivot_data = df_pub.groupby(['treatment', 'time'])['value'].mean().unstack()
sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='viridis', ax=axes[1, 1])
axes[1, 1].set_title('Mean Values Heatmap')

plt.suptitle('Publication-Quality Multi-Panel Figure', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save high-resolution figure
plt.savefig('publication_quality_figure.png', dpi=300, bbox_inches='tight')
plt.show()
```

### LaTeX Integration

```python
# Create plots with LaTeX integration
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

# Generate data for LaTeX example
np.random.seed(42)
x = np.linspace(0, 10, 100)
y1 = 2 * x + np.random.normal(0, 1, 100)
y2 = x**2 + np.random.normal(0, 5, 100)

# Create LaTeX-formatted plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Linear relationship
sns.regplot(x=x, y=y1, scatter_kws={'alpha': 0.6}, ax=axes[0])
axes[0].set_title(r'Linear Relationship: $y = 2x + \epsilon$')
axes[0].set_xlabel(r'Independent Variable ($x$)')
axes[0].set_ylabel(r'Dependent Variable ($y$)')

# Quadratic relationship
sns.regplot(x=x, y=y2, order=2, scatter_kws={'alpha': 0.6}, ax=axes[1])
axes[1].set_title(r'Quadratic Relationship: $y = x^2 + \epsilon$')
axes[1].set_xlabel(r'Independent Variable ($x$)')
axes[1].set_ylabel(r'Dependent Variable ($y$)')

plt.suptitle(r'Mathematical Relationships with LaTeX', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Reset to default
plt.rcParams['text.usetex'] = False
plt.rcParams['font.family'] = 'sans-serif'
```

## Custom Plot Functions

### Custom Statistical Plot Function

```python
def create_statistical_summary_plot(data, groups, plot_type='box', test_type='t_test'):
    """Create comprehensive statistical summary plot"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Main plot
    if plot_type == 'box':
        sns.boxplot(data=data, x=groups, y='value', ax=axes[0, 0])
        sns.stripplot(data=data, x=groups, y='value', color='red', alpha=0.3, size=3, ax=axes[0, 0])
    elif plot_type == 'violin':
        sns.violinplot(data=data, x=groups, y='value', inner='box', ax=axes[0, 0])
    
    axes[0, 0].set_title('Main Plot')
    
    # Plot 2: Distribution
    for group in data[groups].unique():
        subset = data[data[groups] == group]['value']
        sns.kdeplot(data=subset, label=group, ax=axes[0, 1])
    axes[0, 1].set_title('Distribution Comparison')
    axes[0, 1].legend()
    
    # Plot 3: Q-Q plot for normality
    for group in data[groups].unique():
        subset = data[data[groups] == group]['value']
        probplot(subset, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot for Normality')
    
    # Plot 4: Statistical test results
    if test_type == 't_test' and len(data[groups].unique()) == 2:
        group1, group2 = data[groups].unique()
        data1 = data[data[groups] == group1]['value']
        data2 = data[data[groups] == group2]['value']
        
        t_stat, p_val = stats.ttest_ind(data1, data2)
        
        axes[1, 1].text(0.5, 0.5, f't-test results:\nt = {t_stat:.3f}\np = {p_val:.3f}', 
                        transform=axes[1, 1].transAxes, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        axes[1, 1].set_title('Statistical Test Results')
        axes[1, 1].axis('off')
    
    plt.suptitle('Comprehensive Statistical Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

# Use custom function
np.random.seed(42)
test_data = pd.DataFrame({
    'value': np.concatenate([
        np.random.normal(0, 1, 50),
        np.random.normal(1, 1, 50)
    ]),
    'group': ['Control'] * 50 + ['Treatment'] * 50
})

fig = create_statistical_summary_plot(test_data, 'group', 'box', 't_test')
plt.show()
```

### Custom Correlation Analysis Function

```python
def create_correlation_analysis(df, variables, method='pearson'):
    """Create comprehensive correlation analysis"""
    
    # Calculate correlation matrix
    corr_matrix = df[variables].corr(method=method)
    
    # Calculate p-values
    p_matrix = pd.DataFrame(index=variables, columns=variables)
    for i in variables:
        for j in variables:
            if i == j:
                p_matrix.loc[i, j] = 1.0
            else:
                if method == 'pearson':
                    r, p = stats.pearsonr(df[i], df[j])
                elif method == 'spearman':
                    r, p = stats.spearmanr(df[i], df[j])
                elif method == 'kendall':
                    r, p = stats.kendalltau(df[i], df[j])
                p_matrix.loc[i, j] = p
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Correlation heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, ax=axes[0, 0])
    axes[0, 0].set_title(f'{method.capitalize()} Correlation Matrix')
    
    # Plot 2: P-value heatmap
    sns.heatmap(p_matrix, annot=True, cmap='viridis', 
                square=True, linewidths=0.5, ax=axes[0, 1])
    axes[0, 1].set_title('P-Value Matrix')
    
    # Plot 3: Pair plot
    sns.pairplot(df[variables], diag_kind='kde', ax=axes[1, 0])
    axes[1, 0].set_title('Pair Plot')
    
    # Plot 4: Correlation distribution
    corr_values = []
    for i in range(len(variables)):
        for j in range(i+1, len(variables)):
            corr_values.append(corr_matrix.iloc[i, j])
    
    sns.histplot(data=corr_values, kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Correlation Distribution')
    axes[1, 1].set_xlabel('Correlation Coefficient')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.suptitle(f'Comprehensive {method.capitalize()} Correlation Analysis', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

# Use custom correlation function
np.random.seed(42)
corr_data = pd.DataFrame({
    'X1': np.random.randn(100),
    'X2': np.random.randn(100),
    'X3': np.random.randn(100),
    'X4': np.random.randn(100)
})

# Add correlations
corr_data['X2'] = 0.7 * corr_data['X1'] + 0.3 * corr_data['X2']
corr_data['X3'] = 0.5 * corr_data['X1'] + 0.5 * corr_data['X3']

fig = create_correlation_analysis(corr_data, ['X1', 'X2', 'X3', 'X4'], 'pearson')
plt.show()
```

## Advanced Styling

### Custom Style Sheets

```python
# Create custom style sheet
custom_style = {
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.edgecolor': '#dee2e6',
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'grid.color': '#e9ecef',
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.8,
    'xtick.color': '#495057',
    'ytick.color': '#495057',
    'text.color': '#212529',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.facecolor': 'white',
    'legend.edgecolor': '#dee2e6'
}

# Apply custom style
plt.rcParams.update(custom_style)

# Create plot with custom style
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Scatter plot with custom style
sns.scatterplot(data=df_pair, x='X1', y='X2', alpha=0.6, ax=axes[0])
axes[0].set_title('Custom Styled Scatter Plot')
axes[0].set_xlabel('X1 Variable')
axes[0].set_ylabel('X2 Variable')

# Plot 2: Box plot with custom style
sns.boxplot(data=df, x='category', y='value', ax=axes[1])
axes[1].set_title('Custom Styled Box Plot')
axes[1].set_xlabel('Category')
axes[1].set_ylabel('Value')

plt.suptitle('Advanced Custom Styling', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Reset to default style
plt.rcParams.update(plt.rcParamsDefault)
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)
```

### Context-Sensitive Styling

```python
# Create context-sensitive styling
def apply_context_style(context='paper'):
    """Apply context-appropriate styling"""
    
    if context == 'paper':
        # Publication style
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.figsize': (6, 4)
        })
    elif context == 'poster':
        # Poster style
        plt.rcParams.update({
            'font.size': 16,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'figure.figsize': (12, 8)
        })
    elif context == 'talk':
        # Presentation style
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.figsize': (10, 6)
        })

# Apply different contexts
contexts = ['paper', 'talk', 'poster']
figs = []

for context in contexts:
    apply_context_style(context)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df_pair, x='X1', y='X2', alpha=0.6, ax=ax)
    ax.set_title(f'{context.capitalize()} Context Style')
    ax.set_xlabel('X1 Variable')
    ax.set_ylabel('X2 Variable')
    
    figs.append(fig)

# Display all contexts
for i, fig in enumerate(figs):
    plt.figure(fig.number)
    plt.show()

# Reset to default
plt.rcParams.update(plt.rcParamsDefault)
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.2)
```

## Summary

This guide covered advanced Seaborn features:

1. **Custom Color Palettes**: Creating and using custom color schemes
2. **Statistical Annotations**: Adding comprehensive statistical information
3. **Integration with Statistical Libraries**: SciPy and Statsmodels integration
4. **Publication Quality Output**: High-resolution and LaTeX integration
5. **Custom Plot Functions**: Reusable statistical visualization functions
6. **Advanced Styling**: Custom styles and context-sensitive formatting

These advanced features are essential for:
- **Professional Presentations**: Creating publication-quality figures
- **Statistical Analysis**: Comprehensive statistical testing and diagnostics
- **Custom Visualizations**: Tailored plots for specific research needs
- **Reproducible Research**: Consistent and automated plotting workflows

## Best Practices

1. **Choose Appropriate Context**: Use paper, talk, or poster context for different audiences
2. **Maintain Consistency**: Use consistent styling across all figures
3. **Include Statistical Context**: Always provide appropriate statistical annotations
4. **Optimize for Output**: Use appropriate DPI and formats for intended use
5. **Document Your Choices**: Explain styling and statistical decisions

## Next Steps

- Explore interactive visualizations and dashboards
- Learn about automated reporting and figure generation
- Master advanced statistical modeling visualization
- Practice with real-world research datasets
- Customize for specific journal or presentation requirements

Remember: Advanced features should enhance clarity and understanding, not just add complexity! 