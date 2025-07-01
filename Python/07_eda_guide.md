# Python Exploratory Data Analysis Guide

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-blue.svg)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.11%2B-orange.svg)](https://seaborn.pydata.org/)

A comprehensive guide to Exploratory Data Analysis (EDA) in Python for data science and machine learning applications.

## Table of Contents

1. [Introduction to EDA](#introduction-to-eda)
2. [Data Overview](#data-overview)
3. [Univariate Analysis](#univariate-analysis)
4. [Bivariate Analysis](#bivariate-analysis)
5. [Multivariate Analysis](#multivariate-analysis)
6. [Data Visualization](#data-visualization)
7. [Statistical Analysis](#statistical-analysis)
8. [Correlation Analysis](#correlation-analysis)
9. [Outlier Detection](#outlier-detection)
10. [Data Quality Assessment](#data-quality-assessment)

## Introduction to EDA

Exploratory Data Analysis is the process of investigating datasets to understand their main characteristics, patterns, and relationships.

### EDA Goals

- Understand data structure and content
- Identify patterns and relationships
- Detect anomalies and outliers
- Generate hypotheses for further analysis
- Prepare data for modeling

### Basic Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Sample dataset
np.random.seed(42)
data = {
    'age': np.random.normal(35, 10, 1000),
    'salary': np.random.normal(60000, 15000, 1000),
    'experience': np.random.normal(8, 3, 1000),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
    'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], 1000),
    'satisfaction': np.random.uniform(1, 10, 1000)
}

df = pd.DataFrame(data)
```

## Data Overview

### Basic Information

```python
def data_overview(df):
    """Generate comprehensive data overview."""
    
    print("=== DATA OVERVIEW ===")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n=== DATA TYPES ===")
    print(df.dtypes)
    
    print("\n=== MISSING VALUES ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Missing_Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing_Count'] > 0])
    
    print("\n=== UNIQUE VALUES ===")
    for col in df.columns:
        unique_count = df[col].nunique()
        print(f"{col}: {unique_count} unique values")
    
    return missing_df

# Generate overview
overview = data_overview(df)
```

### Descriptive Statistics

```python
def descriptive_statistics(df):
    """Generate descriptive statistics."""
    
    print("=== NUMERICAL STATISTICS ===")
    print(df.describe())
    
    print("\n=== CATEGORICAL STATISTICS ===")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
        print(f"Mode: {df[col].mode()[0] if not df[col].mode().empty else 'None'}")

# Generate statistics
descriptive_statistics(df)
```

## Univariate Analysis

### Numerical Variables

```python
def analyze_numerical_variables(df):
    """Analyze numerical variables with visualizations."""
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Analysis of {col}', fontsize=16)
        
        # Histogram
        axes[0, 0].hist(df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Histogram')
        axes[0, 0].set_xlabel(col)
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[0, 1].boxplot(df[col])
        axes[0, 1].set_title('Box Plot')
        axes[0, 1].set_ylabel(col)
        
        # Q-Q plot
        stats.probplot(df[col], dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # Summary statistics
        stats_text = f"""
        Mean: {df[col].mean():.2f}
        Median: {df[col].median():.2f}
        Std: {df[col].std():.2f}
        Skewness: {df[col].skew():.2f}
        Kurtosis: {df[col].kurtosis():.2f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='center')
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()

# Analyze numerical variables
analyze_numerical_variables(df)
```

### Categorical Variables

```python
def analyze_categorical_variables(df):
    """Analyze categorical variables with visualizations."""
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Analysis of {col}', fontsize=16)
        
        # Bar plot
        value_counts = df[col].value_counts()
        axes[0].bar(range(len(value_counts)), value_counts.values)
        axes[0].set_title('Bar Plot')
        axes[0].set_xlabel(col)
        axes[0].set_ylabel('Count')
        axes[0].set_xticks(range(len(value_counts)))
        axes[0].set_xticklabels(value_counts.index, rotation=45)
        
        # Pie chart
        axes[1].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
        axes[1].set_title('Pie Chart')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\n{col} Statistics:")
        print(f"Unique values: {df[col].nunique()}")
        print(f"Most common: {df[col].mode()[0]}")
        print(f"Least common: {value_counts.index[-1]}")

# Analyze categorical variables
analyze_categorical_variables(df)
```

## Bivariate Analysis

### Numerical vs Numerical

```python
def analyze_numerical_pairs(df):
    """Analyze relationships between numerical variables."""
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Correlation matrix
    correlation_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()
    
    # Scatter plot matrix
    sns.pairplot(df[numerical_cols], diag_kind='kde')
    plt.show()
    
    # Individual scatter plots with regression lines
    for i, col1 in enumerate(numerical_cols):
        for j, col2 in enumerate(numerical_cols[i+1:], i+1):
            plt.figure(figsize=(8, 6))
            sns.regplot(data=df, x=col1, y=col2, scatter_kws={'alpha':0.5})
            plt.title(f'{col1} vs {col2}')
            plt.show()

# Analyze numerical pairs
analyze_numerical_pairs(df)
```

### Numerical vs Categorical

```python
def analyze_numerical_categorical(df):
    """Analyze relationships between numerical and categorical variables."""
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for num_col in numerical_cols:
        for cat_col in categorical_cols:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(f'{num_col} vs {cat_col}', fontsize=16)
            
            # Box plot
            df.boxplot(column=num_col, by=cat_col, ax=axes[0])
            axes[0].set_title('Box Plot')
            axes[0].set_xlabel(cat_col)
            axes[0].set_ylabel(num_col)
            
            # Violin plot
            sns.violinplot(data=df, x=cat_col, y=num_col, ax=axes[1])
            axes[1].set_title('Violin Plot')
            axes[1].set_xlabel(cat_col)
            axes[1].set_ylabel(num_col)
            
            plt.tight_layout()
            plt.show()
            
            # Statistical test
            categories = df[cat_col].unique()
            if len(categories) == 2:
                # T-test for binary categorical
                group1 = df[df[cat_col] == categories[0]][num_col]
                group2 = df[df[cat_col] == categories[1]][num_col]
                t_stat, p_value = stats.ttest_ind(group1, group2)
                print(f"T-test p-value: {p_value:.4f}")
            else:
                # ANOVA for multiple categories
                groups = [df[df[cat_col] == cat][num_col] for cat in categories]
                f_stat, p_value = stats.f_oneway(*groups)
                print(f"ANOVA p-value: {p_value:.4f}")

# Analyze numerical vs categorical
analyze_numerical_categorical(df)
```

## Multivariate Analysis

### Principal Component Analysis

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def perform_pca(df):
    """Perform Principal Component Analysis."""
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(scaled_data)
    
    # Explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Component')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(numerical_cols))],
        index=numerical_cols
    )
    
    print("Component Loadings:")
    print(loadings)
    
    return pca, loadings

# Perform PCA
pca_result, loadings = perform_pca(df)
```

### Cluster Analysis

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def perform_clustering(df, max_clusters=10):
    """Perform K-means clustering analysis."""
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])
    
    # Find optimal number of clusters
    silhouette_scores = []
    inertias = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        silhouette_avg = silhouette_score(scaled_data, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        inertias.append(kmeans.inertia_)
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(range(2, max_clusters + 1), silhouette_scores, 'bo-')
    axes[0].set_xlabel('Number of Clusters')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title('Silhouette Score vs Number of Clusters')
    
    axes[1].plot(range(2, max_clusters + 1), inertias, 'ro-')
    axes[1].set_xlabel('Number of Clusters')
    axes[1].set_ylabel('Inertia')
    axes[1].set_title('Elbow Method')
    
    plt.tight_layout()
    plt.show()
    
    # Optimal clustering
    optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Perform final clustering
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    df['cluster'] = final_kmeans.fit_predict(scaled_data)
    
    return df

# Perform clustering
df_clustered = perform_clustering(df)
```

## Data Visualization

### Advanced Visualizations

```python
def create_advanced_visualizations(df):
    """Create advanced visualizations for EDA."""
    
    # Distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Age distribution by department
    for dept in df['department'].unique():
        dept_data = df[df['department'] == dept]['age']
        axes[0, 0].hist(dept_data, alpha=0.7, label=dept, bins=20)
    axes[0, 0].set_title('Age Distribution by Department')
    axes[0, 0].legend()
    
    # Salary vs Experience with department colors
    for dept in df['department'].unique():
        dept_data = df[df['department'] == dept]
        axes[0, 1].scatter(dept_data['experience'], dept_data['salary'], 
                          alpha=0.6, label=dept)
    axes[0, 1].set_xlabel('Experience')
    axes[0, 1].set_ylabel('Salary')
    axes[0, 1].set_title('Salary vs Experience by Department')
    axes[0, 1].legend()
    
    # Education level distribution
    education_counts = df['education'].value_counts()
    axes[1, 0].pie(education_counts.values, labels=education_counts.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Education Level Distribution')
    
    # Satisfaction score distribution
    axes[1, 1].hist(df['satisfaction'], bins=30, alpha=0.7, color='green')
    axes[1, 1].set_xlabel('Satisfaction Score')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Satisfaction Score Distribution')
    
    plt.tight_layout()
    plt.show()

# Create advanced visualizations
create_advanced_visualizations(df)
```

## Statistical Analysis

### Hypothesis Testing

```python
def perform_statistical_tests(df):
    """Perform various statistical tests."""
    
    print("=== STATISTICAL TESTS ===")
    
    # Test for normal distribution
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        statistic, p_value = stats.normaltest(df[col])
        print(f"{col} - Normality test p-value: {p_value:.4f}")
    
    # Correlation tests
    for i, col1 in enumerate(numerical_cols):
        for col2 in numerical_cols[i+1:]:
            correlation, p_value = stats.pearsonr(df[col1], df[col2])
            print(f"{col1} vs {col2} - Correlation: {correlation:.3f}, p-value: {p_value:.4f}")
    
    # Chi-square test for categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) >= 2:
        contingency_table = pd.crosstab(df[categorical_cols[0]], df[categorical_cols[1]])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"Chi-square test ({categorical_cols[0]} vs {categorical_cols[1]}): p-value = {p_value:.4f}")

# Perform statistical tests
perform_statistical_tests(df)
```

## Summary

Exploratory Data Analysis is essential for understanding your data:

- **Data Overview**: Understand structure, types, and missing values
- **Univariate Analysis**: Explore individual variables
- **Bivariate Analysis**: Examine relationships between pairs of variables
- **Multivariate Analysis**: Understand complex relationships
- **Visualization**: Create informative plots and charts
- **Statistical Testing**: Validate findings with statistical methods

Mastering EDA will help you make informed decisions about data preprocessing and modeling strategies.

## Next Steps

- Practice EDA on real-world datasets
- Explore advanced visualization libraries like Plotly
- Learn about automated EDA tools like pandas-profiling
- Study domain-specific EDA techniques

---

**Happy Data Exploration!** 🔍✨ 