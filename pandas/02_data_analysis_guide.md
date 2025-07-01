# Pandas Data Analysis: A Comprehensive Guide

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

## Table of Contents

1. [Introduction to Data Analysis with Pandas](#introduction-to-data-analysis-with-pandas)
2. [Grouping and Aggregation](#grouping-and-aggregation)
3. [Pivot Tables and Cross-Tabulation](#pivot-tables-and-cross-tabulation)
4. [Data Merging and Joining](#data-merging-and-joining)
5. [Time Series Analysis](#time-series-analysis)
6. [Statistical Analysis](#statistical-analysis)
7. [Data Transformation](#data-transformation)
8. [Advanced Filtering](#advanced-filtering)
9. [Data Validation](#data-validation)
10. [Performance Optimization](#performance-optimization)
11. [Real-World Applications](#real-world-applications)
12. [Best Practices](#best-practices)

## Introduction to Data Analysis with Pandas

Data analysis with pandas involves exploring, transforming, and extracting insights from data. This guide covers:

- **Grouping and aggregation** for summary statistics
- **Pivot tables** for multi-dimensional analysis
- **Data merging** for combining datasets
- **Time series analysis** for temporal data
- **Statistical analysis** for data insights
- **Data transformation** for feature engineering

### Sample Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create sample dataset
np.random.seed(42)
n_samples = 1000

data = {
    'employee_id': range(1, n_samples + 1),
    'name': [f'Employee_{i}' for i in range(1, n_samples + 1)],
    'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing', 'Finance'], n_samples),
    'position': np.random.choice(['Junior', 'Senior', 'Manager', 'Director'], n_samples),
    'age': np.random.normal(35, 10, n_samples).astype(int),
    'salary': np.random.normal(60000, 20000, n_samples).astype(int),
    'experience_years': np.random.exponential(5, n_samples).astype(int),
    'performance_rating': np.random.normal(3.5, 0.8, n_samples),
    'hire_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Boston', 'Seattle'], n_samples),
    'gender': np.random.choice(['M', 'F'], n_samples)
}

df = pd.DataFrame(data)

# Clean up data
df['age'] = df['age'].clip(22, 65)
df['salary'] = df['salary'].clip(30000, 150000)
df['experience_years'] = df['experience_years'].clip(0, 25)
df['performance_rating'] = df['performance_rating'].clip(1, 5)

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
```

## Grouping and Aggregation

### 1. Basic GroupBy Operations

```python
# Group by single column
dept_stats = df.groupby('department').agg({
    'salary': ['mean', 'median', 'std', 'count'],
    'age': ['mean', 'min', 'max'],
    'performance_rating': ['mean', 'std']
})

print(f"Department statistics:\n{dept_stats}")

# Flatten column names
dept_stats.columns = ['_'.join(col).strip() for col in dept_stats.columns]
print(f"\nFlattened column names:\n{dept_stats.columns.tolist()}")

# Group by multiple columns
dept_pos_stats = df.groupby(['department', 'position']).agg({
    'salary': ['mean', 'count'],
    'age': 'mean',
    'performance_rating': 'mean'
}).round(2)

print(f"\nDepartment and position statistics:\n{dept_pos_stats}")
```

### 2. Custom Aggregation Functions

```python
# Custom aggregation functions
def salary_range(x):
    return x.max() - x.min()

def top_performers(x):
    return (x > 4.0).sum()

# Apply custom functions
custom_stats = df.groupby('department').agg({
    'salary': ['mean', salary_range, lambda x: x.quantile(0.75)],
    'performance_rating': ['mean', top_performers],
    'age': ['mean', lambda x: x.std()]
})

print(f"Custom department statistics:\n{custom_stats}")

# Using apply for complex aggregations
def department_summary(group):
    return pd.Series({
        'avg_salary': group['salary'].mean(),
        'salary_range': group['salary'].max() - group['salary'].min(),
        'high_performers': (group['performance_rating'] > 4.0).sum(),
        'employee_count': len(group),
        'avg_experience': group['experience_years'].mean()
    })

dept_summary = df.groupby('department').apply(department_summary)
print(f"\nDepartment summary:\n{dept_summary}")
```

### 3. GroupBy with Transform and Filter

```python
# Transform: Add group-level statistics to original data
df['dept_avg_salary'] = df.groupby('department')['salary'].transform('mean')
df['salary_vs_dept_avg'] = df['salary'] - df['dept_avg_salary']
df['salary_percentile'] = df.groupby('department')['salary'].transform(
    lambda x: x.rank(pct=True)
)

print(f"DataFrame with group statistics:\n{df[['name', 'department', 'salary', 'dept_avg_salary', 'salary_vs_dept_avg', 'salary_percentile']].head()}")

# Filter: Keep groups that meet certain criteria
high_performing_depts = df.groupby('department').filter(
    lambda x: x['performance_rating'].mean() > 3.5
)
print(f"\nHigh-performing departments:\n{high_performing_depts['department'].unique()}")

# Apply: Complex group operations
def analyze_department(group):
    """Analyze department performance and return summary."""
    high_performers = group[group['performance_rating'] > 4.0]
    low_performers = group[group['performance_rating'] < 3.0]
    
    return pd.Series({
        'total_employees': len(group),
        'high_performers': len(high_performers),
        'low_performers': len(low_performers),
        'avg_salary': group['salary'].mean(),
        'salary_inequality': group['salary'].std() / group['salary'].mean(),
        'performance_score': group['performance_rating'].mean()
    })

dept_analysis = df.groupby('department').apply(analyze_department)
print(f"\nDepartment analysis:\n{dept_analysis}")
```

## Pivot Tables and Cross-Tabulation

### 1. Basic Pivot Tables

```python
# Simple pivot table
salary_pivot = df.pivot_table(
    values='salary',
    index='department',
    columns='position',
    aggfunc='mean',
    fill_value=0
)

print(f"Salary pivot table:\n{salary_pivot}")

# Pivot table with multiple aggregations
multi_pivot = df.pivot_table(
    values=['salary', 'performance_rating'],
    index='department',
    columns='position',
    aggfunc={
        'salary': ['mean', 'count'],
        'performance_rating': ['mean', 'std']
    },
    fill_value=0
)

print(f"\nMulti-aggregation pivot table:\n{multi_pivot}")

# Pivot table with margins
pivot_with_margins = df.pivot_table(
    values='salary',
    index='department',
    columns='gender',
    aggfunc='mean',
    margins=True,
    margins_name='Total'
)

print(f"\nPivot table with margins:\n{pivot_with_margins}")
```

### 2. Cross-Tabulation

```python
# Basic cross-tabulation
dept_gender_crosstab = pd.crosstab(
    df['department'], 
    df['gender'], 
    margins=True
)

print(f"Department vs Gender cross-tabulation:\n{dept_gender_crosstab}")

# Cross-tabulation with percentages
dept_gender_pct = pd.crosstab(
    df['department'], 
    df['gender'], 
    normalize='index',
    margins=True
) * 100

print(f"\nDepartment vs Gender percentages:\n{dept_gender_pct.round(1)}")

# Cross-tabulation with multiple variables
multi_crosstab = pd.crosstab(
    [df['department'], df['position']],
    df['gender'],
    margins=True
)

print(f"\nMulti-variable cross-tabulation:\n{multi_crosstab}")
```

### 3. Advanced Pivot Operations

```python
# Pivot table with custom aggregation
def salary_percentile(x):
    return x.quantile(0.75)

advanced_pivot = df.pivot_table(
    values='salary',
    index=['department', 'position'],
    columns='gender',
    aggfunc=['mean', 'count', salary_percentile],
    fill_value=0
)

print(f"Advanced pivot table:\n{advanced_pivot}")

# Pivot table with date grouping
df['hire_year'] = df['hire_date'].dt.year
df['hire_month'] = df['hire_date'].dt.month

time_pivot = df.pivot_table(
    values='salary',
    index='hire_year',
    columns='department',
    aggfunc='mean',
    fill_value=0
)

print(f"\nTime-based pivot table:\n{time_pivot}")
```

## Data Merging and Joining

### 1. Basic Merging

```python
# Create additional datasets
salary_history = pd.DataFrame({
    'employee_id': np.random.choice(df['employee_id'], 500),
    'year': np.random.choice([2022, 2023], 500),
    'salary': np.random.normal(55000, 15000, 500).astype(int),
    'bonus': np.random.normal(5000, 2000, 500).astype(int)
})

performance_reviews = pd.DataFrame({
    'employee_id': np.random.choice(df['employee_id'], 300),
    'review_date': pd.date_range('2023-01-01', periods=300, freq='D'),
    'review_score': np.random.normal(3.5, 0.8, 300),
    'reviewer': np.random.choice(['Manager', 'Director', 'HR'], 300)
})

print(f"Main dataset: {df.shape}")
print(f"Salary history: {salary_history.shape}")
print(f"Performance reviews: {performance_reviews.shape}")

# Inner merge
merged_inner = df.merge(salary_history, on='employee_id', how='inner')
print(f"\nInner merge result: {merged_inner.shape}")

# Left merge
merged_left = df.merge(salary_history, on='employee_id', how='left')
print(f"Left merge result: {merged_left.shape}")

# Right merge
merged_right = df.merge(salary_history, on='employee_id', how='right')
print(f"Right merge result: {merged_right.shape}")

# Outer merge
merged_outer = df.merge(salary_history, on='employee_id', how='outer')
print(f"Outer merge result: {merged_outer.shape}")
```

### 2. Multiple Key Merging

```python
# Merge on multiple keys
salary_history['year'] = salary_history['year'].astype(str)
df['hire_year'] = df['hire_year'].astype(str)

multi_key_merge = df.merge(
    salary_history,
    left_on=['employee_id', 'hire_year'],
    right_on=['employee_id', 'year'],
    how='left'
)

print(f"Multi-key merge result: {multi_key_merge.shape}")

# Merge with suffixes
df_suffix = df.copy()
df_suffix.columns = [col + '_main' for col in df_suffix.columns]

merged_suffix = df.merge(
    df_suffix,
    left_on='employee_id',
    right_on='employee_id_main',
    suffixes=('', '_suffix')
)

print(f"\nMerge with suffixes result: {merged_suffix.shape}")
```

### 3. Concatenation

```python
# Concatenate DataFrames
df_2022 = df[df['hire_date'].dt.year == 2022].copy()
df_2023 = df[df['hire_date'].dt.year == 2023].copy()

# Vertical concatenation
concatenated = pd.concat([df_2022, df_2023], axis=0, ignore_index=True)
print(f"Concatenated result: {concatenated.shape}")

# Horizontal concatenation
df_left = df[['employee_id', 'name', 'department']].head(5)
df_right = df[['employee_id', 'salary', 'age']].head(5)

concatenated_horizontal = pd.concat([df_left, df_right], axis=1)
print(f"\nHorizontal concatenation:\n{concatenated_horizontal}")
```

## Time Series Analysis

### 1. Time Series Basics

```python
# Set hire_date as index
df_time = df.set_index('hire_date').sort_index()

print(f"Time series data:\n{df_time.head()}")

# Time-based indexing
recent_hires = df_time['2023-01-01':'2023-12-31']
print(f"\nRecent hires (2023): {len(recent_hires)}")

# Resampling by time periods
monthly_hires = df_time.resample('M').size()
print(f"\nMonthly hires:\n{monthly_hires}")

# Rolling statistics
rolling_salary = df_time['salary'].rolling(window=30).mean()
print(f"\n30-day rolling average salary:\n{rolling_salary.head()}")
```

### 2. Time Series Aggregation

```python
# Group by time periods
df_time['year'] = df_time.index.year
df_time['month'] = df_time.index.month
df_time['quarter'] = df_time.index.quarter

# Monthly statistics
monthly_stats = df_time.groupby([df_time.index.year, df_time.index.month]).agg({
    'salary': ['mean', 'count'],
    'performance_rating': 'mean',
    'age': 'mean'
})

print(f"Monthly statistics:\n{monthly_stats.head()}")

# Quarterly statistics
quarterly_stats = df_time.groupby([df_time.index.year, df_time.index.quarter]).agg({
    'salary': ['mean', 'std'],
    'performance_rating': ['mean', 'count']
})

print(f"\nQuarterly statistics:\n{quarterly_stats}")
```

### 3. Time Series Visualization

```python
# Monthly hire trends
monthly_hires.plot(figsize=(12, 6), title='Monthly Hires Over Time')
plt.show()

# Salary trends over time
salary_trends = df_time.groupby(df_time.index.to_period('M'))['salary'].mean()
salary_trends.plot(figsize=(12, 6), title='Average Salary Over Time')
plt.show()

# Department hiring trends
dept_hiring = df_time.groupby([df_time.index.to_period('M'), 'department']).size().unstack()
dept_hiring.plot(figsize=(12, 6), title='Department Hiring Trends')
plt.show()
```

## Statistical Analysis

### 1. Descriptive Statistics

```python
# Comprehensive descriptive statistics
descriptive_stats = df.describe(include='all')
print(f"Descriptive statistics:\n{descriptive_stats}")

# Statistics by groups
dept_stats = df.groupby('department').agg({
    'salary': ['mean', 'std', 'min', 'max', 'skew', 'kurtosis'],
    'age': ['mean', 'std', 'min', 'max'],
    'performance_rating': ['mean', 'std', 'min', 'max']
})

print(f"\nDepartment statistics:\n{dept_stats}")

# Correlation analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
print(f"\nCorrelation matrix:\n{correlation_matrix}")
```

### 2. Statistical Tests

```python
from scipy import stats

# T-test for salary difference between genders
male_salary = df[df['gender'] == 'M']['salary']
female_salary = df[df['gender'] == 'F']['salary']

t_stat, p_value = stats.ttest_ind(male_salary, female_salary)
print(f"T-test for salary difference (M vs F):")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant difference: {p_value < 0.05}")

# ANOVA for salary across departments
dept_groups = [group['salary'].values for name, group in df.groupby('department')]
f_stat, p_value = stats.f_oneway(*dept_groups)
print(f"\nANOVA for salary across departments:")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant difference: {p_value < 0.05}")

# Chi-square test for department vs gender
contingency_table = pd.crosstab(df['department'], df['gender'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\nChi-square test for department vs gender:")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant association: {p_value < 0.05}")
```

### 3. Distribution Analysis

```python
# Distribution analysis
def analyze_distribution(data, column):
    """Analyze distribution of a numeric column."""
    skewness = data[column].skew()
    kurtosis = data[column].kurtosis()
    
    print(f"\nDistribution analysis for {column}:")
    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")
    
    if abs(skewness) < 0.5:
        print("Distribution is approximately symmetric")
    elif skewness > 0.5:
        print("Distribution is right-skewed")
    else:
        print("Distribution is left-skewed")
    
    if abs(kurtosis) < 2:
        print("Distribution has normal kurtosis")
    elif kurtosis > 2:
        print("Distribution has heavy tails")
    else:
        print("Distribution has light tails")

analyze_distribution(df, 'salary')
analyze_distribution(df, 'age')
analyze_distribution(df, 'performance_rating')
```

## Data Transformation

### 1. Feature Engineering

```python
# Create new features
df['salary_per_year'] = df['salary'] / df['experience_years']
df['age_group'] = pd.cut(df['age'], bins=[20, 30, 40, 50, 70], labels=['20-30', '30-40', '40-50', '50+'])
df['performance_category'] = pd.cut(df['performance_rating'], bins=[1, 3, 4, 5], labels=['Low', 'Medium', 'High'])
df['tenure_months'] = (pd.Timestamp.now() - df['hire_date']).dt.days / 30

# Create interaction features
df['salary_age_ratio'] = df['salary'] / df['age']
df['experience_performance_ratio'] = df['experience_years'] * df['performance_rating']

print(f"New features created:\n{df[['salary_per_year', 'age_group', 'performance_category', 'tenure_months']].head()}")
```

### 2. Data Normalization and Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standard scaling
scaler = StandardScaler()
df['salary_scaled'] = scaler.fit_transform(df[['salary']])
df['age_scaled'] = scaler.fit_transform(df[['age']])

# Min-max scaling
minmax_scaler = MinMaxScaler()
df['performance_scaled'] = minmax_scaler.fit_transform(df[['performance_rating']])

# Z-score calculation
df['salary_zscore'] = (df['salary'] - df['salary'].mean()) / df['salary'].std()

print(f"Scaled features:\n{df[['salary', 'salary_scaled', 'salary_zscore']].head()}")
```

### 3. Categorical Encoding

```python
# One-hot encoding
dept_encoded = pd.get_dummies(df['department'], prefix='dept')
position_encoded = pd.get_dummies(df['position'], prefix='pos')

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['department_encoded'] = le.fit_transform(df['department'])
df['position_encoded'] = le.fit_transform(df['position'])

# Combine with original data
df_encoded = pd.concat([df, dept_encoded, position_encoded], axis=1)

print(f"Encoded features:\n{df_encoded[['department', 'department_encoded'] + list(dept_encoded.columns)].head()}")
```

## Advanced Filtering

### 1. Complex Boolean Filtering

```python
# Multiple conditions
high_performing_seniors = df[
    (df['position'] == 'Senior') & 
    (df['performance_rating'] > 4.0) & 
    (df['salary'] > 70000)
]

print(f"High-performing seniors: {len(high_performing_seniors)}")

# Using query method
it_managers = df.query("department == 'IT' and position == 'Manager'")
young_high_earners = df.query("age < 30 and salary > 60000")

print(f"IT managers: {len(it_managers)}")
print(f"Young high earners: {len(young_high_earners)}")

# Complex conditions with functions
def is_high_performer(row):
    return (row['performance_rating'] > 4.0 and 
            row['salary'] > row['salary'].mean() and
            row['experience_years'] > 3)

high_performers = df[df.apply(is_high_performer, axis=1)]
print(f"High performers (custom function): {len(high_performers)}")
```

### 2. Advanced String Filtering

```python
# String contains
names_with_a = df[df['name'].str.contains('a', case=False)]
print(f"Names with 'a': {len(names_with_a)}")

# String startswith/endswith
names_starting_with_e = df[df['name'].str.startswith('Employee_1')]
print(f"Names starting with 'Employee_1': {len(names_starting_with_e)}")

# Regular expressions
import re
names_with_numbers = df[df['name'].str.contains(r'\d+', regex=True)]
print(f"Names with numbers: {len(names_with_numbers)}")

# Multiple string conditions
complex_string_filter = df[
    df['name'].str.contains('Employee') &
    df['department'].str.contains('IT|Sales')
]
print(f"Complex string filter: {len(complex_string_filter)}")
```

### 3. Time-Based Filtering

```python
# Date range filtering
recent_hires = df[df['hire_date'] > '2023-01-01']
print(f"Recent hires (2023+): {len(recent_hires)}")

# Time-based conditions
df['hire_year'] = df['hire_date'].dt.year
df['hire_month'] = df['hire_date'].dt.month
df['hire_quarter'] = df['hire_date'].dt.quarter

q1_hires = df[df['hire_quarter'] == 1]
print(f"Q1 hires: {len(q1_hires)}")

# Time since hire
df['days_since_hire'] = (pd.Timestamp.now() - df['hire_date']).dt.days
long_tenure = df[df['days_since_hire'] > 365]
print(f"Long tenure employees (>1 year): {len(long_tenure)}")
```

## Data Validation

### 1. Data Quality Checks

```python
def validate_data(df):
    """Comprehensive data validation function."""
    validation_results = {}
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    validation_results['missing_values'] = missing_counts[missing_counts > 0]
    
    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    validation_results['duplicates'] = duplicate_count
    
    # Check data types
    validation_results['data_types'] = df.dtypes
    
    # Check for outliers (using IQR method)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outliers[col] = outlier_count
    validation_results['outliers'] = outliers
    
    # Check value ranges
    value_ranges = {}
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            value_ranges[col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean()
            }
    validation_results['value_ranges'] = value_ranges
    
    return validation_results

validation_results = validate_data(df)
print("Data validation results:")
for key, value in validation_results.items():
    print(f"\n{key}:")
    print(value)
```

### 2. Data Consistency Checks

```python
def check_data_consistency(df):
    """Check for data consistency issues."""
    issues = []
    
    # Check logical consistency
    if (df['age'] < df['experience_years'] + 18).any():
        issues.append("Some employees have impossible age/experience combinations")
    
    if (df['salary'] < 0).any():
        issues.append("Negative salaries found")
    
    if (df['performance_rating'] < 1).any() or (df['performance_rating'] > 5).any():
        issues.append("Performance ratings outside valid range")
    
    # Check for impossible dates
    if (df['hire_date'] > pd.Timestamp.now()).any():
        issues.append("Future hire dates found")
    
    # Check for department-position consistency
    dept_pos_combinations = df.groupby(['department', 'position']).size()
    if dept_pos_combinations.max() > len(df) * 0.5:
        issues.append("Suspicious department-position distribution")
    
    return issues

consistency_issues = check_data_consistency(df)
print("Data consistency issues:")
for issue in consistency_issues:
    print(f"- {issue}")
```

## Performance Optimization

### 1. Memory Optimization

```python
def optimize_memory(df):
    """Optimize DataFrame memory usage."""
    initial_memory = df.memory_usage(deep=True).sum()
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if column can be converted to category
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        elif df[col].dtype == 'int64':
            # Downcast integers
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'float64':
            # Downcast floats
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    final_memory = df.memory_usage(deep=True).sum()
    memory_reduction = (initial_memory - final_memory) / initial_memory * 100
    
    print(f"Memory optimization:")
    print(f"Initial memory: {initial_memory / 1024**2:.2f} MB")
    print(f"Final memory: {final_memory / 1024**2:.2f} MB")
    print(f"Memory reduction: {memory_reduction:.1f}%")
    
    return df

df_optimized = optimize_memory(df.copy())
```

### 2. Query Optimization

```python
import time

# Measure query performance
def measure_performance(func, *args, **kwargs):
    """Measure function execution time."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

# Test different query methods
def query_method_1(df):
    return df[(df['department'] == 'IT') & (df['salary'] > 60000)]

def query_method_2(df):
    return df.query("department == 'IT' and salary > 60000")

def query_method_3(df):
    it_dept = df[df['department'] == 'IT']
    return it_dept[it_dept['salary'] > 60000]

# Performance comparison
methods = [query_method_1, query_method_2, query_method_3]
method_names = ['Boolean indexing', 'Query method', 'Chained filtering']

for method, name in zip(methods, method_names):
    result, execution_time = measure_performance(method, df)
    print(f"{name}: {execution_time:.6f} seconds, {len(result)} results")
```

## Real-World Applications

### 1. Employee Analytics Dashboard

```python
def create_employee_dashboard(df):
    """Create comprehensive employee analytics dashboard."""
    dashboard = {}
    
    # Key metrics
    dashboard['total_employees'] = len(df)
    dashboard['departments'] = df['department'].nunique()
    dashboard['avg_salary'] = df['salary'].mean()
    dashboard['avg_performance'] = df['performance_rating'].mean()
    
    # Department analysis
    dept_analysis = df.groupby('department').agg({
        'employee_id': 'count',
        'salary': ['mean', 'std'],
        'performance_rating': 'mean',
        'age': 'mean'
    }).round(2)
    dashboard['department_analysis'] = dept_analysis
    
    # Performance distribution
    performance_dist = df['performance_rating'].value_counts().sort_index()
    dashboard['performance_distribution'] = performance_dist
    
    # Salary analysis
    salary_quartiles = df['salary'].quantile([0.25, 0.5, 0.75])
    dashboard['salary_quartiles'] = salary_quartiles
    
    # Hiring trends
    hiring_trends = df.groupby(df['hire_date'].dt.to_period('M')).size()
    dashboard['hiring_trends'] = hiring_trends
    
    return dashboard

dashboard = create_employee_dashboard(df)
print("Employee Analytics Dashboard:")
for key, value in dashboard.items():
    print(f"\n{key}:")
    print(value)
```

### 2. Salary Equity Analysis

```python
def analyze_salary_equity(df):
    """Analyze salary equity across different dimensions."""
    equity_analysis = {}
    
    # Gender pay gap
    gender_pay = df.groupby('gender')['salary'].agg(['mean', 'std', 'count'])
    gender_pay_gap = (gender_pay.loc['M', 'mean'] - gender_pay.loc['F', 'mean']) / gender_pay.loc['F', 'mean'] * 100
    equity_analysis['gender_pay_gap_percent'] = gender_pay_gap
    
    # Department pay analysis
    dept_pay = df.groupby('department')['salary'].agg(['mean', 'std', 'count'])
    equity_analysis['department_pay_analysis'] = dept_pay
    
    # Position pay analysis
    position_pay = df.groupby('position')['salary'].agg(['mean', 'std', 'count'])
    equity_analysis['position_pay_analysis'] = position_pay
    
    # Experience vs salary correlation
    experience_salary_corr = df['experience_years'].corr(df['salary'])
    equity_analysis['experience_salary_correlation'] = experience_salary_corr
    
    # Performance vs salary correlation
    performance_salary_corr = df['performance_rating'].corr(df['salary'])
    equity_analysis['performance_salary_correlation'] = performance_salary_corr
    
    return equity_analysis

equity_analysis = analyze_salary_equity(df)
print("Salary Equity Analysis:")
for key, value in equity_analysis.items():
    print(f"\n{key}:")
    print(value)
```

### 3. Predictive Analytics Preparation

```python
def prepare_for_prediction(df):
    """Prepare data for predictive analytics."""
    # Feature engineering
    df_prep = df.copy()
    
    # Create target variable (high performer)
    df_prep['is_high_performer'] = (df_prep['performance_rating'] > 4.0).astype(int)
    
    # Create features
    df_prep['salary_per_experience'] = df_prep['salary'] / (df_prep['experience_years'] + 1)
    df_prep['age_experience_ratio'] = df_prep['age'] / (df_prep['experience_years'] + 1)
    df_prep['tenure_years'] = (pd.Timestamp.now() - df_prep['hire_date']).dt.days / 365.25
    
    # Encode categorical variables
    df_prep = pd.get_dummies(df_prep, columns=['department', 'position', 'gender', 'city'])
    
    # Select features for modeling
    feature_cols = [col for col in df_prep.columns if col not in [
        'employee_id', 'name', 'hire_date', 'is_high_performer'
    ]]
    
    X = df_prep[feature_cols]
    y = df_prep['is_high_performer']
    
    return X, y, feature_cols

X, y, feature_cols = prepare_for_prediction(df)
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Number of features: {len(feature_cols)}")
```

## Best Practices

### 1. Code Organization

```python
class EmployeeAnalyzer:
    """Class for employee data analysis."""
    
    def __init__(self, df):
        self.df = df.copy()
        self.validate_data()
    
    def validate_data(self):
        """Validate input data."""
        required_cols = ['employee_id', 'name', 'department', 'salary']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def get_department_stats(self):
        """Get department statistics."""
        return self.df.groupby('department').agg({
            'salary': ['mean', 'std', 'count'],
            'performance_rating': 'mean'
        })
    
    def get_salary_analysis(self):
        """Get salary analysis."""
        return {
            'mean': self.df['salary'].mean(),
            'median': self.df['salary'].median(),
            'std': self.df['salary'].std(),
            'quartiles': self.df['salary'].quantile([0.25, 0.5, 0.75])
        }
    
    def get_performance_analysis(self):
        """Get performance analysis."""
        return self.df['performance_rating'].describe()

# Usage
analyzer = EmployeeAnalyzer(df)
dept_stats = analyzer.get_department_stats()
salary_analysis = analyzer.get_salary_analysis()
performance_analysis = analyzer.get_performance_analysis()

print("Department statistics:")
print(dept_stats)
print("\nSalary analysis:")
print(salary_analysis)
print("\nPerformance analysis:")
print(performance_analysis)
```

### 2. Error Handling

```python
def safe_analysis(df, analysis_type):
    """Safely perform data analysis with error handling."""
    try:
        if analysis_type == 'department':
            return df.groupby('department')['salary'].mean()
        elif analysis_type == 'performance':
            return df.groupby('department')['performance_rating'].mean()
        elif analysis_type == 'age':
            return df.groupby('department')['age'].mean()
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
    except KeyError as e:
        print(f"Missing column: {e}")
        return None
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None

# Test safe analysis
results = safe_analysis(df, 'department')
print(f"Department salary analysis: {results}")
```

### 3. Documentation

```python
def analyze_employee_turnover(df, threshold_days=365):
    """
    Analyze employee turnover patterns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Employee data with hire_date column
    threshold_days : int, default 365
        Number of days to consider for turnover analysis
    
    Returns:
    --------
    dict
        Dictionary containing turnover statistics
    
    Examples:
    ---------
    >>> turnover_stats = analyze_employee_turnover(employee_df)
    >>> print(turnover_stats['turnover_rate'])
    """
    # Calculate days since hire
    days_since_hire = (pd.Timestamp.now() - df['hire_date']).dt.days
    
    # Identify employees who left
    left_employees = days_since_hire > threshold_days
    
    # Calculate turnover rate
    turnover_rate = left_employees.mean()
    
    # Analyze by department
    dept_turnover = df.groupby('department').apply(
        lambda x: ((pd.Timestamp.now() - x['hire_date']).dt.days > threshold_days).mean()
    )
    
    return {
        'turnover_rate': turnover_rate,
        'department_turnover': dept_turnover,
        'total_employees': len(df),
        'employees_left': left_employees.sum()
    }

# Example usage
turnover_analysis = analyze_employee_turnover(df)
print("Turnover analysis:")
for key, value in turnover_analysis.items():
    print(f"{key}: {value}")
```

## Summary

This guide covered comprehensive data analysis with pandas:

1. **Grouping and Aggregation**: Summarizing data by categories
2. **Pivot Tables**: Multi-dimensional data analysis
3. **Data Merging**: Combining datasets effectively
4. **Time Series Analysis**: Working with temporal data
5. **Statistical Analysis**: Descriptive and inferential statistics
6. **Data Transformation**: Feature engineering and scaling
7. **Advanced Filtering**: Complex data selection
8. **Data Validation**: Ensuring data quality
9. **Performance Optimization**: Efficient data processing
10. **Real-World Applications**: Practical use cases
11. **Best Practices**: Writing maintainable code

### Key Takeaways

- **GroupBy operations** are powerful for data summarization
- **Pivot tables** provide multi-dimensional analysis capabilities
- **Time series analysis** is essential for temporal data
- **Statistical analysis** helps understand data patterns
- **Performance optimization** is crucial for large datasets
- **Data validation** ensures analysis reliability

### Next Steps

- Practice with real-world datasets
- Explore advanced visualization techniques
- Learn about machine learning integration
- Study big data processing with pandas
- Master time series forecasting

### Additional Resources

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [Pandas Cookbook](https://pandas.pydata.org/docs/user_guide/cookbook.html)
- [Python for Data Analysis](https://wesmckinney.com/book/)

---

**Ready to explore data visualization and advanced analytics? Check out the visualization and machine learning guides!** 