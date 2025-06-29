# Pandas Basics: A Comprehensive Guide

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

## Table of Contents

1. [Introduction to Pandas](#introduction-to-pandas)
2. [Series: One-Dimensional Data](#series-one-dimensional-data)
3. [DataFrame: Two-Dimensional Data](#dataframe-two-dimensional-data)
4. [Data Loading and Export](#data-loading-and-export)
5. [Data Exploration](#data-exploration)
6. [Data Selection and Indexing](#data-selection-and-indexing)
7. [Data Cleaning](#data-cleaning)
8. [Data Manipulation](#data-manipulation)
9. [Basic Operations](#basic-operations)
10. [Best Practices](#best-practices)
11. [Common Pitfalls](#common-pitfalls)
12. [Performance Tips](#performance-tips)

## Introduction to Pandas

Pandas is a powerful data manipulation and analysis library for Python. It provides:

- **Series**: One-dimensional labeled arrays
- **DataFrame**: Two-dimensional labeled data structures
- **Data loading**: Read from various file formats
- **Data cleaning**: Handle missing values and data types
- **Data analysis**: Statistical operations and aggregations
- **Data visualization**: Basic plotting capabilities

### Importing Pandas

```python
import pandas as pd
import numpy as np

# Common aliases
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)
```

## Series: One-Dimensional Data

### 1. Creating Series

```python
# From list
s1 = pd.Series([1, 2, 3, 4, 5])
print(f"Series from list:\n{s1}")
print(f"Type: {type(s1)}")
print(f"Index: {s1.index}")
print(f"Values: {s1.values}")

# From list with custom index
s2 = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(f"\nSeries with custom index:\n{s2}")

# From dictionary
data_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
s3 = pd.Series(data_dict)
print(f"\nSeries from dictionary:\n{s3}")

# From scalar value
s4 = pd.Series(5, index=['a', 'b', 'c', 'd'])
print(f"\nSeries from scalar:\n{s4}")
```

### 2. Series Operations

```python
# Basic arithmetic
s1 = pd.Series([1, 2, 3, 4, 5])
s2 = pd.Series([10, 20, 30, 40, 50])

print(f"Series 1: {s1}")
print(f"Series 2: {s2}")
print(f"Addition: {s1 + s2}")
print(f"Multiplication: {s1 * s2}")
print(f"Square root: {np.sqrt(s1)}")

# Boolean operations
print(f"\nGreater than 3: {s1 > 3}")
print(f"Even numbers: {s1 % 2 == 0}")

# Statistical operations
print(f"\nMean: {s1.mean()}")
print(f"Standard deviation: {s1.std()}")
print(f"Sum: {s1.sum()}")
print(f"Minimum: {s1.min()}")
print(f"Maximum: {s1.max()}")
```

### 3. Series Indexing

```python
s = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])

print(f"Series: {s}")

# Integer indexing
print(f"First element: {s.iloc[0]}")
print(f"Last element: {s.iloc[-1]}")
print(f"Slice [1:3]: {s.iloc[1:3]}")

# Label indexing
print(f"Element 'c': {s.loc['c']}")
print(f"Elements 'a' and 'd': {s.loc[['a', 'd']]}")

# Boolean indexing
mask = s > 25
print(f"Elements > 25: {s[mask]}")

# Mixed indexing
print(f"First 3 elements: {s.head(3)}")
print(f"Last 3 elements: {s.tail(3)}")
```

## DataFrame: Two-Dimensional Data

### 1. Creating DataFrames

```python
# From dictionary
data_dict = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['NYC', 'LA', 'Chicago', 'Boston'],
    'Salary': [50000, 60000, 70000, 55000]
}
df = pd.DataFrame(data_dict)
print(f"DataFrame from dictionary:\n{df}")

# From list of lists
data_list = [
    ['Alice', 25, 'NYC', 50000],
    ['Bob', 30, 'LA', 60000],
    ['Charlie', 35, 'Chicago', 70000],
    ['Diana', 28, 'Boston', 55000]
]
columns = ['Name', 'Age', 'City', 'Salary']
df2 = pd.DataFrame(data_list, columns=columns)
print(f"\nDataFrame from list:\n{df2}")

# From NumPy array
arr = np.random.randn(4, 3)
df3 = pd.DataFrame(arr, columns=['A', 'B', 'C'])
print(f"\nDataFrame from NumPy array:\n{df3}")

# From Series
series_dict = {
    'A': pd.Series([1, 2, 3]),
    'B': pd.Series([4, 5, 6, 7])  # Note: different lengths
}
df4 = pd.DataFrame(series_dict)
print(f"\nDataFrame from Series:\n{df4}")
```

### 2. DataFrame Properties

```python
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'City': ['NYC', 'LA', 'Chicago', 'Boston'],
    'Salary': [50000, 60000, 70000, 55000]
})

print(f"DataFrame:\n{df}")
print(f"\nShape: {df.shape}")
print(f"Size: {df.size}")
print(f"Number of dimensions: {df.ndim}")
print(f"Data types:\n{df.dtypes}")
print(f"Index: {df.index}")
print(f"Columns: {df.columns}")
print(f"Values:\n{df.values}")
```

### 3. DataFrame Information

```python
# Basic information
print(f"Info:\n{df.info()}")

# Statistical summary
print(f"\nDescribe:\n{df.describe()}")

# Memory usage
print(f"\nMemory usage:\n{df.memory_usage()}")

# Null values
print(f"\nNull values:\n{df.isnull().sum()}")

# Unique values
print(f"\nUnique cities: {df['City'].unique()}")
print(f"Value counts for cities:\n{df['City'].value_counts()}")
```

## Data Loading and Export

### 1. Reading Data

```python
# Reading CSV files
# df = pd.read_csv('data.csv')

# Reading Excel files
# df = pd.read_excel('data.xlsx', sheet_name='Sheet1')

# Reading JSON files
# df = pd.read_json('data.json')

# Reading from database
# df = pd.read_sql('SELECT * FROM table', connection)

# Reading from clipboard
# df = pd.read_clipboard()

# Example with sample data
sample_data = """Name,Age,City,Salary
Alice,25,NYC,50000
Bob,30,LA,60000
Charlie,35,Chicago,70000
Diana,28,Boston,55000"""

df = pd.read_csv(pd.StringIO(sample_data))
print(f"Loaded data:\n{df}")
```

### 2. Writing Data

```python
# Writing to CSV
df.to_csv('output.csv', index=False)

# Writing to Excel
df.to_excel('output.xlsx', sheet_name='Data', index=False)

# Writing to JSON
df.to_json('output.json', orient='records')

# Writing to HTML
df.to_html('output.html', index=False)

# Writing to clipboard
df.to_clipboard(index=False)

print("Data exported to various formats")
```

### 3. Data Format Options

```python
# CSV options
df.to_csv('output.csv', 
          index=False,           # Don't include index
          sep=',',              # Separator
          encoding='utf-8',     # Encoding
          na_rep='NA')          # Missing value representation

# Excel options
with pd.ExcelWriter('output.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Data', index=False)
    df.describe().to_excel(writer, sheet_name='Summary')

# JSON options
df.to_json('output.json', 
           orient='records',     # Format: list of dictionaries
           indent=2,            # Pretty print
           date_format='iso')   # Date format
```

## Data Exploration

### 1. Viewing Data

```python
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank'],
    'Age': [25, 30, 35, 28, 32, 29],
    'City': ['NYC', 'LA', 'Chicago', 'Boston', 'Seattle', 'Miami'],
    'Salary': [50000, 60000, 70000, 55000, 65000, 58000],
    'Department': ['IT', 'HR', 'IT', 'Sales', 'IT', 'HR']
})

print(f"Full DataFrame:\n{df}")
print(f"\nFirst 3 rows:\n{df.head(3)}")
print(f"\nLast 3 rows:\n{df.tail(3)}")
print(f"\nRandom 3 rows:\n{df.sample(3)}")
```

### 2. Data Information

```python
# Basic information
print(f"DataFrame info:")
df.info()

print(f"\nData types:")
print(df.dtypes)

print(f"\nShape: {df.shape}")
print(f"Number of rows: {len(df)}")
print(f"Number of columns: {df.shape[1]}")

# Statistical summary
print(f"\nNumeric columns summary:")
print(df.describe())

print(f"\nAll columns summary:")
print(df.describe(include='all'))
```

### 3. Data Quality Checks

```python
# Missing values
print(f"Missing values per column:")
print(df.isnull().sum())

print(f"\nMissing values percentage:")
print(df.isnull().sum() / len(df) * 100)

# Duplicates
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

# Unique values
print(f"\nUnique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# Value counts
print(f"\nValue counts for categorical columns:")
for col in df.select_dtypes(include=['object']).columns:
    print(f"\n{col}:")
    print(df[col].value_counts())
```

## Data Selection and Indexing

### 1. Column Selection

```python
# Single column
names = df['Name']
print(f"Names column:\n{names}")

# Multiple columns
subset = df[['Name', 'Age', 'Salary']]
print(f"\nSubset of columns:\n{subset}")

# Using dot notation (only for valid column names)
ages = df.Age
print(f"\nAges using dot notation:\n{ages}")

# Column types
numeric_cols = df.select_dtypes(include=[np.number])
print(f"\nNumeric columns:\n{numeric_cols}")

categorical_cols = df.select_dtypes(include=['object'])
print(f"\nCategorical columns:\n{categorical_cols}")
```

### 2. Row Selection

```python
# Integer indexing with iloc
print(f"First row:\n{df.iloc[0]}")
print(f"\nFirst 3 rows:\n{df.iloc[:3]}")
print(f"\nRows 1 to 3:\n{df.iloc[1:4]}")
print(f"\nLast row:\n{df.iloc[-1]}")

# Label indexing with loc
print(f"\nRow with index 0:\n{df.loc[0]}")
print(f"\nRows with specific indices:\n{df.loc[[0, 2, 4]]}")

# Boolean indexing
high_salary = df[df['Salary'] > 60000]
print(f"\nHigh salary employees:\n{high_salary}")

it_employees = df[df['Department'] == 'IT']
print(f"\nIT employees:\n{it_employees}")

# Multiple conditions
young_it = df[(df['Age'] < 30) & (df['Department'] == 'IT')]
print(f"\nYoung IT employees:\n{young_it}")
```

### 3. Advanced Indexing

```python
# Using query method
young_high_salary = df.query('Age < 30 and Salary > 55000')
print(f"Young high salary employees:\n{young_high_salary}")

# Using isin
cities = ['NYC', 'LA', 'Boston']
selected_cities = df[df['City'].isin(cities)]
print(f"\nEmployees in selected cities:\n{selected_cities}")

# Using between
age_range = df[df['Age'].between(25, 35)]
print(f"\nEmployees aged 25-35:\n{age_range}")

# Using string methods
names_with_a = df[df['Name'].str.contains('a', case=False)]
print(f"\nNames containing 'a':\n{names_with_a}")
```

## Data Cleaning

### 1. Handling Missing Values

```python
# Create DataFrame with missing values
df_missing = pd.DataFrame({
    'Name': ['Alice', 'Bob', None, 'Diana'],
    'Age': [25, None, 35, 28],
    'City': ['NYC', 'LA', 'Chicago', None],
    'Salary': [50000, 60000, None, 55000]
})

print(f"DataFrame with missing values:\n{df_missing}")
print(f"\nMissing values:\n{df_missing.isnull().sum()}")

# Remove rows with missing values
df_clean = df_missing.dropna()
print(f"\nAfter dropping missing values:\n{df_clean}")

# Fill missing values
df_filled = df_missing.fillna({
    'Name': 'Unknown',
    'Age': df_missing['Age'].mean(),
    'City': 'Unknown',
    'Salary': df_missing['Salary'].median()
})
print(f"\nAfter filling missing values:\n{df_filled}")

# Forward fill
df_ffill = df_missing.fillna(method='ffill')
print(f"\nAfter forward fill:\n{df_ffill}")

# Backward fill
df_bfill = df_missing.fillna(method='bfill')
print(f"\nAfter backward fill:\n{df_bfill}")
```

### 2. Handling Duplicates

```python
# Create DataFrame with duplicates
df_duplicates = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
    'Age': [25, 30, 25, 35, 30],
    'City': ['NYC', 'LA', 'NYC', 'Chicago', 'LA']
})

print(f"DataFrame with duplicates:\n{df_duplicates}")
print(f"\nNumber of duplicates: {df_duplicates.duplicated().sum()}")

# Remove duplicates
df_no_duplicates = df_duplicates.drop_duplicates()
print(f"\nAfter removing duplicates:\n{df_no_duplicates}")

# Remove duplicates based on specific columns
df_no_name_duplicates = df_duplicates.drop_duplicates(subset=['Name'])
print(f"\nAfter removing name duplicates:\n{df_no_name_duplicates}")

# Keep last occurrence
df_keep_last = df_duplicates.drop_duplicates(keep='last')
print(f"\nKeeping last occurrence:\n{df_keep_last}")
```

### 3. Data Type Conversion

```python
# Create DataFrame with mixed types
df_mixed = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': ['25', '30', '35'],
    'Salary': ['50000', '60000', '70000'],
    'Date': ['2023-01-01', '2023-02-01', '2023-03-01']
})

print(f"Original data types:\n{df_mixed.dtypes}")

# Convert data types
df_converted = df_mixed.copy()
df_converted['Age'] = df_converted['Age'].astype(int)
df_converted['Salary'] = df_converted['Salary'].astype(float)
df_converted['Date'] = pd.to_datetime(df_converted['Date'])

print(f"\nConverted data types:\n{df_converted.dtypes}")
print(f"\nConverted DataFrame:\n{df_converted}")
```

## Data Manipulation

### 1. Adding and Removing Columns

```python
# Add new column
df['Bonus'] = df['Salary'] * 0.1
print(f"After adding bonus column:\n{df}")

# Add column based on condition
df['High_Salary'] = df['Salary'] > 60000
print(f"\nAfter adding high salary flag:\n{df}")

# Add column using apply
df['Salary_Category'] = df['Salary'].apply(
    lambda x: 'High' if x > 60000 else 'Medium' if x > 50000 else 'Low'
)
print(f"\nAfter adding salary category:\n{df}")

# Remove columns
df_clean = df.drop(['Bonus', 'High_Salary'], axis=1)
print(f"\nAfter removing columns:\n{df_clean}")

# Remove columns inplace
df.drop(['Salary_Category'], axis=1, inplace=True)
print(f"\nAfter inplace removal:\n{df}")
```

### 2. Sorting and Ranking

```python
# Sort by single column
df_sorted = df.sort_values('Salary')
print(f"Sorted by salary:\n{df_sorted}")

# Sort by multiple columns
df_multi_sorted = df.sort_values(['Department', 'Salary'], ascending=[True, False])
print(f"\nSorted by department (asc) and salary (desc):\n{df_multi_sorted}")

# Ranking
df['Salary_Rank'] = df['Salary'].rank(ascending=False)
print(f"\nWith salary ranking:\n{df}")

# Percentile ranking
df['Salary_Percentile'] = df['Salary'].rank(pct=True)
print(f"\nWith salary percentile:\n{df}")
```

### 3. Reshaping Data

```python
# Melt (wide to long)
df_melt = df.melt(
    id_vars=['Name', 'Department'],
    value_vars=['Age', 'Salary'],
    var_name='Attribute',
    value_name='Value'
)
print(f"Melted DataFrame:\n{df_melt}")

# Pivot (long to wide)
df_pivot = df_melt.pivot(
    index='Name',
    columns='Attribute',
    values='Value'
)
print(f"\nPivoted DataFrame:\n{df_pivot}")

# Pivot table (with aggregation)
df_pivot_table = df.pivot_table(
    values='Salary',
    index='Department',
    columns='City',
    aggfunc='mean',
    fill_value=0
)
print(f"\nPivot table:\n{df_pivot_table}")
```

## Basic Operations

### 1. Mathematical Operations

```python
# Basic arithmetic
df['Salary_K'] = df['Salary'] / 1000
df['Age_Squared'] = df['Age'] ** 2
df['Salary_Age_Ratio'] = df['Salary'] / df['Age']

print(f"After mathematical operations:\n{df}")

# Statistical operations
print(f"\nSalary statistics:")
print(f"Mean: {df['Salary'].mean()}")
print(f"Median: {df['Salary'].median()}")
print(f"Standard deviation: {df['Salary'].std()}")
print(f"Variance: {df['Salary'].var()}")
print(f"Skewness: {df['Salary'].skew()}")
print(f"Kurtosis: {df['Salary'].kurtosis()}")
```

### 2. String Operations

```python
# String methods
df['Name_Upper'] = df['Name'].str.upper()
df['Name_Lower'] = df['Name'].str.lower()
df['Name_Length'] = df['Name'].str.len()
df['First_Letter'] = df['Name'].str[0]

print(f"After string operations:\n{df}")

# String contains
names_with_a = df[df['Name'].str.contains('a', case=False)]
print(f"\nNames containing 'a':\n{names_with_a}")

# String replace
df['City_Short'] = df['City'].str.replace('New York', 'NY')
print(f"\nAfter string replacement:\n{df}")
```

### 3. Date and Time Operations

```python
# Create DataFrame with dates
df_dates = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Hire_Date': ['2023-01-15', '2023-02-20', '2023-03-10'],
    'Birth_Date': ['1995-05-10', '1990-08-15', '1988-12-03']
})

# Convert to datetime
df_dates['Hire_Date'] = pd.to_datetime(df_dates['Hire_Date'])
df_dates['Birth_Date'] = pd.to_datetime(df_dates['Birth_Date'])

# Date operations
df_dates['Hire_Year'] = df_dates['Hire_Date'].dt.year
df_dates['Hire_Month'] = df_dates['Hire_Date'].dt.month
df_dates['Hire_Day'] = df_dates['Hire_Date'].dt.day
df_dates['Age_at_Hire'] = (df_dates['Hire_Date'] - df_dates['Birth_Date']).dt.days / 365.25

print(f"DataFrame with date operations:\n{df_dates}")
```

## Best Practices

### 1. Memory Efficiency

```python
# Check memory usage
print(f"Memory usage:\n{df.memory_usage(deep=True)}")

# Optimize data types
def optimize_dtypes(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to category if few unique values
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        elif df[col].dtype == 'int64':
            # Try to downcast integers
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'float64':
            # Try to downcast floats
            df[col] = pd.to_numeric(df[col], downcast='float')
    return df

df_optimized = optimize_dtypes(df.copy())
print(f"\nMemory usage after optimization:\n{df_optimized.memory_usage(deep=True)}")
```

### 2. Code Organization

```python
# Good: Use descriptive variable names
employee_data = pd.DataFrame({
    'employee_name': ['Alice', 'Bob', 'Charlie'],
    'employee_age': [25, 30, 35],
    'employee_salary': [50000, 60000, 70000]
})

# Good: Use constants for magic numbers
SALARY_THRESHOLD = 60000
AGE_THRESHOLD = 30

high_earners = employee_data[employee_data['employee_salary'] > SALARY_THRESHOLD]
young_employees = employee_data[employee_data['employee_age'] < AGE_THRESHOLD]

# Good: Document complex operations
def calculate_bonus(salary, performance_rating):
    """
    Calculate bonus based on salary and performance rating.
    
    Parameters:
    salary: Employee salary
    performance_rating: Performance rating (1-5)
    
    Returns:
    float: Calculated bonus
    """
    base_bonus_rate = 0.1
    performance_multiplier = performance_rating / 5
    return salary * base_bonus_rate * performance_multiplier
```

### 3. Error Handling

```python
def safe_read_csv(file_path):
    """Safely read CSV file with error handling."""
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File {file_path} is empty")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def safe_operation(df, operation):
    """Safely perform operation on DataFrame."""
    try:
        result = operation(df)
        return result
    except Exception as e:
        print(f"Error during operation: {e}")
        return None
```

## Common Pitfalls

### 1. SettingWithCopyWarning

```python
# Pitfall: Chained indexing
df_subset = df[df['Salary'] > 60000]
df_subset['Bonus'] = 1000  # This may cause SettingWithCopyWarning

# Solution: Use .loc or .copy()
df_subset = df[df['Salary'] > 60000].copy()
df_subset['Bonus'] = 1000

# Or use .loc
df.loc[df['Salary'] > 60000, 'Bonus'] = 1000
```

### 2. Data Type Issues

```python
# Pitfall: Mixed data types
mixed_data = ['1', '2', '3', '4', '5']
series = pd.Series(mixed_data)
print(f"Data type: {series.dtype}")  # object

# Solution: Convert to appropriate type
numeric_series = pd.to_numeric(series, errors='coerce')
print(f"Converted data type: {numeric_series.dtype}")  # float64
```

### 3. Memory Issues

```python
# Pitfall: Large DataFrames in memory
# df = pd.read_csv('large_file.csv')  # May cause memory issues

# Solution: Read in chunks
chunk_list = []
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    chunk_list.append(chunk)
df = pd.concat(chunk_list, ignore_index=True)

# Or use dask for very large datasets
# import dask.dataframe as dd
# df = dd.read_csv('large_file.csv')
```

## Performance Tips

### 1. Vectorized Operations

```python
import time

# Slow: Using apply
start = time.time()
df['Salary_Category'] = df['Salary'].apply(
    lambda x: 'High' if x > 60000 else 'Medium' if x > 50000 else 'Low'
)
slow_time = time.time() - start

# Fast: Using vectorized operations
start = time.time()
df['Salary_Category_Fast'] = pd.cut(
    df['Salary'], 
    bins=[0, 50000, 60000, float('inf')], 
    labels=['Low', 'Medium', 'High']
)
fast_time = time.time() - start

print(f"Slow method: {slow_time:.6f}s")
print(f"Fast method: {fast_time:.6f}s")
print(f"Speedup: {slow_time/fast_time:.1f}x")
```

### 2. Efficient Filtering

```python
# Efficient: Use query for complex conditions
efficient_filter = df.query('Salary > 60000 and Age < 30')

# Efficient: Use isin for multiple values
cities = ['NYC', 'LA', 'Boston']
efficient_cities = df[df['City'].isin(cities)]

# Efficient: Use between for ranges
efficient_age = df[df['Age'].between(25, 35)]
```

### 3. Memory Optimization

```python
# Use appropriate data types
df_optimized = df.copy()

# Convert object columns to category if appropriate
for col in df_optimized.select_dtypes(include=['object']):
    if df_optimized[col].nunique() / len(df_optimized) < 0.5:
        df_optimized[col] = df_optimized[col].astype('category')

# Downcast numeric types
for col in df_optimized.select_dtypes(include=['int64']):
    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')

for col in df_optimized.select_dtypes(include=['float64']):
    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')

print(f"Original memory usage: {df.memory_usage(deep=True).sum()}")
print(f"Optimized memory usage: {df_optimized.memory_usage(deep=True).sum()}")
```

## Summary

This guide covered the fundamental concepts of pandas:

1. **Series and DataFrames**: Core data structures
2. **Data Loading and Export**: Reading and writing data
3. **Data Exploration**: Understanding your data
4. **Data Selection and Indexing**: Accessing data efficiently
5. **Data Cleaning**: Handling missing values and duplicates
6. **Data Manipulation**: Transforming and reshaping data
7. **Basic Operations**: Mathematical and string operations
8. **Best Practices**: Writing efficient and maintainable code
9. **Common Pitfalls**: Avoiding typical mistakes
10. **Performance Tips**: Optimizing for speed and memory

### Key Takeaways

- **Pandas** provides powerful data manipulation capabilities
- **Vectorized operations** are much faster than loops
- **Memory optimization** is important for large datasets
- **Data cleaning** is a crucial step in data analysis
- **Best practices** help write maintainable code

### Next Steps

- Practice with real-world datasets
- Explore advanced pandas features
- Learn about data visualization
- Study time series analysis
- Master data aggregation and grouping

### Additional Resources

- [Pandas Official Documentation](https://pandas.pydata.org/docs/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
- [Pandas API Reference](https://pandas.pydata.org/docs/reference/index.html)
- [Pandas Cookbook](https://pandas.pydata.org/docs/user_guide/cookbook.html)

---

**Ready to explore more advanced pandas features? Check out the data analysis and visualization guides!** 