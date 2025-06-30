# Data Manipulation Guide

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Level](https://img.shields.io/badge/Level-Intermediate-yellow.svg)](https://github.com/yourusername/Toolkit)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-blue.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-orange.svg)](https://numpy.org/)
[![Topics](https://img.shields.io/badge/Topics-Data%20Processing%2C%20ETL%2C%20Cleaning-orange.svg)](https://github.com/yourusername/Toolkit)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](https://github.com/yourusername/Toolkit)

A comprehensive guide to data manipulation techniques in Python for AI/ML and data science applications.

## Table of Contents
1. [Working with Different Data Formats](#working-with-different-data-formats)
2. [Data Structures and Operations](#data-structures-and-operations)
3. [Data Transformation](#data-transformation)
4. [Data Aggregation](#data-aggregation)
5. [Data Merging and Joining](#data-merging-and-joining)
6. [Data Reshaping](#data-reshaping)
7. [Working with Time Series Data](#working-with-time-series-data)
8. [Data Validation](#data-validation)
9. [Performance Optimization](#performance-optimization)

## Working with Different Data Formats

### CSV Files

```python
import pandas as pd
import csv

# Reading CSV with pandas
df = pd.read_csv('data.csv')
df = pd.read_csv('data.csv', index_col=0)  # Use first column as index
df = pd.read_csv('data.csv', header=None)  # No header row

# Reading CSV with standard library
with open('data.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        print(row)

# Writing CSV
df.to_csv('output.csv', index=False)
df.to_csv('output.csv', index=True, header=True)

# Custom CSV writing
with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Age', 'City'])
    writer.writerows([['John', 30, 'NYC'], ['Jane', 25, 'LA']])
```

### JSON Files

```python
import json

# Reading JSON
with open('data.json', 'r') as file:
    data = json.load(file)

# Reading JSON with pandas
df = pd.read_json('data.json')
df = pd.read_json('data.json', orient='records')

# Writing JSON
with open('output.json', 'w') as file:
    json.dump(data, file, indent=2)

# Writing JSON with pandas
df.to_json('output.json', orient='records', indent=2)

# Working with JSON strings
json_string = '{"name": "John", "age": 30}'
data = json.loads(json_string)
```

### Excel Files

```python
# Reading Excel files
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df = pd.read_excel('data.xlsx', sheet_name=0)  # First sheet

# Reading multiple sheets
all_sheets = pd.read_excel('data.xlsx', sheet_name=None)
sheet1 = all_sheets['Sheet1']

# Writing Excel files
df.to_excel('output.xlsx', sheet_name='Data', index=False)

# Writing multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sheet1', index=False)
    df2.to_excel(writer, sheet_name='Sheet2', index=False)
```

### Database Connections

```python
import sqlite3
import psycopg2
from sqlalchemy import create_engine

# SQLite
conn = sqlite3.connect('database.db')
df = pd.read_sql_query("SELECT * FROM table_name", conn)
conn.close()

# PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="mydb",
    user="username",
    password="password"
)
df = pd.read_sql_query("SELECT * FROM table_name", conn)

# SQLAlchemy
engine = create_engine('postgresql://username:password@localhost/mydb')
df = pd.read_sql_query("SELECT * FROM table_name", engine)

# Writing to database
df.to_sql('table_name', engine, if_exists='replace', index=False)
```

## Data Structures and Operations

### Pandas DataFrame Basics

```python
import pandas as pd
import numpy as np

# Creating DataFrames
data = {
    'name': ['John', 'Jane', 'Bob', 'Alice'],
    'age': [30, 25, 35, 28],
    'city': ['NYC', 'LA', 'Chicago', 'Boston'],
    'salary': [75000, 65000, 80000, 70000]
}
df = pd.DataFrame(data)

# From lists
names = ['John', 'Jane', 'Bob']
ages = [30, 25, 35]
df = pd.DataFrame({'name': names, 'age': ages})

# From NumPy arrays
array = np.random.randn(100, 3)
df = pd.DataFrame(array, columns=['A', 'B', 'C'])

# Basic operations
print(df.shape)        # (rows, columns)
print(df.dtypes)       # Data types
print(df.columns)      # Column names
print(df.index)        # Index
print(df.head())       # First 5 rows
print(df.tail())       # Last 5 rows
print(df.info())       # Summary information
print(df.describe())   # Statistical summary
```

### Data Selection and Indexing

```python
# Column selection
df['name']                    # Single column
df[['name', 'age']]          # Multiple columns
df.name                      # Attribute access

# Row selection
df.iloc[0]                   # First row by position
df.iloc[0:5]                 # Rows 0-4
df.iloc[[0, 2, 4]]          # Specific rows
df.loc[0]                    # First row by label
df.loc[0:5]                  # Rows 0-5 (inclusive)

# Boolean indexing
df[df['age'] > 30]           # Filter by condition
df[(df['age'] > 30) & (df['salary'] > 70000)]  # Multiple conditions
df[df['city'].isin(['NYC', 'LA'])]             # In operator

# Query method
df.query('age > 30 and salary > 70000')
df.query('city in ["NYC", "LA"]')
```

### Data Types and Conversion

```python
# Checking data types
print(df.dtypes)

# Converting data types
df['age'] = df['age'].astype(int)
df['salary'] = df['salary'].astype(float)
df['name'] = df['name'].astype('category')

# Converting to datetime
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Converting to numeric
df['value'] = pd.to_numeric(df['value'], errors='coerce')

# Custom conversion
def convert_salary(x):
    return float(x.replace('$', '').replace(',', ''))
df['salary'] = df['salary'].apply(convert_salary)
```

## Data Transformation

### Handling Missing Data

```python
# Checking for missing data
print(df.isnull().sum())
print(df.isnull().any())
print(df.isnull().sum().sum())

# Removing missing data
df.dropna()                    # Remove rows with any NaN
df.dropna(subset=['age'])      # Remove rows with NaN in 'age'
df.dropna(how='all')           # Remove rows with all NaN
df.dropna(axis=1)              # Remove columns with any NaN

# Filling missing data
df.fillna(0)                   # Fill with 0
df.fillna(df.mean())           # Fill with mean
df.fillna(df.median())         # Fill with median
df.fillna(method='ffill')      # Forward fill
df.fillna(method='bfill')      # Backward fill

# Interpolation
df.interpolate()               # Linear interpolation
df.interpolate(method='time')  # Time-based interpolation
```

### Data Cleaning

```python
# Removing duplicates
df.drop_duplicates()                    # Remove all duplicates
df.drop_duplicates(subset=['name'])     # Remove duplicates based on column
df.drop_duplicates(keep='first')        # Keep first occurrence

# String cleaning
df['name'] = df['name'].str.strip()     # Remove whitespace
df['name'] = df['name'].str.lower()     # Convert to lowercase
df['name'] = df['name'].str.replace(' ', '_')  # Replace spaces

# Regular expressions
import re
df['phone'] = df['phone'].str.replace(r'\D', '', regex=True)  # Keep only digits
df['email'] = df['email'].str.extract(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})')

# Value replacement
df['gender'] = df['gender'].replace({'M': 'Male', 'F': 'Female'})
df['status'] = df['status'].map({'A': 'Active', 'I': 'Inactive'})
```

### Feature Engineering

```python
# Creating new columns
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 100], 
                        labels=['Young', 'Adult', 'Middle', 'Senior'])
df['salary_category'] = pd.qcut(df['salary'], q=4, 
                               labels=['Low', 'Medium', 'High', 'Very High'])

# Date features
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter

# String features
df['name_length'] = df['name'].str.len()
df['first_name'] = df['name'].str.split().str[0]
df['last_name'] = df['name'].str.split().str[-1]

# Mathematical transformations
df['log_salary'] = np.log(df['salary'])
df['salary_squared'] = df['salary'] ** 2
df['age_salary_ratio'] = df['age'] / df['salary']
```

## Data Aggregation

### GroupBy Operations

```python
# Basic grouping
grouped = df.groupby('city')
print(grouped.size())                    # Count by group
print(grouped['salary'].mean())          # Mean salary by city
print(grouped['age'].agg(['mean', 'std', 'min', 'max']))  # Multiple aggregations

# Multiple group columns
grouped = df.groupby(['city', 'gender'])
print(grouped['salary'].mean())

# Custom aggregation functions
def custom_agg(x):
    return {'mean': x.mean(), 'count': len(x), 'std': x.std()}

result = df.groupby('city')['salary'].apply(custom_agg)

# Aggregating multiple columns
result = df.groupby('city').agg({
    'salary': ['mean', 'std'],
    'age': ['min', 'max'],
    'name': 'count'
})
```

### Pivot Tables

```python
# Basic pivot table
pivot = df.pivot_table(
    values='salary',
    index='city',
    columns='gender',
    aggfunc='mean'
)

# Multiple aggregations
pivot = df.pivot_table(
    values='salary',
    index='city',
    columns='gender',
    aggfunc=['mean', 'count', 'std'],
    fill_value=0
)

# Cross tabulation
crosstab = pd.crosstab(df['city'], df['gender'], values=df['salary'], aggfunc='mean')
```

### Window Functions

```python
# Rolling window
df['salary_rolling_mean'] = df['salary'].rolling(window=3).mean()
df['salary_rolling_std'] = df['salary'].rolling(window=3).std()

# Expanding window
df['salary_expanding_mean'] = df['salary'].expanding().mean()

# Grouped rolling
df['salary_group_rolling'] = df.groupby('city')['salary'].rolling(window=2).mean()

# Ranking
df['salary_rank'] = df['salary'].rank(ascending=False)
df['salary_rank_by_city'] = df.groupby('city')['salary'].rank(ascending=False)
```

## Data Merging and Joining

### Concatenation

```python
# Vertical concatenation (stacking)
df_combined = pd.concat([df1, df2], axis=0, ignore_index=True)
df_combined = pd.concat([df1, df2], axis=0, keys=['set1', 'set2'])

# Horizontal concatenation (side by side)
df_combined = pd.concat([df1, df2], axis=1)

# Appending
df_combined = df1.append(df2, ignore_index=True)
```

### Merging DataFrames

```python
# Inner join
merged = pd.merge(df1, df2, on='id', how='inner')

# Left join
merged = pd.merge(df1, df2, on='id', how='left')

# Right join
merged = pd.merge(df1, df2, on='id', how='right')

# Outer join
merged = pd.merge(df1, df2, on='id', how='outer')

# Multiple join keys
merged = pd.merge(df1, df2, on=['id', 'date'], how='inner')

# Different column names
merged = pd.merge(df1, df2, left_on='id1', right_on='id2', how='inner')
```

### Joining DataFrames

```python
# Join on index
joined = df1.join(df2, how='inner')

# Join with different index
joined = df1.join(df2.set_index('id'), on='id', how='inner')
```

## Data Reshaping

### Melting (Wide to Long)

```python
# Basic melt
melted = pd.melt(df, id_vars=['name', 'city'], 
                value_vars=['salary_2020', 'salary_2021', 'salary_2022'],
                var_name='year', value_name='salary')

# Melt with custom variable names
melted = pd.melt(df, id_vars=['name'], 
                value_vars=['age', 'salary'],
                var_name='metric', value_name='value')
```

### Pivoting (Long to Wide)

```python
# Basic pivot
pivoted = df.pivot(index='name', columns='year', values='salary')

# Pivot with multiple values
pivoted = df.pivot(index='name', columns='metric', values='value')
```

### Stacking and Unstacking

```python
# Stack (columns to rows)
stacked = df.set_index(['name', 'city']).stack()

# Unstack (rows to columns)
unstacked = df.set_index(['name', 'city']).unstack('city')
```

## Working with Time Series Data

### Time Series Basics

```python
# Creating time series
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100), index=dates)

# Resampling
daily = ts.resample('D').mean()
weekly = ts.resample('W').sum()
monthly = ts.resample('M').mean()

# Shifting
ts_shifted = ts.shift(1)      # Shift forward
ts_lagged = ts.shift(-1)      # Shift backward

# Rolling statistics
rolling_mean = ts.rolling(window=7).mean()
rolling_std = ts.rolling(window=7).std()
```

### Time Series Operations

```python
# Time-based indexing
ts['2023-01']                 # January 2023
ts['2023-01-01':'2023-01-31'] # Date range
ts.between_time('09:00', '17:00')  # Time range

# Time zone handling
ts_utc = ts.tz_localize('UTC')
ts_est = ts_utc.tz_convert('US/Eastern')

# Period conversion
ts.to_period('M')             # Convert to monthly periods
ts.to_timestamp()             # Convert periods to timestamps
```

## Data Validation

### Data Quality Checks

```python
# Range validation
def validate_age(age):
    return 0 <= age <= 120

df['age_valid'] = df['age'].apply(validate_age)

# Format validation
import re
def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

df['email_valid'] = df['email'].apply(validate_email)

# Duplicate detection
duplicates = df.duplicated(subset=['email'], keep=False)
df_duplicates = df[duplicates]

# Outlier detection
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['salary'] < Q1 - 1.5 * IQR) | (df['salary'] > Q3 + 1.5 * IQR)]
```

### Data Profiling

```python
# Basic profiling
print(df.describe())
print(df.info())

# Missing data analysis
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_summary = pd.DataFrame({
    'Missing_Count': missing_data,
    'Missing_Percent': missing_percent
})

# Data type analysis
type_summary = df.dtypes.value_counts()

# Unique value analysis
unique_counts = df.nunique()
```

## Performance Optimization

### Memory Optimization

```python
# Memory usage analysis
print(df.memory_usage(deep=True))

# Optimizing data types
def optimize_dtypes(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        elif df[col].dtype == 'int64':
            if df[col].min() >= 0:
                if df[col].max() < 255:
                    df[col] = df[col].astype('uint8')
                elif df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
    return df

df_optimized = optimize_dtypes(df.copy())
```

### Efficient Operations

```python
# Vectorized operations (faster)
df['new_col'] = df['col1'] + df['col2']  # Vectorized
df['new_col'] = df.apply(lambda row: row['col1'] + row['col2'], axis=1)  # Slower

# Using NumPy for calculations
df['result'] = np.where(df['condition'], df['value1'], df['value2'])

# Efficient filtering
mask = df['condition']
filtered_df = df[mask]  # More efficient than df[df['condition']]

# Chunked processing for large files
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    process_chunk(chunk)
```

### Parallel Processing

```python
from multiprocessing import Pool
import pandas as pd

def process_data_chunk(chunk):
    # Process chunk
    return processed_chunk

# Split data into chunks
chunks = np.array_split(df, 4)

# Process in parallel
with Pool(4) as pool:
    results = pool.map(process_data_chunk, chunks)

# Combine results
final_result = pd.concat(results)
```

## Best Practices

### Code Organization

```python
# Separate data loading, processing, and saving
def load_data(file_path):
    """Load data from file"""
    return pd.read_csv(file_path)

def process_data(df):
    """Process and clean data"""
    # Data cleaning steps
    return df

def save_data(df, file_path):
    """Save processed data"""
    df.to_csv(file_path, index=False)

# Main workflow
df = load_data('raw_data.csv')
df_processed = process_data(df)
save_data(df_processed, 'processed_data.csv')
```

### Error Handling

```python
def safe_data_operation(df, operation):
    """Safely perform data operations with error handling"""
    try:
        result = operation(df)
        return result
    except Exception as e:
        print(f"Error in operation: {e}")
        return None

# Usage
result = safe_data_operation(df, lambda x: x.groupby('city')['salary'].mean())
```

### Documentation

```python
def clean_customer_data(df):
    """
    Clean customer data by removing duplicates, handling missing values,
    and standardizing formats.
    
    Args:
        df (pd.DataFrame): Raw customer data
        
    Returns:
        pd.DataFrame: Cleaned customer data
        
    Raises:
        ValueError: If required columns are missing
    """
    required_columns = ['customer_id', 'name', 'email']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {required_columns}")
    
    # Data cleaning logic
    return cleaned_df
```

## Exercises

1. **Data Loading**: Load data from multiple sources (CSV, JSON, Excel) and combine them.
2. **Data Cleaning**: Create a function to clean messy data with missing values and duplicates.
3. **Feature Engineering**: Build features from date columns and categorical variables.
4. **Data Aggregation**: Create summary statistics by multiple grouping variables.
5. **Performance Optimization**: Optimize memory usage and processing speed for large datasets.

## Next Steps

After mastering data manipulation, explore:
- [Data Cleaning and Preprocessing](data_cleaning_guide.md)
- [Exploratory Data Analysis](eda_guide.md)
- [Feature Engineering](feature_engineering_guide.md)
- [Working with Pandas](../pandas/pandas_basics_guide.md) 