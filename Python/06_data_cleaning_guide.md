# Python Data Cleaning Guide

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-blue.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-orange.svg)](https://numpy.org/)

A comprehensive guide to data cleaning and preprocessing techniques in Python for data science and machine learning applications.

## Table of Contents

1. [Introduction to Data Cleaning](#introduction-to-data-cleaning)
2. [Handling Missing Data](#handling-missing-data)
3. [Dealing with Outliers](#dealing-with-outliers)
4. [Data Type Conversion](#data-type-conversion)
5. [String Cleaning](#string-cleaning)
6. [Duplicate Detection and Removal](#duplicate-detection-and-removal)
7. [Data Validation](#data-validation)
8. [Feature Engineering for Cleaning](#feature-engineering-for-cleaning)
9. [Automated Data Cleaning](#automated-data-cleaning)
10. [Best Practices](#best-practices)

## Introduction to Data Cleaning

Data cleaning is the process of detecting and correcting (or removing) corrupt or inaccurate records from a dataset.

### Why Data Cleaning Matters

- **Model Performance**: Clean data leads to better model performance
- **Reliability**: Ensures results are trustworthy and reproducible
- **Efficiency**: Reduces computational time and resources
- **Compliance**: Meets data quality standards and regulations

### Data Quality Issues

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample dataset with common data quality issues
data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry'],
    'age': [25, 30, None, 35, 40, 45, 50, 55, 60, 65],
    'salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000],
    'email': ['john@email.com', 'jane@email.com', 'bob@email.com', 'alice@email.com', 
              'charlie@email.com', 'diana@email.com', 'eve@email.com', 'frank@email.com', 
              'grace@email.com', 'henry@email.com'],
    'department': ['IT', 'HR', 'IT', 'HR', 'IT', 'HR', 'IT', 'HR', 'IT', 'HR'],
    'start_date': ['2020-01-15', '2019-03-20', '2021-07-10', '2018-11-05', '2022-02-28',
                   '2020-09-12', '2019-06-18', '2021-04-25', '2018-08-30', '2022-01-07']
}

df = pd.DataFrame(data)
print("Original Dataset:")
print(df)
print("\nDataset Info:")
print(df.info())
```

## Handling Missing Data

### Detecting Missing Data

```python
def analyze_missing_data(df):
    """Analyze missing data in a DataFrame."""
    
    # Count missing values
    missing_counts = df.isnull().sum()
    missing_percentages = (missing_counts / len(df)) * 100
    
    # Create summary
    missing_summary = pd.DataFrame({
        'Column': missing_counts.index,
        'Missing_Count': missing_counts.values,
        'Missing_Percentage': missing_percentages.values
    })
    
    # Filter columns with missing data
    missing_summary = missing_summary[missing_summary['Missing_Count'] > 0]
    missing_summary = missing_summary.sort_values('Missing_Percentage', ascending=False)
    
    return missing_summary

# Analyze missing data
missing_analysis = analyze_missing_data(df)
print("Missing Data Analysis:")
print(missing_analysis)

# Visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()
```

### Strategies for Handling Missing Data

```python
def handle_missing_data(df, strategy='auto'):
    """
    Handle missing data using various strategies.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    strategy : str
        Strategy to use: 'drop', 'fill_mean', 'fill_median', 'fill_mode', 'interpolate', 'auto'
    """
    
    df_cleaned = df.copy()
    
    if strategy == 'drop':
        # Remove rows with any missing values
        df_cleaned = df_cleaned.dropna()
        
    elif strategy == 'fill_mean':
        # Fill numeric columns with mean
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
            df_cleaned[numeric_columns].mean()
        )
        
    elif strategy == 'fill_median':
        # Fill numeric columns with median
        numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
            df_cleaned[numeric_columns].median()
        )
        
    elif strategy == 'fill_mode':
        # Fill categorical columns with mode
        categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            mode_value = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown'
            df_cleaned[col] = df_cleaned[col].fillna(mode_value)
            
    elif strategy == 'interpolate':
        # Use interpolation for time series data
        df_cleaned = df_cleaned.interpolate(method='linear')
        
    elif strategy == 'auto':
        # Automatic strategy based on data type and missing percentage
        for column in df_cleaned.columns:
            missing_pct = df_cleaned[column].isnull().sum() / len(df_cleaned)
            
            if missing_pct > 0.5:
                # If more than 50% missing, drop the column
                df_cleaned = df_cleaned.drop(columns=[column])
            elif df_cleaned[column].dtype in ['int64', 'float64']:
                # Numeric columns: fill with median
                df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].median())
            else:
                # Categorical columns: fill with mode
                mode_value = df_cleaned[column].mode()[0] if not df_cleaned[column].mode().empty else 'Unknown'
                df_cleaned[column] = df_cleaned[column].fillna(mode_value)
    
    return df_cleaned

# Test different strategies
strategies = ['drop', 'fill_mean', 'fill_median', 'fill_mode', 'auto']

for strategy in strategies:
    cleaned_df = handle_missing_data(df, strategy)
    print(f"\nStrategy: {strategy}")
    print(f"Rows after cleaning: {len(cleaned_df)}")
    print(f"Missing values remaining: {cleaned_df.isnull().sum().sum()}")
```

### Advanced Missing Data Techniques

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor

def advanced_missing_data_imputation(df, method='knn'):
    """
    Advanced missing data imputation techniques.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    method : str
        Method to use: 'knn', 'random_forest', 'multiple_imputation'
    """
    
    df_cleaned = df.copy()
    numeric_columns = df_cleaned.select_dtypes(include=[np.number]).columns
    
    if method == 'knn':
        # K-Nearest Neighbors imputation
        imputer = KNNImputer(n_neighbors=5)
        df_cleaned[numeric_columns] = imputer.fit_transform(df_cleaned[numeric_columns])
        
    elif method == 'random_forest':
        # Random Forest imputation
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        df_temp = imputer.fit_transform(df_cleaned[numeric_columns])
        
        # Use Random Forest to predict missing values
        for col in numeric_columns:
            missing_mask = df_cleaned[col].isnull()
            if missing_mask.sum() > 0:
                # Features for prediction (other numeric columns)
                feature_cols = [c for c in numeric_columns if c != col]
                
                # Train model on non-missing data
                train_data = df_temp[~missing_mask]
                train_features = train_data[:, [numeric_columns.get_loc(c) for c in feature_cols]]
                train_target = train_data[:, numeric_columns.get_loc(col)]
                
                if len(train_data) > 0:
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(train_features, train_target)
                    
                    # Predict missing values
                    missing_features = df_temp[missing_mask][:, [numeric_columns.get_loc(c) for c in feature_cols]]
                    predictions = rf_model.predict(missing_features)
                    
                    # Fill missing values
                    df_cleaned.loc[missing_mask, col] = predictions
    
    return df_cleaned

# Test advanced imputation
advanced_cleaned = advanced_missing_data_imputation(df, 'knn')
print("Advanced KNN Imputation Results:")
print(advanced_cleaned)
```

## Dealing with Outliers

### Outlier Detection Methods

```python
def detect_outliers(df, method='iqr', columns=None):
    """
    Detect outliers using various methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    method : str
        Method to use: 'iqr', 'zscore', 'isolation_forest', 'local_outlier_factor'
    columns : list
        Columns to check for outliers (default: all numeric columns)
    """
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outliers_summary = {}
    
    for column in columns:
        data = df[column].dropna()
        
        if method == 'iqr':
            # Interquartile Range method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs((data - data.mean()) / data.std())
            outliers = data[z_scores > 3]
            
        elif method == 'isolation_forest':
            # Isolation Forest method
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(data.values.reshape(-1, 1))
            outliers = data[outlier_labels == -1]
            
        elif method == 'local_outlier_factor':
            # Local Outlier Factor method
            from sklearn.neighbors import LocalOutlierFactor
            
            lof = LocalOutlierFactor(contamination=0.1)
            outlier_labels = lof.fit_predict(data.values.reshape(-1, 1))
            outliers = data[outlier_labels == -1]
        
        outliers_summary[column] = {
            'outlier_count': len(outliers),
            'outlier_percentage': (len(outliers) / len(data)) * 100,
            'outlier_indices': outliers.index.tolist(),
            'outlier_values': outliers.values.tolist()
        }
    
    return outliers_summary

# Detect outliers
outliers = detect_outliers(df, method='iqr')
print("Outlier Detection Results:")
for col, info in outliers.items():
    print(f"\n{col}:")
    print(f"  Outlier count: {info['outlier_count']}")
    print(f"  Outlier percentage: {info['outlier_percentage']:.2f}%")
```

### Outlier Treatment Strategies

```python
def treat_outliers(df, method='cap', outlier_info=None):
    """
    Treat outliers using various strategies.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    method : str
        Method to use: 'remove', 'cap', 'transform', 'winsorize'
    outlier_info : dict
        Outlier information from detect_outliers function
    """
    
    df_cleaned = df.copy()
    
    if outlier_info is None:
        outlier_info = detect_outliers(df)
    
    for column, info in outlier_info.items():
        if info['outlier_count'] == 0:
            continue
            
        outlier_indices = info['outlier_indices']
        data = df_cleaned[column].dropna()
        
        if method == 'remove':
            # Remove outliers
            df_cleaned = df_cleaned.drop(index=outlier_indices)
            
        elif method == 'cap':
            # Cap outliers at percentiles
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_cleaned.loc[df_cleaned[column] < lower_bound, column] = lower_bound
            df_cleaned.loc[df_cleaned[column] > upper_bound, column] = upper_bound
            
        elif method == 'transform':
            # Apply log transformation
            if (df_cleaned[column] > 0).all():
                df_cleaned[column] = np.log1p(df_cleaned[column])
            else:
                # Use square root for negative values
                df_cleaned[column] = np.sqrt(np.abs(df_cleaned[column]))
                
        elif method == 'winsorize':
            # Winsorize outliers
            from scipy.stats.mstats import winsorize
            df_cleaned[column] = winsorize(df_cleaned[column], limits=[0.05, 0.05])
    
    return df_cleaned

# Test outlier treatment
cleaned_df = treat_outliers(df, method='cap')
print("After outlier treatment:")
print(cleaned_df.describe())
```

## Data Type Conversion

### Automatic Data Type Detection

```python
def detect_and_convert_dtypes(df):
    """
    Detect and convert data types automatically.
    """
    
    df_cleaned = df.copy()
    
    # Convert date columns
    date_columns = []
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            # Try to convert to datetime
            try:
                pd.to_datetime(df_cleaned[col], errors='raise')
                date_columns.append(col)
            except:
                pass
    
    for col in date_columns:
        df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
    
    # Convert numeric columns
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            # Try to convert to numeric
            try:
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='raise')
            except:
                pass
    
    # Convert boolean columns
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            unique_values = df_cleaned[col].dropna().unique()
            if len(unique_values) == 2 and set(unique_values).issubset({'True', 'False', 'true', 'false', 1, 0}):
                df_cleaned[col] = df_cleaned[col].map({'True': True, 'False': False, 'true': True, 'false': False, 1: True, 0: False})
    
    return df_cleaned

# Convert data types
df_converted = detect_and_convert_dtypes(df)
print("Data types after conversion:")
print(df_converted.dtypes)
```

### Custom Data Type Conversion

```python
def convert_salary_to_numeric(df, salary_column='salary'):
    """
    Convert salary column to numeric, handling various formats.
    """
    
    df_cleaned = df.copy()
    
    def clean_salary(salary):
        if pd.isna(salary):
            return np.nan
        
        # Convert to string
        salary_str = str(salary)
        
        # Remove currency symbols and commas
        salary_str = salary_str.replace('$', '').replace(',', '').replace(' ', '')
        
        # Handle 'K' (thousands) and 'M' (millions)
        if 'K' in salary_str.upper():
            salary_str = salary_str.upper().replace('K', '')
            return float(salary_str) * 1000
        elif 'M' in salary_str.upper():
            salary_str = salary_str.upper().replace('M', '')
            return float(salary_str) * 1000000
        
        # Try to convert to float
        try:
            return float(salary_str)
        except:
            return np.nan
    
    df_cleaned[salary_column] = df_cleaned[salary_column].apply(clean_salary)
    
    return df_cleaned

# Test salary conversion
salary_data = {
    'name': ['John', 'Jane', 'Bob'],
    'salary': ['$50,000', '60K', '75M', '80000']
}
salary_df = pd.DataFrame(salary_data)
cleaned_salary_df = convert_salary_to_numeric(salary_df)
print("Salary conversion results:")
print(cleaned_salary_df)
```

## String Cleaning

### Text Cleaning Functions

```python
import re
import unicodedata

def clean_text(text):
    """
    Clean text data by removing special characters, normalizing whitespace, etc.
    """
    
    if pd.isna(text):
        return text
    
    # Convert to string
    text = str(text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters (keep alphanumeric and basic punctuation)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text)
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def clean_email(email):
    """
    Clean and validate email addresses.
    """
    
    if pd.isna(email):
        return email
    
    email = str(email).strip().lower()
    
    # Basic email validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if re.match(email_pattern, email):
        return email
    else:
        return np.nan

def clean_phone_number(phone):
    """
    Clean and standardize phone numbers.
    """
    
    if pd.isna(phone):
        return phone
    
    phone = str(phone)
    
    # Remove all non-digit characters
    digits = re.sub(r'\D', '', phone)
    
    # Handle different formats
    if len(digits) == 10:
        return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
    elif len(digits) == 11 and digits[0] == '1':
        return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
    else:
        return np.nan

# Apply text cleaning
df_cleaned = df.copy()
df_cleaned['name'] = df_cleaned['name'].apply(clean_text)
df_cleaned['email'] = df_cleaned['email'].apply(clean_email)

print("Text cleaning results:")
print(df_cleaned[['name', 'email']])
```

### Advanced String Processing

```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
except:
    pass

def advanced_text_cleaning(text):
    """
    Advanced text cleaning with NLP techniques.
    """
    
    if pd.isna(text):
        return text
    
    text = str(text)
    
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Remove short tokens
    tokens = [token for token in tokens if len(token) > 2]
    
    # Lemmatization
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Example with text data
text_data = {
    'id': [1, 2, 3],
    'description': [
        'This is a very good product that I really like!',
        'The customer service was terrible and unhelpful.',
        'Amazing experience with fast delivery and great quality.'
    ]
}

text_df = pd.DataFrame(text_data)
text_df['cleaned_description'] = text_df['description'].apply(advanced_text_cleaning)

print("Advanced text cleaning results:")
print(text_df)
```

## Duplicate Detection and Removal

### Duplicate Detection Methods

```python
def detect_duplicates(df, subset=None, method='exact'):
    """
    Detect duplicates using various methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    subset : list
        Columns to consider for duplicate detection
    method : str
        Method to use: 'exact', 'fuzzy', 'similarity'
    """
    
    if method == 'exact':
        # Exact duplicate detection
        duplicates = df.duplicated(subset=subset, keep=False)
        duplicate_rows = df[duplicates]
        
    elif method == 'fuzzy':
        # Fuzzy duplicate detection for string columns
        from fuzzywuzzy import fuzz
        
        string_columns = df.select_dtypes(include=['object']).columns
        if subset:
            string_columns = [col for col in subset if col in string_columns]
        
        duplicate_indices = set()
        
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                similarity_scores = []
                for col in string_columns:
                    if pd.notna(df.iloc[i][col]) and pd.notna(df.iloc[j][col]):
                        score = fuzz.ratio(str(df.iloc[i][col]), str(df.iloc[j][col]))
                        similarity_scores.append(score)
                
                if similarity_scores and np.mean(similarity_scores) > 80:
                    duplicate_indices.add(i)
                    duplicate_indices.add(j)
        
        duplicate_rows = df.iloc[list(duplicate_indices)]
        
    elif method == 'similarity':
        # Similarity-based duplicate detection
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Combine text columns
        text_columns = df.select_dtypes(include=['object']).columns
        if subset:
            text_columns = [col for col in subset if col in text_columns]
        
        if len(text_columns) > 0:
            # Create text representation
            text_data = df[text_columns].fillna('').astype(str).agg(' '.join, axis=1)
            
            # Vectorize text
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            text_vectors = vectorizer.fit_transform(text_data)
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(text_vectors)
            
            # Find similar pairs
            duplicate_indices = set()
            for i in range(len(similarity_matrix)):
                for j in range(i + 1, len(similarity_matrix)):
                    if similarity_matrix[i, j] > 0.8:
                        duplicate_indices.add(i)
                        duplicate_indices.add(j)
            
            duplicate_rows = df.iloc[list(duplicate_indices)]
        else:
            duplicate_rows = pd.DataFrame()
    
    return duplicate_rows

# Detect duplicates
duplicates = detect_duplicates(df, method='exact')
print(f"Found {len(duplicates)} duplicate rows")
print(duplicates)
```

### Duplicate Removal Strategies

```python
def remove_duplicates(df, subset=None, keep='first', method='exact'):
    """
    Remove duplicates using various strategies.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    subset : list
        Columns to consider for duplicate detection
    keep : str
        Which duplicates to keep: 'first', 'last', False
    method : str
        Method to use: 'exact', 'fuzzy', 'similarity'
    """
    
    df_cleaned = df.copy()
    
    if method == 'exact':
        # Remove exact duplicates
        df_cleaned = df_cleaned.drop_duplicates(subset=subset, keep=keep)
        
    elif method == 'fuzzy':
        # Remove fuzzy duplicates
        duplicates = detect_duplicates(df_cleaned, subset, method='fuzzy')
        df_cleaned = df_cleaned.drop(duplicates.index)
        
    elif method == 'similarity':
        # Remove similarity-based duplicates
        duplicates = detect_duplicates(df_cleaned, subset, method='similarity')
        df_cleaned = df_cleaned.drop(duplicates.index)
    
    return df_cleaned

# Remove duplicates
df_no_duplicates = remove_duplicates(df, method='exact')
print(f"Original rows: {len(df)}")
print(f"After removing duplicates: {len(df_no_duplicates)}")
```

## Data Validation

### Schema Validation

```python
from pydantic import BaseModel, validator
from typing import Optional, List
from datetime import datetime

class EmployeeSchema(BaseModel):
    """Schema for employee data validation."""
    
    id: int
    name: str
    age: Optional[int]
    salary: float
    email: str
    department: str
    start_date: datetime
    
    @validator('age')
    def validate_age(cls, v):
        if v is not None and (v < 18 or v > 100):
            raise ValueError('Age must be between 18 and 100')
        return v
    
    @validator('salary')
    def validate_salary(cls, v):
        if v < 0:
            raise ValueError('Salary cannot be negative')
        return v
    
    @validator('email')
    def validate_email(cls, v):
        import re
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, v):
            raise ValueError('Invalid email format')
        return v

def validate_dataframe(df, schema_class):
    """
    Validate DataFrame against a Pydantic schema.
    """
    
    validation_errors = []
    valid_rows = []
    
    for index, row in df.iterrows():
        try:
            # Convert row to dict and validate
            row_dict = row.to_dict()
            validated_data = schema_class(**row_dict)
            valid_rows.append(validated_data.dict())
        except Exception as e:
            validation_errors.append({
                'row_index': index,
                'error': str(e),
                'data': row.to_dict()
            })
    
    return valid_rows, validation_errors

# Validate data
valid_rows, errors = validate_dataframe(df, EmployeeSchema)
print(f"Valid rows: {len(valid_rows)}")
print(f"Validation errors: {len(errors)}")

if errors:
    print("\nValidation errors:")
    for error in errors[:3]:  # Show first 3 errors
        print(f"Row {error['row_index']}: {error['error']}")
```

### Business Rule Validation

```python
def validate_business_rules(df):
    """
    Validate business rules for the dataset.
    """
    
    violations = []
    
    # Rule 1: Salary should be positive
    negative_salary = df[df['salary'] < 0]
    if len(negative_salary) > 0:
        violations.append({
            'rule': 'Salary must be positive',
            'violations': len(negative_salary),
            'rows': negative_salary.index.tolist()
        })
    
    # Rule 2: Age should be reasonable for employment
    invalid_age = df[(df['age'] < 16) | (df['age'] > 80)]
    if len(invalid_age) > 0:
        violations.append({
            'rule': 'Age must be between 16 and 80',
            'violations': len(invalid_age),
            'rows': invalid_age.index.tolist()
        })
    
    # Rule 3: Start date should not be in the future
    future_dates = df[df['start_date'] > pd.Timestamp.now()]
    if len(future_dates) > 0:
        violations.append({
            'rule': 'Start date cannot be in the future',
            'violations': len(future_dates),
            'rows': future_dates.index.tolist()
        })
    
    # Rule 4: Email should be unique
    duplicate_emails = df[df.duplicated(subset=['email'], keep=False)]
    if len(duplicate_emails) > 0:
        violations.append({
            'rule': 'Email addresses must be unique',
            'violations': len(duplicate_emails),
            'rows': duplicate_emails.index.tolist()
        })
    
    return violations

# Validate business rules
violations = validate_business_rules(df)
print("Business rule violations:")
for violation in violations:
    print(f"\n{violation['rule']}: {violation['violations']} violations")
```

## Feature Engineering for Cleaning

### Creating Cleaning Features

```python
def create_cleaning_features(df):
    """
    Create features that help identify data quality issues.
    """
    
    df_with_features = df.copy()
    
    # Length features for string columns
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        df_with_features[f'{col}_length'] = df[col].astype(str).str.len()
        df_with_features[f'{col}_word_count'] = df[col].astype(str).str.split().str.len()
    
    # Missing value indicators
    for col in df.columns:
        df_with_features[f'{col}_is_missing'] = df[col].isnull().astype(int)
    
    # Outlier indicators
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df_with_features[f'{col}_is_outlier'] = (
            (df[col] < lower_bound) | (df[col] > upper_bound)
        ).astype(int)
    
    # Data quality score
    df_with_features['data_quality_score'] = (
        1 - df_with_features[[col for col in df_with_features.columns if col.endswith('_is_missing')]].mean(axis=1)
    )
    
    return df_with_features

# Create cleaning features
df_with_features = create_cleaning_features(df)
print("Features created for data cleaning:")
print(df_with_features.columns.tolist())
```

## Automated Data Cleaning

### Complete Data Cleaning Pipeline

```python
class DataCleaner:
    """
    Automated data cleaning pipeline.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.cleaning_history = []
        
    def clean_dataset(self, df):
        """
        Apply complete data cleaning pipeline.
        """
        
        df_cleaned = df.copy()
        
        # Step 1: Handle missing data
        if self.config.get('handle_missing', True):
            df_cleaned = self._handle_missing_data(df_cleaned)
            self.cleaning_history.append('missing_data_handled')
        
        # Step 2: Remove duplicates
        if self.config.get('remove_duplicates', True):
            df_cleaned = self._remove_duplicates(df_cleaned)
            self.cleaning_history.append('duplicates_removed')
        
        # Step 3: Handle outliers
        if self.config.get('handle_outliers', True):
            df_cleaned = self._handle_outliers(df_cleaned)
            self.cleaning_history.append('outliers_handled')
        
        # Step 4: Convert data types
        if self.config.get('convert_dtypes', True):
            df_cleaned = self._convert_dtypes(df_cleaned)
            self.cleaning_history.append('dtypes_converted')
        
        # Step 5: Clean strings
        if self.config.get('clean_strings', True):
            df_cleaned = self._clean_strings(df_cleaned)
            self.cleaning_history.append('strings_cleaned')
        
        # Step 6: Validate data
        if self.config.get('validate_data', True):
            validation_results = self._validate_data(df_cleaned)
            self.cleaning_history.append(f'data_validated: {validation_results}')
        
        return df_cleaned
    
    def _handle_missing_data(self, df):
        """Handle missing data."""
        return handle_missing_data(df, strategy='auto')
    
    def _remove_duplicates(self, df):
        """Remove duplicates."""
        return remove_duplicates(df, method='exact')
    
    def _handle_outliers(self, df):
        """Handle outliers."""
        return treat_outliers(df, method='cap')
    
    def _convert_dtypes(self, df):
        """Convert data types."""
        return detect_and_convert_dtypes(df)
    
    def _clean_strings(self, df):
        """Clean string data."""
        df_cleaned = df.copy()
        string_columns = df.select_dtypes(include=['object']).columns
        
        for col in string_columns:
            if 'email' in col.lower():
                df_cleaned[col] = df_cleaned[col].apply(clean_email)
            elif 'phone' in col.lower():
                df_cleaned[col] = df_cleaned[col].apply(clean_phone_number)
            else:
                df_cleaned[col] = df_cleaned[col].apply(clean_text)
        
        return df_cleaned
    
    def _validate_data(self, df):
        """Validate data quality."""
        violations = validate_business_rules(df)
        return len(violations)
    
    def get_cleaning_report(self):
        """Generate cleaning report."""
        return {
            'cleaning_steps': self.cleaning_history,
            'total_steps': len(self.cleaning_history)
        }

# Use the automated cleaner
config = {
    'handle_missing': True,
    'remove_duplicates': True,
    'handle_outliers': True,
    'convert_dtypes': True,
    'clean_strings': True,
    'validate_data': True
}

cleaner = DataCleaner(config)
cleaned_df = cleaner.clean_dataset(df)

print("Automated cleaning completed!")
print("Cleaning report:")
print(cleaner.get_cleaning_report())
```

## Best Practices

### 1. Always Backup Original Data

```python
def safe_data_cleaning(df, backup_path=None):
    """
    Safe data cleaning with backup.
    """
    
    # Create backup
    if backup_path:
        df.to_csv(backup_path, index=False)
    
    # Create copy for cleaning
    df_cleaned = df.copy()
    
    # Apply cleaning steps
    # ... cleaning logic here
    
    return df_cleaned
```

### 2. Document Cleaning Decisions

```python
def document_cleaning_decisions(df_original, df_cleaned, decisions_file):
    """
    Document all cleaning decisions made.
    """
    
    decisions = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'original_shape': df_original.shape,
        'cleaned_shape': df_cleaned.shape,
        'rows_removed': len(df_original) - len(df_cleaned),
        'columns_removed': len(df_original.columns) - len(df_cleaned.columns),
        'missing_data_handled': {
            'original_missing': df_original.isnull().sum().sum(),
            'final_missing': df_cleaned.isnull().sum().sum()
        },
        'outliers_handled': 'Applied IQR method with capping',
        'duplicates_removed': len(df_original) - len(df_original.drop_duplicates()),
        'data_type_changes': {
            'original_dtypes': df_original.dtypes.to_dict(),
            'final_dtypes': df_cleaned.dtypes.to_dict()
        }
    }
    
    # Save decisions
    import json
    with open(decisions_file, 'w') as f:
        json.dump(decisions, f, indent=2, default=str)
    
    return decisions
```

### 3. Validate Cleaning Results

```python
def validate_cleaning_results(df_original, df_cleaned):
    """
    Validate that cleaning improved data quality.
    """
    
    validation_results = {
        'missing_data_reduced': df_cleaned.isnull().sum().sum() < df_original.isnull().sum().sum(),
        'duplicates_removed': len(df_cleaned) < len(df_original),
        'data_types_improved': len(df_cleaned.select_dtypes(include=[np.number]).columns) >= 
                              len(df_original.select_dtypes(include=[np.number]).columns),
        'no_data_loss': len(df_cleaned) > 0,
        'columns_preserved': len(df_cleaned.columns) > 0
    }
    
    return validation_results
```

## Summary

Data cleaning is a crucial step in the data science pipeline:

- **Missing Data**: Use appropriate imputation strategies based on data type and missing percentage
- **Outliers**: Detect and treat outliers using statistical methods
- **Data Types**: Convert data to appropriate types for analysis
- **String Cleaning**: Normalize and validate text data
- **Duplicates**: Remove exact and fuzzy duplicates
- **Validation**: Ensure data meets business rules and quality standards
- **Automation**: Use pipelines for consistent and reproducible cleaning

Mastering data cleaning techniques will significantly improve the quality of your data science projects.

## Next Steps

- Practice with real-world datasets that have various data quality issues
- Explore advanced cleaning libraries like `great_expectations` and `pandas-profiling`
- Learn about data quality monitoring and automated cleaning pipelines
- Study domain-specific cleaning requirements for different industries

---

**Happy Data Cleaning!** 🧹✨ 