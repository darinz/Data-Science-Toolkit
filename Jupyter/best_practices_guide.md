# Jupyter Best Practices: Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Notebook Organization](#notebook-organization)
3. [Code Quality](#code-quality)
4. [Documentation](#documentation)
5. [Version Control](#version-control)
6. [Performance Optimization](#performance-optimization)
7. [Collaboration](#collaboration)
8. [Security](#security)
9. [Testing](#testing)
10. [Deployment](#deployment)

## Introduction

Following best practices in Jupyter notebooks ensures maintainable, reproducible, and collaborative data science workflows. This guide covers essential practices for professional notebook development.

### Key Principles
- **Reproducibility**: Notebooks should produce consistent results
- **Maintainability**: Code should be clean, well-documented, and organized
- **Collaboration**: Notebooks should be easy for others to understand and use
- **Performance**: Efficient code execution and resource management
- **Security**: Safe handling of sensitive data and credentials

## Notebook Organization

### Structure and Flow

```python
# Recommended notebook structure:

# 1. Title and Introduction (Markdown)
"""
# Project Title

## Overview
Brief description of the project, objectives, and methodology.

## Author
Your name and contact information.

## Date
Last updated: [Date]

## Dependencies
List of key packages and versions used.
"""

# 2. Setup and Imports (Code)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
%matplotlib inline

# 3. Data Loading (Code + Markdown)
# Load and examine data
df = pd.read_csv('data.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# 4. Data Exploration (Code + Markdown)
# Explore data structure and quality

# 5. Data Preprocessing (Code + Markdown)
# Clean and prepare data

# 6. Analysis (Code + Markdown)
# Perform analysis and create visualizations

# 7. Results and Conclusions (Markdown)
# Summarize findings and next steps
```

### Cell Organization

```python
# Good cell organization practices:

# 1. Keep cells focused on single tasks
# Good: One cell for data loading
df = pd.read_csv('data.csv')
print(f"Loaded {len(df)} rows")

# Good: Separate cell for data inspection
print(df.info())
print(df.describe())

# 2. Use descriptive cell names (in JupyterLab)
# Right-click cell → Add Cell Tag → Add descriptive tag

# 3. Group related operations
# Data loading cells together
# Data cleaning cells together
# Analysis cells together
# Visualization cells together

# 4. Use markdown cells for documentation
# Explain what each code cell does
# Document assumptions and decisions
# Provide context for results
```

### File Naming Conventions

```python
# Recommended naming conventions:

# Descriptive names with dates
# 2024-01-15_data_exploration.ipynb
# 2024-01-15_model_training.ipynb
# 2024-01-15_results_analysis.ipynb

# Use prefixes for different types
# EDA_2024-01-15_customer_data.ipynb
# MODEL_2024-01-15_random_forest.ipynb
# VIZ_2024-01-15_sales_dashboard.ipynb

# Include version numbers
# v1.0_initial_analysis.ipynb
# v1.1_improved_model.ipynb
# v2.0_final_results.ipynb
```

## Code Quality

### Python Style Guidelines

```python
# Follow PEP 8 style guidelines

# 1. Use descriptive variable names
# Good
customer_data = pd.read_csv('customers.csv')
total_sales = customer_data['sales'].sum()

# Bad
df = pd.read_csv('c.csv')
ts = df['s'].sum()

# 2. Use functions for reusable code
def calculate_engagement_score(frequency, duration, weights=(0.6, 0.4)):
    """
    Calculate user engagement score.
    
    Parameters:
    -----------
    frequency : float
        User login frequency
    duration : float
        Average session duration
    weights : tuple, optional
        Weights for frequency and duration
        
    Returns:
    --------
    float
        Engagement score
    """
    return frequency * weights[0] + duration * weights[1]

# 3. Use type hints (Python 3.6+)
from typing import List, Dict, Optional, Union

def process_data(data: List[Dict]) -> Optional[pd.DataFrame]:
    """Process list of dictionaries into DataFrame."""
    if not data:
        return None
    return pd.DataFrame(data)

# 4. Handle errors gracefully
def safe_divide(a: float, b: float) -> Optional[float]:
    """Safely divide two numbers."""
    try:
        return a / b
    except ZeroDivisionError:
        print("Warning: Division by zero")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
```

### Code Organization

```python
# Organize code into logical sections

# 1. Configuration
CONFIG = {
    'random_seed': 42,
    'test_size': 0.2,
    'n_estimators': 100,
    'max_depth': 10
}

# 2. Utility functions
def load_data(filepath: str) -> pd.DataFrame:
    """Load data from file."""
    return pd.read_csv(filepath)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset."""
    # Data cleaning logic here
    return df

def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """Evaluate model performance."""
    predictions = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions, average='weighted'),
        'recall': recall_score(y_test, predictions, average='weighted')
    }

# 3. Main execution
if __name__ == "__main__":
    # Set random seed
    np.random.seed(CONFIG['random_seed'])
    
    # Load and process data
    data = load_data('data.csv')
    processed_data = preprocess_data(data)
    
    # Model training and evaluation
    # ... rest of the code
```

### Error Handling

```python
# Comprehensive error handling

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def robust_data_processing(filepath: str) -> Optional[pd.DataFrame]:
    """Robust data processing with error handling."""
    try:
        # Load data
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
        
        # Validate data
        if df.empty:
            raise ValueError("Dataset is empty")
        
        # Process data
        logger.info("Processing data...")
        df = preprocess_data(df)
        
        logger.info(f"Successfully processed {len(df)} rows")
        return df
        
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return None
    except pd.errors.EmptyDataError:
        logger.error("File is empty or corrupted")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

# Usage
result = robust_data_processing('data.csv')
if result is not None:
    print("Data processing successful")
else:
    print("Data processing failed")
```

## Documentation

### Markdown Documentation

```markdown
# Comprehensive Documentation Example

## Project: Customer Churn Analysis

### Overview
This notebook analyzes customer churn patterns to identify factors that contribute to customer retention.

### Objectives
- Identify key factors influencing customer churn
- Build predictive model for churn risk
- Provide actionable insights for retention strategies

### Methodology
1. **Data Exploration**: Analyze customer demographics and behavior
2. **Feature Engineering**: Create relevant features for modeling
3. **Model Development**: Train and evaluate multiple algorithms
4. **Results Analysis**: Interpret findings and recommendations

### Key Assumptions
- Customer data is representative of the population
- Historical patterns will continue in the future
- Churn is defined as no activity for 90+ days

### Data Sources
- Customer database (last updated: 2024-01-15)
- Transaction history (2020-2024)
- Customer service interactions

### Dependencies
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.1.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
```

### Code Documentation

```python
# Comprehensive function documentation

def analyze_customer_churn(data: pd.DataFrame, 
                          churn_threshold: int = 90,
                          test_size: float = 0.2,
                          random_state: int = 42) -> Dict[str, Any]:
    """
    Analyze customer churn patterns and build predictive model.
    
    This function performs a comprehensive analysis of customer churn,
    including data preprocessing, feature engineering, model training,
    and performance evaluation.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Customer dataset with columns: customer_id, last_activity_date,
        total_purchases, avg_order_value, customer_service_calls, etc.
    churn_threshold : int, default=90
        Number of days without activity to consider customer as churned
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing:
        - 'model': Trained Random Forest model
        - 'feature_importance': DataFrame with feature importance scores
        - 'performance_metrics': Dictionary with accuracy, precision, recall
        - 'churn_rate': Overall churn rate in the dataset
        - 'key_insights': List of main findings
        
    Raises:
    ------
    ValueError
        If data is empty or missing required columns
    KeyError
        If required columns are not found in the dataset
        
    Examples:
    --------
    >>> df = pd.read_csv('customers.csv')
    >>> results = analyze_customer_churn(df, churn_threshold=60)
    >>> print(f"Churn rate: {results['churn_rate']:.2%}")
    >>> print(f"Model accuracy: {results['performance_metrics']['accuracy']:.3f}")
    
    Notes:
    -----
    - The function automatically handles missing values
    - Categorical variables are encoded using one-hot encoding
    - Feature scaling is applied to numerical variables
    - Cross-validation is used for model evaluation
    """
    
    # Implementation here
    pass
```

### Inline Comments

```python
# Good inline commenting practices

# Load customer data from CSV file
df = pd.read_csv('customers.csv')

# Calculate days since last activity (churn indicator)
df['days_since_activity'] = (pd.Timestamp.now() - pd.to_datetime(df['last_activity'])).dt.days

# Create churn target variable (1 if churned, 0 if active)
df['churned'] = (df['days_since_activity'] > 90).astype(int)

# Feature engineering: Create interaction terms
df['purchase_frequency'] = df['total_purchases'] / df['customer_age_days']  # Purchases per day
df['avg_order_size'] = df['total_revenue'] / df['total_purchases']  # Revenue per purchase

# Handle missing values in numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
```

## Version Control

### Git Configuration

```bash
# Configure Git for Jupyter notebooks

# 1. Create .gitattributes file
echo "*.ipynb filter=strip-notebook-output" > .gitattributes

# 2. Configure Git filter
git config --global filter.strip-notebook-output.clean \
    "jupyter nbconvert --clear-output --stdin --stdout --log-level=ERROR"

# 3. Create .gitignore file
cat > .gitignore << EOF
# Jupyter
.ipynb_checkpoints/
*/.ipynb_checkpoints/*

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/

# Data files
*.csv
*.xlsx
*.json
*.parquet
*.h5

# Model files
*.pkl
*.joblib
*.h5

# Logs
*.log
logs/

# Environment variables
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/kynan/nbstripout
    rev: 0.6.1
    hooks:
      - id: nbstripout
        args: [--strip-empty-cells]

  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3
        args: [--line-length=88]

  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
        args: [--profile=black]

  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]
```

### Commit Messages

```bash
# Good commit message format

# Format: type(scope): description
# Types: feat, fix, docs, style, refactor, test, chore

# Examples:
git commit -m "feat(analysis): add customer churn prediction model"
git commit -m "fix(data): handle missing values in customer dataset"
git commit -m "docs(readme): update installation instructions"
git commit -m "style(code): format code with black"
git commit -m "refactor(utils): extract data preprocessing functions"
git commit -m "test(model): add unit tests for model evaluation"
git commit -m "chore(deps): update pandas to version 1.5.0"
```

## Performance Optimization

### Memory Management

```python
# Efficient memory usage

import psutil
import gc

def monitor_memory_usage():
    """Monitor current memory usage."""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.2f} MB")
    return memory_mb

# Monitor memory before and after operations
initial_memory = monitor_memory_usage()

# Load data in chunks for large files
def load_large_csv(filepath: str, chunk_size: int = 10000) -> pd.DataFrame:
    """Load large CSV file in chunks to manage memory."""
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        # Process each chunk
        processed_chunk = preprocess_chunk(chunk)
        chunks.append(processed_chunk)
    
    return pd.concat(chunks, ignore_index=True)

# Use appropriate data types
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by using appropriate data types."""
    for col in df.columns:
        if df[col].dtype == 'object':
            # Convert object columns to category if they have few unique values
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
        elif df[col].dtype == 'int64':
            # Downcast integers
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif df[col].dtype == 'float64':
            # Downcast floats
            df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

# Clear memory when done with large objects
del large_dataframe
gc.collect()

final_memory = monitor_memory_usage()
print(f"Memory saved: {initial_memory - final_memory:.2f} MB")
```

### Code Optimization

```python
# Performance optimization techniques

# 1. Use vectorized operations instead of loops
# Bad: Loop-based operation
def slow_calculation(data):
    result = []
    for value in data:
        result.append(value * 2 + 1)
    return result

# Good: Vectorized operation
def fast_calculation(data):
    return data * 2 + 1

# 2. Use efficient data structures
# Bad: List for large datasets
large_list = list(range(1000000))

# Good: NumPy array for numerical operations
import numpy as np
large_array = np.arange(1000000)

# 3. Profile code to identify bottlenecks
%load_ext line_profiler

def expensive_function():
    # Some expensive computation
    result = 0
    for i in range(1000000):
        result += i ** 2
    return result

%lprun -f expensive_function expensive_function()

# 4. Use parallel processing for independent operations
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def parallel_processing(data_list):
    """Process data in parallel."""
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        results = list(executor.map(process_item, data_list))
    return results
```

### Caching Results

```python
# Cache expensive computations

import joblib
from functools import lru_cache
import hashlib

# 1. Using joblib for disk caching
@joblib.Memory(location='./cache', verbose=0).cache
def expensive_computation(data):
    """Expensive computation that will be cached."""
    # Simulate expensive computation
    import time
    time.sleep(2)
    return data * 2

# 2. Using lru_cache for memory caching
@lru_cache(maxsize=128)
def cached_function(x, y):
    """Function with memory caching."""
    return x ** 2 + y ** 2

# 3. Custom caching with hash-based keys
def get_cache_key(data):
    """Generate cache key from data."""
    return hashlib.md5(str(data).encode()).hexdigest()

cache = {}

def cached_operation(data):
    """Custom cached operation."""
    key = get_cache_key(data)
    if key not in cache:
        cache[key] = expensive_computation(data)
    return cache[key]
```

## Collaboration

### Code Review Guidelines

```python
# Code review checklist

"""
Code Review Checklist for Jupyter Notebooks:

1. Structure and Organization
   □ Clear notebook structure with logical flow
   □ Descriptive cell names and markdown documentation
   □ Proper separation of concerns

2. Code Quality
   □ Follows PEP 8 style guidelines
   □ Uses descriptive variable and function names
   □ Includes proper error handling
   □ No hardcoded values or magic numbers

3. Documentation
   □ Clear project overview and objectives
   □ Well-documented functions with docstrings
   □ Inline comments for complex logic
   □ Dependencies and setup instructions

4. Reproducibility
   □ Random seeds set for reproducibility
   □ Clear data loading and preprocessing steps
   □ Version information for key packages
   □ No absolute file paths

5. Performance
   □ Efficient data structures and operations
   □ Memory usage considerations
   □ Appropriate use of caching

6. Testing
   □ Unit tests for critical functions
   □ Data validation and sanity checks
   □ Error handling for edge cases
"""

# Example of review-ready code
def analyze_customer_data(filepath: str, 
                         output_dir: str = './results',
                         random_seed: int = 42) -> Dict[str, Any]:
    """
    Analyze customer data and generate insights.
    
    This function performs comprehensive customer analysis including
    data loading, preprocessing, analysis, and report generation.
    
    Parameters:
    -----------
    filepath : str
        Path to the customer data CSV file
    output_dir : str, default='./results'
        Directory to save analysis results
    random_seed : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    Dict[str, Any]
        Analysis results including metrics and visualizations
        
    Raises:
    ------
    FileNotFoundError
        If the input file doesn't exist
    ValueError
        If the data is invalid or empty
    """
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Validate inputs
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load and validate data
        df = load_and_validate_data(filepath)
        
        # Perform analysis
        results = perform_analysis(df)
        
        # Generate reports
        generate_reports(results, output_dir)
        
        return results
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise
```

### Team Workflow

```python
# Team collaboration best practices

# 1. Branch naming convention
# feature/add-customer-churn-analysis
# bugfix/fix-data-loading-error
# hotfix/critical-security-patch

# 2. Pull request template
"""
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Refactoring

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No hardcoded values
- [ ] Error handling included
- [ ] Performance considered

## Screenshots (if applicable)
Add screenshots for UI changes.

## Additional Notes
Any additional information or context.
"""

# 3. Code standards enforcement
# .flake8 configuration
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,build,dist

# 4. Automated testing
def test_data_loading():
    """Test data loading functionality."""
    df = load_data('test_data.csv')
    assert not df.empty, "Data should not be empty"
    assert 'customer_id' in df.columns, "Should have customer_id column"

def test_preprocessing():
    """Test data preprocessing."""
    test_data = pd.DataFrame({
        'value': [1, 2, None, 4],
        'category': ['A', 'B', 'A', None]
    })
    processed = preprocess_data(test_data)
    assert processed.isnull().sum().sum() == 0, "No missing values should remain"
```

## Security

### Data Security

```python
# Security best practices

import os
from cryptography.fernet import Fernet
import json

# 1. Environment variables for sensitive data
# Create .env file (not committed to git)
"""
DATABASE_URL=postgresql://user:password@localhost/db
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
"""

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

database_url = os.getenv('DATABASE_URL')
api_key = os.getenv('API_KEY')

# 2. Encrypt sensitive data
def encrypt_sensitive_data(data: str, key: bytes) -> bytes:
    """Encrypt sensitive data."""
    f = Fernet(key)
    return f.encrypt(data.encode())

def decrypt_sensitive_data(encrypted_data: bytes, key: bytes) -> str:
    """Decrypt sensitive data."""
    f = Fernet(key)
    return f.decrypt(encrypted_data).decode()

# 3. Secure data handling
def secure_data_processing(data: pd.DataFrame) -> pd.DataFrame:
    """Process data with security considerations."""
    # Remove sensitive columns
    sensitive_columns = ['ssn', 'credit_card', 'password']
    for col in sensitive_columns:
        if col in data.columns:
            data = data.drop(columns=[col])
    
    # Anonymize identifiers
    if 'customer_id' in data.columns:
        data['customer_id'] = data['customer_id'].apply(hash)
    
    return data

# 4. Input validation
def validate_input_data(data: pd.DataFrame) -> bool:
    """Validate input data for security."""
    # Check for SQL injection patterns
    sql_patterns = ["'", "DROP", "DELETE", "INSERT", "UPDATE"]
    for col in data.select_dtypes(include=['object']).columns:
        for pattern in sql_patterns:
            if data[col].astype(str).str.contains(pattern, case=False).any():
                raise ValueError(f"Potential SQL injection detected in column {col}")
    
    return True
```

### Access Control

```python
# Access control and permissions

import logging
from datetime import datetime

# 1. Logging for audit trails
def setup_audit_logging():
    """Setup audit logging for security events."""
    logging.basicConfig(
        filename='audit.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def log_data_access(user: str, dataset: str, action: str):
    """Log data access for audit purposes."""
    logging.info(f"User {user} {action} dataset {dataset} at {datetime.now()}")

# 2. User authentication
def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user credentials."""
    # In production, use proper authentication system
    valid_users = {
        'admin': 'secure_password_hash',
        'analyst': 'analyst_password_hash'
    }
    
    if username in valid_users:
        # Verify password hash
        return verify_password(password, valid_users[username])
    return False

# 3. Data access control
def check_data_permissions(user: str, dataset: str) -> bool:
    """Check if user has permission to access dataset."""
    permissions = {
        'admin': ['all'],
        'analyst': ['customer_data', 'sales_data'],
        'viewer': ['public_data']
    }
    
    user_perms = permissions.get(user, [])
    return 'all' in user_perms or dataset in user_perms

# 4. Secure data loading
def secure_load_data(filepath: str, user: str) -> Optional[pd.DataFrame]:
    """Securely load data with access control."""
    try:
        # Check permissions
        if not check_data_permissions(user, filepath):
            logging.warning(f"Unauthorized access attempt by {user} to {filepath}")
            return None
        
        # Log access
        log_data_access(user, filepath, "accessed")
        
        # Load data
        df = pd.read_csv(filepath)
        
        # Apply security filters
        df = secure_data_processing(df)
        
        return df
        
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None
```

## Testing

### Unit Testing

```python
# Comprehensive testing for notebooks

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

class TestDataAnalysis(unittest.TestCase):
    """Test cases for data analysis functions."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_data = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'purchase_amount': [100, 200, 150, 300, 250],
            'churned': [0, 1, 0, 1, 0]
        })
    
    def test_data_loading(self):
        """Test data loading functionality."""
        with patch('pandas.read_csv') as mock_read:
            mock_read.return_value = self.sample_data
            result = load_data('test.csv')
            
            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 5)
            mock_read.assert_called_once_with('test.csv')
    
    def test_preprocessing(self):
        """Test data preprocessing."""
        # Test with missing values
        data_with_nulls = self.sample_data.copy()
        data_with_nulls.loc[0, 'purchase_amount'] = np.nan
        
        result = preprocess_data(data_with_nulls)
        
        self.assertFalse(result.isnull().any().any())
        self.assertEqual(len(result), 5)
    
    def test_feature_engineering(self):
        """Test feature engineering."""
        result = create_features(self.sample_data)
        
        expected_features = ['purchase_amount', 'churned', 'high_value_customer']
        for feature in expected_features:
            self.assertIn(feature, result.columns)
    
    def test_model_training(self):
        """Test model training."""
        X = self.sample_data[['purchase_amount']]
        y = self.sample_data['churned']
        
        model = train_model(X, y)
        
        self.assertIsNotNone(model)
        predictions = model.predict(X)
        self.assertEqual(len(predictions), len(y))
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics calculation."""
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1]
        
        metrics = calculate_metrics(y_true, y_pred)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)

# Run tests
if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
# Integration testing for complete workflows

def test_complete_analysis_workflow():
    """Test the complete analysis workflow."""
    
    # Create test data
    test_data = pd.DataFrame({
        'customer_id': range(100),
        'purchase_amount': np.random.normal(100, 30, 100),
        'customer_age': np.random.randint(18, 80, 100),
        'churned': np.random.choice([0, 1], 100, p=[0.7, 0.3])
    })
    
    # Save test data
    test_file = 'test_data.csv'
    test_data.to_csv(test_file, index=False)
    
    try:
        # Run complete analysis
        results = analyze_customer_data(test_file)
        
        # Verify results structure
        assert 'model' in results
        assert 'metrics' in results
        assert 'insights' in results
        
        # Verify model performance
        assert results['metrics']['accuracy'] > 0.5
        assert results['metrics']['accuracy'] <= 1.0
        
        # Verify insights
        assert len(results['insights']) > 0
        
        print("Integration test passed!")
        
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)

# Run integration test
test_complete_analysis_workflow()
```

## Deployment

### Production Deployment

```python
# Production deployment considerations

# 1. Environment configuration
import configparser

def load_config(config_file: str = 'config.ini') -> dict:
    """Load configuration from file."""
    config = configparser.ConfigParser()
    config.read(config_file)
    
    return {
        'database_url': config['database']['url'],
        'api_key': config['api']['key'],
        'log_level': config['logging']['level'],
        'output_dir': config['output']['directory']
    }

# config.ini example:
"""
[database]
url = postgresql://user:password@localhost/prod_db

[api]
key = your_production_api_key

[logging]
level = INFO

[output]
directory = /var/www/results
"""

# 2. Production-ready data processing
class ProductionDataProcessor:
    """Production-ready data processor with error handling and logging."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = self._setup_logging()
    
    def _setup_logging(self):
        """Setup production logging."""
        logging.basicConfig(
            level=getattr(logging, self.config['log_level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('production.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process data with production considerations."""
        try:
            self.logger.info(f"Processing {len(data)} records")
            
            # Validate input
            self._validate_input(data)
            
            # Process data
            processed_data = self._apply_transformations(data)
            
            # Validate output
            self._validate_output(processed_data)
            
            self.logger.info("Data processing completed successfully")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Data processing failed: {e}")
            raise
    
    def _validate_input(self, data: pd.DataFrame):
        """Validate input data."""
        if data.empty:
            raise ValueError("Input data is empty")
        
        required_columns = ['customer_id', 'purchase_amount']
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    def _validate_output(self, data: pd.DataFrame):
        """Validate output data."""
        if data.empty:
            raise ValueError("Output data is empty")
        
        # Check for reasonable value ranges
        if 'purchase_amount' in data.columns:
            if (data['purchase_amount'] < 0).any():
                raise ValueError("Negative purchase amounts detected")

# 3. Scheduled execution
import schedule
import time

def scheduled_analysis():
    """Run analysis on schedule."""
    try:
        config = load_config()
        processor = ProductionDataProcessor(config)
        
        # Load and process data
        data = load_data(config['database_url'])
        results = processor.process_data(data)
        
        # Save results
        save_results(results, config['output_dir'])
        
        print(f"Analysis completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")

# Schedule daily analysis at 2 AM
schedule.every().day.at("02:00").do(scheduled_analysis)

# Run scheduler
while True:
    schedule.run_pending()
    time.sleep(60)
```

### Containerization

```dockerfile
# Dockerfile for Jupyter notebook deployment
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 jupyter
RUN chown -R jupyter:jupyter /app
USER jupyter

# Expose port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8888/api/health || exit 1

# Start Jupyter
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

```yaml
# docker-compose.yml for complete deployment
version: '3.8'

services:
  jupyter:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/app/notebooks
      - ./data:/app/data
      - ./results:/app/results
    environment:
      - JUPYTER_TOKEN=your_secure_token
    depends_on:
      - postgres
    restart: unless-stopped

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: analysis_db
      POSTGRES_USER: analyst
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

## Conclusion

Following these best practices ensures that your Jupyter notebooks are:

- **Reproducible**: Consistent results across different environments
- **Maintainable**: Clean, well-documented, and organized code
- **Collaborative**: Easy for team members to understand and contribute
- **Secure**: Proper handling of sensitive data and access control
- **Performant**: Efficient code execution and resource management
- **Testable**: Comprehensive testing for reliability

### Key Takeaways

1. **Structure matters**: Organize notebooks with clear sections and flow
2. **Document everything**: Use markdown and docstrings extensively
3. **Version control**: Use Git with proper configuration and workflows
4. **Test thoroughly**: Implement unit and integration tests
5. **Optimize performance**: Monitor memory usage and optimize code
6. **Secure data**: Handle sensitive information appropriately
7. **Deploy properly**: Use containers and proper production practices

### Resources

- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Jupyter Best Practices](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.html)
- [Git Best Practices](https://git-scm.com/book/en/v2)
- [Python Testing](https://docs.python.org/3/library/unittest.html)
- [Docker Documentation](https://docs.docker.com/)

---

**Happy Notebooking!** 📓✨ 