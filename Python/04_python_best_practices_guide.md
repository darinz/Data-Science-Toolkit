# Python Best Practices Guide

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PEP8](https://img.shields.io/badge/PEP8-Style%20Guide-green.svg)](https://www.python.org/dev/peps/pep-0008/)
[![Best Practices](https://img.shields.io/badge/Best-Practices-orange.svg)](https://docs.python.org/3/tutorial/index.html)

A comprehensive guide to Python best practices for data science and machine learning applications.

## Table of Contents

1. [Code Style and PEP 8](#code-style-and-pep-8)
2. [Naming Conventions](#naming-conventions)
3. [Documentation](#documentation)
4. [Error Handling](#error-handling)
5. [Performance Optimization](#performance-optimization)
6. [Testing](#testing)
7. [Code Organization](#code-organization)
8. [Security Best Practices](#security-best-practices)
9. [Data Science Specific Practices](#data-science-specific-practices)
10. [Tools and Utilities](#tools-and-utilities)

## Code Style and PEP 8

### Basic PEP 8 Guidelines

```python
# Good - proper spacing and naming
def calculate_mean(data):
    """Calculate the mean of a list of numbers."""
    if not data:
        return 0
    return sum(data) / len(data)

# Bad - poor spacing and naming
def calc_mean(d):
    if not d:return 0
    return sum(d)/len(d)

# Good - proper line length and indentation
def process_large_dataset(
    data,
    threshold=0.5,
    include_zeros=True,
    normalize=False
):
    """Process a large dataset with various options."""
    result = []
    for item in data:
        if item > threshold or (include_zeros and item == 0):
            processed_item = item
            if normalize:
                processed_item = item / max(data)
            result.append(processed_item)
    return result

# Bad - long lines and poor formatting
def process_large_dataset(data,threshold=0.5,include_zeros=True,normalize=False):
    result=[]
    for item in data:
        if item>threshold or (include_zeros and item==0):
            processed_item=item
            if normalize:processed_item=item/max(data)
            result.append(processed_item)
    return result
```

### Import Organization

```python
# Good - organized imports
# Standard library imports
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Local imports
from .utils import helper_function
from .models import MyModel

# Bad - disorganized imports
import numpy as np
import os
from .models import MyModel
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path
```

### Line Length and Breaking

```python
# Good - proper line breaking
def create_complex_visualization(
    data,
    title="My Plot",
    x_label="X Axis",
    y_label="Y Axis",
    figsize=(10, 6),
    style="default"
):
    """Create a complex visualization with many parameters."""
    plt.figure(figsize=figsize)
    plt.plot(data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.style.use(style)
    plt.show()

# Good - breaking long expressions
long_calculation = (
    (value1 * coefficient1) +
    (value2 * coefficient2) +
    (value3 * coefficient3)
)

# Good - breaking long strings
long_message = (
    "This is a very long message that needs to be "
    "broken across multiple lines for readability "
    "and to comply with PEP 8 guidelines."
)
```

## Naming Conventions

### Variable and Function Names

```python
# Good - descriptive names
def calculate_standard_deviation(data_points):
    """Calculate standard deviation of data points."""
    mean_value = sum(data_points) / len(data_points)
    squared_differences = [(x - mean_value) ** 2 for x in data_points]
    variance = sum(squared_differences) / len(data_points)
    return variance ** 0.5

# Bad - unclear names
def calc_std(d):
    m = sum(d) / len(d)
    sd = [(x - m) ** 2 for x in d]
    v = sum(sd) / len(d)
    return v ** 0.5

# Good - boolean variables
is_valid = True
has_data = len(data) > 0
can_process = all(isinstance(x, (int, float)) for x in data)

# Good - constants
MAX_ITERATIONS = 1000
DEFAULT_THRESHOLD = 0.5
CONFIG_FILE_PATH = "config.yaml"
```

### Class Names

```python
# Good - PascalCase for classes
class DataProcessor:
    """Process and transform data."""
    pass

class MachineLearningModel:
    """Base class for ML models."""
    pass

class NeuralNetwork:
    """Neural network implementation."""
    pass

# Bad - lowercase or unclear names
class dataprocessor:
    pass

class mlmodel:
    pass

class nn:
    pass
```

### Module and Package Names

```python
# Good - lowercase with underscores
data_processor.py
machine_learning_utils.py
visualization_tools.py

# Good - package structure
my_package/
    __init__.py
    data_processing/
        __init__.py
        preprocessor.py
        validator.py
    models/
        __init__.py
        linear_model.py
        neural_network.py
    utils/
        __init__.py
        helpers.py
```

## Documentation

### Docstrings

```python
def calculate_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient between two arrays.
    
    Parameters
    ----------
    x : array-like
        First array of numerical values
    y : array-like
        Second array of numerical values
        
    Returns
    -------
    float
        Pearson correlation coefficient between -1 and 1
        
    Raises
    ------
    ValueError
        If arrays have different lengths or contain non-numeric values
        
    Examples
    --------
    >>> x = [1, 2, 3, 4, 5]
    >>> y = [2, 4, 5, 4, 5]
    >>> calculate_correlation(x, y)
    0.8
    """
    if len(x) != len(y):
        raise ValueError("Arrays must have the same length")
    
    if not all(isinstance(val, (int, float)) for val in x + y):
        raise ValueError("All values must be numeric")
    
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    denominator_x = sum((xi - mean_x) ** 2 for xi in x)
    denominator_y = sum((yi - mean_y) ** 2 for yi in y)
    
    if denominator_x == 0 or denominator_y == 0:
        return 0
    
    return numerator / (denominator_x * denominator_y) ** 0.5
```

### Class Documentation

```python
class DataProcessor:
    """
    A class for processing and transforming data.
    
    This class provides methods for cleaning, normalizing, and transforming
    data for machine learning applications.
    
    Attributes
    ----------
    data : array-like
        The input data to be processed
    is_processed : bool
        Whether the data has been processed
    processing_history : list
        List of processing steps applied
        
    Methods
    -------
    clean_data()
        Remove missing values and outliers
    normalize_data()
        Normalize data to [0, 1] range
    transform_data()
        Apply various transformations
    """
    
    def __init__(self, data):
        """
        Initialize the DataProcessor.
        
        Parameters
        ----------
        data : array-like
            Input data to be processed
        """
        self.data = data
        self.is_processed = False
        self.processing_history = []
    
    def clean_data(self):
        """
        Clean the data by removing missing values and outliers.
        
        Returns
        -------
        self
            Returns self for method chaining
        """
        # Implementation here
        self.processing_history.append("cleaned")
        return self
```

### Inline Comments

```python
# Good - explain why, not what
def process_data(data):
    # Skip processing if data is empty to avoid division by zero
    if not data:
        return []
    
    # Normalize to [0, 1] range for better model performance
    min_val = min(data)
    max_val = max(data)
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    
    return normalized_data

# Bad - obvious comments
def process_data(data):
    # Check if data is empty
    if not data:
        return []
    
    # Find minimum value
    min_val = min(data)
    # Find maximum value
    max_val = max(data)
    # Normalize data
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    
    return normalized_data
```

## Error Handling

### Proper Exception Handling

```python
def load_data(file_path):
    """
    Load data from a file with proper error handling.
    
    Parameters
    ----------
    file_path : str
        Path to the data file
        
    Returns
    -------
    pandas.DataFrame
        Loaded data
        
    Raises
    ------
    FileNotFoundError
        If the file doesn't exist
    ValueError
        If the file format is not supported
    """
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"File is empty: {file_path}")
        raise ValueError(f"File is empty: {file_path}")
    except Exception as e:
        logger.error(f"Unexpected error loading {file_path}: {e}")
        raise

# Good - specific exception handling
def divide_numbers(a, b):
    """Divide two numbers with proper error handling."""
    try:
        return a / b
    except ZeroDivisionError:
        raise ValueError("Cannot divide by zero")
    except TypeError:
        raise TypeError("Both arguments must be numbers")

# Good - context managers
def process_file(file_path):
    """Process a file using context manager."""
    try:
        with open(file_path, 'r') as file:
            data = file.read()
            return process_data(data)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except PermissionError:
        logger.error(f"Permission denied: {file_path}")
        raise
```

### Custom Exceptions

```python
class DataProcessingError(Exception):
    """Base exception for data processing errors."""
    pass

class InvalidDataError(DataProcessingError):
    """Raised when data is invalid or malformed."""
    pass

class ProcessingTimeoutError(DataProcessingError):
    """Raised when processing takes too long."""
    pass

def process_large_dataset(data, timeout=60):
    """Process large dataset with custom exceptions."""
    if not isinstance(data, (list, np.ndarray)):
        raise InvalidDataError("Data must be a list or numpy array")
    
    if len(data) == 0:
        raise InvalidDataError("Data cannot be empty")
    
    try:
        # Simulate processing with timeout
        import time
        start_time = time.time()
        
        # Processing logic here
        result = []
        for item in data:
            if time.time() - start_time > timeout:
                raise ProcessingTimeoutError(f"Processing timed out after {timeout} seconds")
            result.append(process_item(item))
        
        return result
    
    except ProcessingTimeoutError:
        logger.error("Processing timed out")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}")
        raise DataProcessingError(f"Processing failed: {e}")
```

## Performance Optimization

### Efficient Data Structures

```python
# Good - use appropriate data structures
from collections import defaultdict, Counter

# For counting occurrences
def count_elements_efficient(data):
    """Count elements efficiently using Counter."""
    return Counter(data)

# For grouping data
def group_data_efficient(data, key_func):
    """Group data efficiently using defaultdict."""
    grouped = defaultdict(list)
    for item in data:
        grouped[key_func(item)].append(item)
    return dict(grouped)

# Bad - inefficient counting
def count_elements_inefficient(data):
    """Count elements inefficiently."""
    counts = {}
    for item in data:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1
    return counts
```

### List Comprehensions vs Loops

```python
# Good - list comprehension for simple operations
def square_numbers_comprehension(numbers):
    """Square numbers using list comprehension."""
    return [x ** 2 for x in numbers]

# Good - generator expression for large datasets
def square_numbers_generator(numbers):
    """Square numbers using generator expression."""
    return (x ** 2 for x in numbers)

# Bad - explicit loop for simple operations
def square_numbers_loop(numbers):
    """Square numbers using explicit loop."""
    result = []
    for x in numbers:
        result.append(x ** 2)
    return result

# Good - use appropriate method
def filter_and_transform(data, condition, transform):
    """Filter and transform data efficiently."""
    return [transform(x) for x in data if condition(x)]
```

### Memory Optimization

```python
# Good - use generators for large datasets
def process_large_file(file_path):
    """Process large file using generator."""
    with open(file_path, 'r') as file:
        for line in file:
            yield process_line(line)

# Good - use numpy for numerical operations
import numpy as np

def calculate_statistics_numpy(data):
    """Calculate statistics using numpy."""
    arr = np.array(data)
    return {
        'mean': np.mean(arr),
        'std': np.std(arr),
        'min': np.min(arr),
        'max': np.max(arr)
    }

# Bad - inefficient for large datasets
def calculate_statistics_python(data):
    """Calculate statistics using pure Python."""
    return {
        'mean': sum(data) / len(data),
        'std': (sum((x - sum(data) / len(data)) ** 2 for x in data) / len(data)) ** 0.5,
        'min': min(data),
        'max': max(data)
    }
```

## Testing

### Unit Testing

```python
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DataProcessor([1, 2, 3, 4, 5])
    
    def test_initialization(self):
        """Test DataProcessor initialization."""
        self.assertEqual(len(self.processor.data), 5)
        self.assertFalse(self.processor.is_processed)
        self.assertEqual(self.processor.processing_history, [])
    
    def test_clean_data(self):
        """Test data cleaning functionality."""
        processor = DataProcessor([1, 2, None, 4, 5])
        result = processor.clean_data()
        
        self.assertIsInstance(result, DataProcessor)
        self.assertIn("cleaned", processor.processing_history)
    
    def test_empty_data(self):
        """Test handling of empty data."""
        processor = DataProcessor([])
        with self.assertRaises(ValueError):
            processor.process()
    
    def test_invalid_data_type(self):
        """Test handling of invalid data types."""
        with self.assertRaises(TypeError):
            DataProcessor("not a list")
    
    def tearDown(self):
        """Clean up after tests."""
        pass

# Run tests
if __name__ == '__main__':
    unittest.main()
```

### Property-Based Testing

```python
from hypothesis import given, strategies as st

class TestDataProcessing:
    """Property-based tests for data processing."""
    
    @given(st.lists(st.floats(min_value=-1000, max_value=1000)))
    def test_normalization_range(self, data):
        """Test that normalized data is always in [0, 1] range."""
        if len(data) > 0:
            normalized = normalize_data(data)
            assert all(0 <= x <= 1 for x in normalized)
    
    @given(st.lists(st.integers(min_value=1, max_value=100)))
    def test_statistics_properties(self, data):
        """Test statistical properties."""
        if len(data) > 0:
            stats = calculate_statistics(data)
            assert stats['min'] <= stats['mean'] <= stats['max']
            assert stats['std'] >= 0
```

## Code Organization

### Module Structure

```python
# Good - well-organized module
"""
Data processing utilities for machine learning.

This module provides functions and classes for cleaning, transforming,
and analyzing data for machine learning applications.
"""

# Imports
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union

# Constants
DEFAULT_THRESHOLD = 0.5
SUPPORTED_FORMATS = ['.csv', '.json', '.xlsx']

# Utility functions
def validate_data(data: List[float]) -> bool:
    """Validate that data contains only numeric values."""
    return all(isinstance(x, (int, float)) for x in data)

def normalize_data(data: List[float]) -> List[float]:
    """Normalize data to [0, 1] range."""
    if not data:
        return []
    min_val = min(data)
    max_val = max(data)
    if max_val == min_val:
        return [0.5] * len(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

# Classes
class DataProcessor:
    """Main data processing class."""
    pass

class DataValidator:
    """Data validation utilities."""
    pass

# Main execution
if __name__ == '__main__':
    # Example usage
    data = [1, 2, 3, 4, 5]
    print(f"Original: {data}")
    print(f"Normalized: {normalize_data(data)}")
```

### Package Structure

```
my_data_science_package/
├── __init__.py
├── README.md
├── setup.py
├── requirements.txt
├── tests/
│   ├── __init__.py
│   ├── test_data_processing.py
│   └── test_models.py
├── docs/
│   ├── api.md
│   └── examples.md
├── examples/
│   ├── basic_usage.py
│   └── advanced_usage.py
└── src/
    └── my_package/
        ├── __init__.py
        ├── data_processing/
        │   ├── __init__.py
        │   ├── cleaner.py
        │   └── transformer.py
        ├── models/
        │   ├── __init__.py
        │   ├── base.py
        │   └── linear.py
        └── utils/
            ├── __init__.py
            ├── helpers.py
            └── validators.py
```

## Security Best Practices

### Input Validation

```python
import re
from pathlib import Path

def validate_file_path(file_path: str) -> bool:
    """Validate file path for security."""
    # Check for path traversal attempts
    if '..' in file_path or file_path.startswith('/'):
        return False
    
    # Check for allowed file extensions
    allowed_extensions = {'.csv', '.json', '.xlsx', '.txt'}
    if not any(file_path.endswith(ext) for ext in allowed_extensions):
        return False
    
    return True

def sanitize_input(user_input: str) -> str:
    """Sanitize user input to prevent injection attacks."""
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\']', '', user_input)
    return sanitized.strip()

def load_data_safely(file_path: str) -> pd.DataFrame:
    """Load data with security validation."""
    if not validate_file_path(file_path):
        raise ValueError("Invalid file path")
    
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        raise
```

### Environment Variables

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Good - use environment variables for sensitive data
DATABASE_URL = os.getenv('DATABASE_URL')
API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')

# Validate required environment variables
required_vars = ['DATABASE_URL', 'API_KEY']
missing_vars = [var for var in required_vars if not os.getenv(var)]

if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {missing_vars}")

# Bad - hardcoded sensitive data
DATABASE_URL = "postgresql://user:password@localhost/db"
API_KEY = "sk-1234567890abcdef"
```

## Data Science Specific Practices

### Reproducibility

```python
import random
import numpy as np

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # Set other random seeds as needed
    # torch.manual_seed(seed)  # for PyTorch
    # tf.random.set_seed(seed)  # for TensorFlow

def save_experiment_config(config: Dict, file_path: str):
    """Save experiment configuration for reproducibility."""
    import json
    from datetime import datetime
    
    config['timestamp'] = datetime.now().isoformat()
    config['random_seed'] = 42
    
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2)

def load_experiment_config(file_path: str) -> Dict:
    """Load experiment configuration."""
    import json
    
    with open(file_path, 'r') as f:
        return json.load(f)
```

### Data Versioning

```python
import hashlib
import json
from pathlib import Path

def calculate_data_hash(data: pd.DataFrame) -> str:
    """Calculate hash of data for versioning."""
    # Convert to string and hash
    data_str = data.to_string()
    return hashlib.md5(data_str.encode()).hexdigest()

def save_data_version(data: pd.DataFrame, version: str, metadata: Dict = None):
    """Save data with version information."""
    data_hash = calculate_data_hash(data)
    
    version_info = {
        'version': version,
        'hash': data_hash,
        'timestamp': datetime.now().isoformat(),
        'shape': data.shape,
        'columns': list(data.columns),
        'metadata': metadata or {}
    }
    
    # Save data
    data.to_csv(f'data_v{version}.csv', index=False)
    
    # Save version info
    with open(f'data_v{version}_info.json', 'w') as f:
        json.dump(version_info, f, indent=2)
```

### Model Versioning

```python
import pickle
import joblib
from datetime import datetime

class ModelVersion:
    """Model versioning utility."""
    
    def __init__(self, model, name: str, version: str):
        self.model = model
        self.name = name
        self.version = version
        self.timestamp = datetime.now()
        self.metadata = {}
    
    def save(self, directory: str = "models"):
        """Save model with version information."""
        Path(directory).mkdir(exist_ok=True)
        
        # Save model
        model_path = f"{directory}/{self.name}_v{self.version}.pkl"
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'name': self.name,
            'version': self.version,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
        
        metadata_path = f"{directory}/{self.name}_v{self.version}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, name: str, version: str, directory: str = "models"):
        """Load model with version information."""
        model_path = f"{directory}/{name}_v{version}.pkl"
        metadata_path = f"{directory}/{name}_v{version}_metadata.json"
        
        model = joblib.load(model_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        instance = cls(model, name, version)
        instance.timestamp = datetime.fromisoformat(metadata['timestamp'])
        instance.metadata = metadata['metadata']
        
        return instance
```

## Tools and Utilities

### Code Formatting

```python
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3
  
  - repo: https://github.com/pycqa/isort
    rev: 5.10.1
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8

# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py38']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3
```

### Type Checking

```python
# mypy configuration
# mypy.ini
[mypy]
python_version = 3.8
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True

# Example with type hints
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
import pandas as pd

def process_data(
    data: Union[List[float], np.ndarray],
    threshold: float = 0.5,
    normalize: bool = True
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Process data with type hints.
    
    Parameters
    ----------
    data : Union[List[float], np.ndarray]
        Input data
    threshold : float, optional
        Processing threshold
    normalize : bool, optional
        Whether to normalize data
        
    Returns
    -------
    Tuple[np.ndarray, Dict[str, float]]
        Processed data and statistics
    """
    # Convert to numpy array if needed
    if isinstance(data, list):
        data = np.array(data)
    
    # Process data
    if normalize:
        data = (data - np.mean(data)) / np.std(data)
    
    # Calculate statistics
    stats = {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data))
    }
    
    return data, stats
```

### Logging

```python
import logging
from pathlib import Path

def setup_logging(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Usage
logger = setup_logging("data_processor", log_file="processing.log")

def process_data_with_logging(data):
    """Process data with comprehensive logging."""
    logger.info(f"Starting data processing for {len(data)} items")
    
    try:
        # Processing logic
        result = some_processing_function(data)
        logger.info("Data processing completed successfully")
        return result
    
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise
```

## Summary

Python best practices for data science include:

- **Code Style**: Follow PEP 8 guidelines for readability
- **Naming**: Use descriptive, consistent naming conventions
- **Documentation**: Write clear docstrings and comments
- **Error Handling**: Implement proper exception handling
- **Performance**: Use efficient data structures and algorithms
- **Testing**: Write comprehensive unit and integration tests
- **Organization**: Structure code logically and modularly
- **Security**: Validate inputs and protect sensitive data
- **Reproducibility**: Set random seeds and version data/models
- **Tools**: Use formatting, linting, and type checking tools

Following these best practices will help you write more maintainable, efficient, and professional data science code.

## Next Steps

- Set up pre-commit hooks for automatic code formatting
- Implement comprehensive testing for your data science projects
- Use type hints to improve code quality and IDE support
- Establish logging and monitoring for production systems

---

**Happy Coding with Best Practices!** 🐍✨ 