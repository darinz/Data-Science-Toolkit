# Python Object-Oriented Programming Guide

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![OOP](https://img.shields.io/badge/OOP-Object--Oriented-green.svg)](https://docs.python.org/3/tutorial/classes.html)

A comprehensive guide to Object-Oriented Programming in Python for data science and machine learning applications.

## Table of Contents

1. [Introduction to OOP](#introduction-to-oop)
2. [Classes and Objects](#classes-and-objects)
3. [Inheritance](#inheritance)
4. [Encapsulation](#encapsulation)
5. [Polymorphism](#polymorphism)
6. [Special Methods](#special-methods)
7. [Abstract Classes](#abstract-classes)
8. [Design Patterns](#design-patterns)
9. [OOP in Data Science](#oop-in-data-science)
10. [Best Practices](#best-practices)

## Introduction to OOP

Object-Oriented Programming (OOP) is a programming paradigm that organizes code into objects that contain data and code.

### Key Concepts

- **Class**: Blueprint for creating objects
- **Object**: Instance of a class
- **Attribute**: Data stored in objects
- **Method**: Functions that operate on objects
- **Inheritance**: Creating new classes from existing ones
- **Encapsulation**: Bundling data and methods together
- **Polymorphism**: Using objects of different types interchangeably

### Why OOP in Data Science?

```python
# Procedural approach
def calculate_mean(data):
    return sum(data) / len(data)

def calculate_std(data):
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return variance ** 0.5

# OOP approach
class Statistics:
    def __init__(self, data):
        self.data = data
    
    def mean(self):
        return sum(self.data) / len(self.data)
    
    def std(self):
        mean = self.mean()
        variance = sum((x - mean) ** 2 for x in self.data) / len(self.data)
        return variance ** 0.5
```

## Classes and Objects

### Basic Class Definition

```python
class DataPoint:
    """A simple class to represent a data point."""
    
    def __init__(self, x, y, label=None):
        """Initialize a data point with coordinates and optional label."""
        self.x = x
        self.y = y
        self.label = label
    
    def distance_to(self, other_point):
        """Calculate Euclidean distance to another point."""
        return ((self.x - other_point.x) ** 2 + 
                (self.y - other_point.y) ** 2) ** 0.5
    
    def __str__(self):
        """String representation of the data point."""
        return f"DataPoint({self.x}, {self.y}, label='{self.label}')"
    
    def __repr__(self):
        """Detailed string representation."""
        return f"DataPoint(x={self.x}, y={self.y}, label='{self.label}')"

# Creating objects
point1 = DataPoint(1, 2, "A")
point2 = DataPoint(4, 6, "B")

print(point1)  # DataPoint(1, 2, label='A')
print(point1.distance_to(point2))  # 5.0
```

### Class Variables vs Instance Variables

```python
class Dataset:
    # Class variable (shared by all instances)
    version = "1.0"
    
    def __init__(self, name, data):
        # Instance variables (unique to each instance)
        self.name = name
        self.data = data
        self.size = len(data)
    
    @classmethod
    def get_version(cls):
        """Class method to access class variables."""
        return cls.version
    
    @staticmethod
    def validate_data(data):
        """Static method that doesn't need instance or class."""
        return isinstance(data, (list, tuple)) and len(data) > 0

# Usage
dataset1 = Dataset("Training", [1, 2, 3, 4, 5])
dataset2 = Dataset("Test", [6, 7, 8])

print(Dataset.version)  # 1.0
print(dataset1.version)  # 1.0
print(dataset2.version)  # 1.0

# Class method
print(Dataset.get_version())  # 1.0

# Static method
print(Dataset.validate_data([1, 2, 3]))  # True
print(Dataset.validate_data([]))  # False
```

## Inheritance

### Basic Inheritance

```python
class BaseModel:
    """Base class for machine learning models."""
    
    def __init__(self, name):
        self.name = name
        self.is_trained = False
    
    def train(self, X, y):
        """Train the model."""
        raise NotImplementedError("Subclasses must implement train()")
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        raise NotImplementedError("Subclasses must implement predict()")
    
    def evaluate(self, X, y):
        """Evaluate model performance."""
        predictions = self.predict(X)
        # Basic accuracy calculation
        correct = sum(1 for p, t in zip(predictions, y) if p == t)
        return correct / len(y)

class LinearRegression(BaseModel):
    """Simple linear regression model."""
    
    def __init__(self, learning_rate=0.01):
        super().__init__("Linear Regression")
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
    
    def train(self, X, y, epochs=100):
        """Train using gradient descent."""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(epochs):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Gradients
            dw = (2/n_samples) * np.dot(X.T, (y_pred - y))
            db = (2/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        self.is_trained = True
    
    def predict(self, X):
        """Make predictions."""
        super().predict(X)  # Check if trained
        return np.dot(X, self.weights) + self.bias

# Usage
import numpy as np

# Create sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([2, 4, 6, 8])

# Create and train model
model = LinearRegression(learning_rate=0.01)
model.train(X, y, epochs=100)

# Make predictions
predictions = model.predict(X)
print(f"Predictions: {predictions}")
print(f"Accuracy: {model.evaluate(X, y):.2f}")
```

### Multiple Inheritance

```python
class DataProcessor:
    """Mixin for data processing capabilities."""
    
    def normalize(self, data):
        """Normalize data to [0, 1] range."""
        min_val = np.min(data)
        max_val = np.max(data)
        return (data - min_val) / (max_val - min_val)
    
    def standardize(self, data):
        """Standardize data to zero mean and unit variance."""
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std

class ModelEvaluator:
    """Mixin for model evaluation capabilities."""
    
    def calculate_rmse(self, y_true, y_pred):
        """Calculate Root Mean Square Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def calculate_mae(self, y_true, y_pred):
        """Calculate Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))

class AdvancedModel(BaseModel, DataProcessor, ModelEvaluator):
    """Advanced model with data processing and evaluation capabilities."""
    
    def __init__(self, name):
        super().__init__(name)
    
    def train(self, X, y):
        """Train with preprocessing."""
        # Preprocess data
        X_processed = self.standardize(X)
        # Training logic here...
        self.is_trained = True
    
    def predict(self, X):
        """Predict with preprocessing."""
        super().predict(X)
        X_processed = self.standardize(X)
        # Prediction logic here...
        return np.zeros(len(X))  # Placeholder
```

## Encapsulation

### Private and Protected Attributes

```python
class SecureModel:
    """Example of encapsulation with private and protected attributes."""
    
    def __init__(self, name):
        self.name = name  # Public attribute
        self._version = "1.0"  # Protected attribute (convention)
        self.__secret_key = "abc123"  # Private attribute
    
    def get_version(self):
        """Public method to access protected attribute."""
        return self._version
    
    def __get_secret(self):
        """Private method."""
        return self.__secret_key
    
    def public_method(self):
        """Public method that can access private members."""
        return f"Model: {self.name}, Secret: {self.__get_secret()}"

# Usage
model = SecureModel("MyModel")
print(model.name)  # MyModel
print(model.get_version())  # 1.0
print(model.public_method())  # Model: MyModel, Secret: abc123

# These would raise errors:
# print(model.__secret_key)  # AttributeError
# print(model.__get_secret())  # AttributeError
```

### Property Decorators

```python
class DataModel:
    """Example using property decorators for controlled access."""
    
    def __init__(self, data):
        self._data = data
        self._is_processed = False
    
    @property
    def data(self):
        """Getter for data."""
        return self._data
    
    @data.setter
    def data(self, value):
        """Setter for data with validation."""
        if not isinstance(value, (list, np.ndarray)):
            raise ValueError("Data must be a list or numpy array")
        self._data = value
        self._is_processed = False  # Reset processed flag
    
    @property
    def is_processed(self):
        """Read-only property."""
        return self._is_processed
    
    @property
    def size(self):
        """Computed property."""
        return len(self._data)
    
    def process_data(self):
        """Process the data."""
        if self.size == 0:
            raise ValueError("No data to process")
        # Processing logic here...
        self._is_processed = True

# Usage
model = DataModel([1, 2, 3, 4, 5])
print(model.size)  # 5
print(model.is_processed)  # False

model.process_data()
print(model.is_processed)  # True

# Property setter
model.data = [6, 7, 8]
print(model.is_processed)  # False (reset when data changed)
```

## Polymorphism

### Method Overriding

```python
class BaseClassifier:
    """Base class for classifiers."""
    
    def predict(self, X):
        """Base prediction method."""
        raise NotImplementedError
    
    def get_accuracy(self, X, y):
        """Calculate accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

class LogisticRegression(BaseClassifier):
    """Logistic regression classifier."""
    
    def predict(self, X):
        """Predict class labels."""
        # Simplified logistic regression prediction
        scores = np.dot(X, self.weights) + self.bias
        return (scores > 0).astype(int)

class RandomForest(BaseClassifier):
    """Random forest classifier."""
    
    def predict(self, X):
        """Predict class labels."""
        # Simplified random forest prediction
        predictions = []
        for x in X:
            # Simplified tree-based prediction
            pred = 1 if np.sum(x) > 0 else 0
            predictions.append(pred)
        return np.array(predictions)

# Polymorphic usage
classifiers = [
    LogisticRegression(),
    RandomForest()
]

X_test = np.array([[1, 2], [3, 4], [5, 6]])
y_test = np.array([1, 0, 1])

for classifier in classifiers:
    accuracy = classifier.get_accuracy(X_test, y_test)
    print(f"{classifier.__class__.__name__}: {accuracy:.2f}")
```

### Duck Typing

```python
class DataVisualizer:
    """Example of duck typing."""
    
    def plot(self, data_source):
        """Plot data from any object that has the required methods."""
        # Duck typing: we don't care about the type, just the interface
        if hasattr(data_source, 'get_data') and hasattr(data_source, 'get_labels'):
            data = data_source.get_data()
            labels = data_source.get_labels()
            # Plotting logic here...
            return f"Plotted {len(data)} points with labels: {labels}"
        else:
            raise TypeError("Data source must have get_data() and get_labels() methods")

class CSVDataSource:
    """CSV data source."""
    
    def __init__(self, filename):
        self.filename = filename
    
    def get_data(self):
        return [1, 2, 3, 4, 5]  # Simulated data
    
    def get_labels(self):
        return ["A", "B", "C", "D", "E"]

class DatabaseDataSource:
    """Database data source."""
    
    def __init__(self, connection_string):
        self.connection_string = connection_string
    
    def get_data(self):
        return [6, 7, 8, 9, 10]  # Simulated data
    
    def get_labels(self):
        return ["F", "G", "H", "I", "J"]

# Both work with the same visualizer
visualizer = DataVisualizer()

csv_source = CSVDataSource("data.csv")
db_source = DatabaseDataSource("db://localhost")

print(visualizer.plot(csv_source))  # Works
print(visualizer.plot(db_source))   # Works
```

## Special Methods

### Common Special Methods

```python
class DataCollection:
    """Example class with various special methods."""
    
    def __init__(self, name, data=None):
        self.name = name
        self.data = data or []
    
    def __len__(self):
        """Return the length of the collection."""
        return len(self.data)
    
    def __getitem__(self, index):
        """Allow indexing: collection[0]."""
        return self.data[index]
    
    def __setitem__(self, index, value):
        """Allow assignment: collection[0] = value."""
        self.data[index] = value
    
    def __contains__(self, item):
        """Allow 'in' operator: item in collection."""
        return item in self.data
    
    def __iter__(self):
        """Allow iteration: for item in collection."""
        return iter(self.data)
    
    def __add__(self, other):
        """Allow addition: collection1 + collection2."""
        if isinstance(other, DataCollection):
            combined_data = self.data + other.data
            return DataCollection(f"{self.name}+{other.name}", combined_data)
        return NotImplemented
    
    def __str__(self):
        """String representation."""
        return f"DataCollection('{self.name}', {len(self.data)} items)"
    
    def __repr__(self):
        """Detailed string representation."""
        return f"DataCollection(name='{self.name}', data={self.data})"
    
    def __eq__(self, other):
        """Equality comparison."""
        if isinstance(other, DataCollection):
            return self.name == other.name and self.data == other.data
        return False

# Usage
collection1 = DataCollection("A", [1, 2, 3])
collection2 = DataCollection("B", [4, 5, 6])

print(len(collection1))  # 3
print(collection1[0])    # 1
collection1[0] = 10
print(collection1[0])    # 10
print(10 in collection1) # True

for item in collection1:
    print(item)  # 10, 2, 3

combined = collection1 + collection2
print(combined)  # DataCollection('A+B', 6 items)
```

## Abstract Classes

### Using ABC (Abstract Base Classes)

```python
from abc import ABC, abstractmethod

class Model(ABC):
    """Abstract base class for models."""
    
    def __init__(self, name):
        self.name = name
        self.is_trained = False
    
    @abstractmethod
    def train(self, X, y):
        """Abstract method that must be implemented."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Abstract method that must be implemented."""
        pass
    
    def evaluate(self, X, y):
        """Concrete method that can be inherited."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

class ConcreteModel(Model):
    """Concrete implementation of the abstract Model class."""
    
    def __init__(self, name):
        super().__init__(name)
        self.weights = None
    
    def train(self, X, y):
        """Implement the abstract train method."""
        self.weights = np.random.randn(X.shape[1])
        self.is_trained = True
    
    def predict(self, X):
        """Implement the abstract predict method."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        return (np.dot(X, self.weights) > 0).astype(int)

# This works
model = ConcreteModel("MyModel")
model.train(np.array([[1, 2], [3, 4]]), np.array([0, 1]))

# This would raise an error
# model = Model("AbstractModel")  # TypeError
```

## Design Patterns

### Singleton Pattern

```python
class ModelRegistry:
    """Singleton pattern for model registry."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.models = {}
        return cls._instance
    
    def register_model(self, name, model):
        """Register a model."""
        self.models[name] = model
    
    def get_model(self, name):
        """Get a model by name."""
        return self.models.get(name)
    
    def list_models(self):
        """List all registered models."""
        return list(self.models.keys())

# Usage
registry1 = ModelRegistry()
registry2 = ModelRegistry()

print(registry1 is registry2)  # True (same instance)

registry1.register_model("model1", "some_model")
print(registry2.get_model("model1"))  # some_model
```

### Factory Pattern

```python
class ModelFactory:
    """Factory pattern for creating different types of models."""
    
    @staticmethod
    def create_model(model_type, **kwargs):
        """Create a model based on type."""
        if model_type == "linear":
            return LinearRegression(**kwargs)
        elif model_type == "logistic":
            return LogisticRegression(**kwargs)
        elif model_type == "random_forest":
            return RandomForest(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# Usage
factory = ModelFactory()
linear_model = factory.create_model("linear", learning_rate=0.01)
logistic_model = factory.create_model("logistic", learning_rate=0.1)
```

## OOP in Data Science

### Custom Dataset Class

```python
class CustomDataset:
    """Custom dataset class for machine learning."""
    
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx] if self.labels is not None else None
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label
    
    def split(self, train_ratio=0.8):
        """Split dataset into train and test sets."""
        n_train = int(len(self) * train_ratio)
        
        train_data = self.data[:n_train]
        train_labels = self.labels[:n_train] if self.labels is not None else None
        
        test_data = self.data[n_train:]
        test_labels = self.labels[n_train:] if self.labels is not None else None
        
        train_dataset = CustomDataset(train_data, train_labels, self.transform)
        test_dataset = CustomDataset(test_data, test_labels, self.transform)
        
        return train_dataset, test_dataset

# Usage
data = np.random.randn(100, 10)
labels = np.random.randint(0, 2, 100)

dataset = CustomDataset(data, labels)
train_ds, test_ds = dataset.split(0.8)

print(f"Train size: {len(train_ds)}")
print(f"Test size: {len(test_ds)}")
```

### Pipeline Class

```python
class DataPipeline:
    """Pipeline for data processing and model training."""
    
    def __init__(self, steps):
        self.steps = steps
    
    def fit(self, X, y):
        """Fit all steps in the pipeline."""
        X_transformed = X
        for step in self.steps:
            if hasattr(step, 'fit'):
                step.fit(X_transformed, y)
            if hasattr(step, 'transform'):
                X_transformed = step.transform(X_transformed)
        return self
    
    def predict(self, X):
        """Make predictions through the pipeline."""
        X_transformed = X
        for step in self.steps[:-1]:  # All except the last step
            if hasattr(step, 'transform'):
                X_transformed = step.transform(X_transformed)
        
        # Last step should be a model with predict method
        return self.steps[-1].predict(X_transformed)

# Example usage with custom steps
class StandardScaler:
    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        return (X - self.mean) / self.std

# Create pipeline
pipeline = DataPipeline([
    StandardScaler(),
    LinearRegression()
])

# Fit and predict
pipeline.fit(X, y)
predictions = pipeline.predict(X_test)
```

## Best Practices

### 1. Use Descriptive Names

```python
# Good
class CustomerDataProcessor:
    def process_customer_transactions(self):
        pass

# Bad
class DataProc:
    def proc(self):
        pass
```

### 2. Follow the Single Responsibility Principle

```python
# Good - each class has one responsibility
class DataLoader:
    def load_data(self):
        pass

class DataPreprocessor:
    def preprocess(self):
        pass

class ModelTrainer:
    def train(self):
        pass

# Bad - one class doing everything
class DataSciencePipeline:
    def load_data(self):
        pass
    
    def preprocess(self):
        pass
    
    def train(self):
        pass
```

### 3. Use Composition Over Inheritance

```python
# Good - composition
class ModelWithLogger:
    def __init__(self, model, logger):
        self.model = model
        self.logger = logger
    
    def train(self, X, y):
        self.logger.log("Starting training")
        self.model.train(X, y)
        self.logger.log("Training completed")

# Bad - inheritance
class ModelWithLogger(BaseModel):
    def __init__(self):
        super().__init__()
        self.logger = Logger()
```

### 4. Use Type Hints

```python
from typing import List, Optional, Union, Dict, Any

class DataProcessor:
    def __init__(self, data: List[float]) -> None:
        self.data = data
    
    def process(self, method: str = "mean") -> Union[float, List[float]]:
        if method == "mean":
            return np.mean(self.data)
        elif method == "std":
            return np.std(self.data)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_statistics(self) -> Dict[str, float]:
        return {
            "mean": np.mean(self.data),
            "std": np.std(self.data),
            "min": np.min(self.data),
            "max": np.max(self.data)
        }
```

### 5. Document Your Classes

```python
class ModelEvaluator:
    """
    A class for evaluating machine learning models.
    
    This class provides methods to calculate various evaluation metrics
    for classification and regression models.
    
    Attributes:
        metrics (Dict[str, callable]): Dictionary of available metrics
        
    Methods:
        evaluate: Evaluate a model using specified metrics
        cross_validate: Perform cross-validation
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator with default metrics."""
        self.metrics = {
            "accuracy": self._accuracy,
            "precision": self._precision,
            "recall": self._recall
        }
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                metrics: List[str] = None) -> Dict[str, float]:
        """
        Evaluate predictions using specified metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            metrics: List of metric names to calculate
            
        Returns:
            Dictionary of metric names and their values
            
        Raises:
            ValueError: If unknown metric is specified
        """
        if metrics is None:
            metrics = ["accuracy"]
        
        results = {}
        for metric in metrics:
            if metric in self.metrics:
                results[metric] = self.metrics[metric](y_true, y_pred)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        return results
```

## Summary

Object-Oriented Programming in Python provides powerful tools for organizing data science code:

- **Classes and Objects**: Create reusable, organized code structures
- **Inheritance**: Build upon existing functionality
- **Encapsulation**: Control access to data and methods
- **Polymorphism**: Write flexible, reusable code
- **Special Methods**: Customize object behavior
- **Design Patterns**: Solve common programming problems

Mastering OOP concepts will help you write more maintainable, scalable, and professional data science code.

## Next Steps

- Practice creating custom classes for your specific use cases
- Study design patterns relevant to data science
- Learn about Python's built-in classes and how to extend them
- Explore frameworks like scikit-learn to see OOP in action

---

**Happy Object-Oriented Programming!** 🐍✨ 