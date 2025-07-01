# Feature Engineering with scikit-learn

A comprehensive guide to feature engineering: preprocessing, scaling, encoding, feature selection, and transformation.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Preprocessing](#data-preprocessing)
3. [Feature Scaling](#feature-scaling)
4. [Encoding Categorical Variables](#encoding-categorical-variables)
5. [Feature Generation](#feature-generation)
6. [Feature Selection](#feature-selection)
7. [Feature Transformation](#feature-transformation)
8. [Pipelines](#pipelines)

## Introduction

Feature engineering is the process of transforming raw data into features that better represent the underlying problem to predictive models, improving model accuracy.

## Data Preprocessing

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Example: Handling missing values
X = np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]])
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
print(X_imputed)
```

## Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
```

## Encoding Categorical Variables

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

# Example data
df = pd.DataFrame({'color': ['red', 'blue', 'green'], 'size': ['S', 'M', 'L']})

# One-hot encoding
onehot = OneHotEncoder(sparse=False)
print(onehot.fit_transform(df[['color']]))

# Label encoding
label = LabelEncoder()
print(label.fit_transform(df['size']))

# Ordinal encoding
ordinal = OrdinalEncoder(categories=[['S', 'M', 'L']])
print(ordinal.fit_transform(df[['size']]))
```

## Feature Generation

```python
# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
X = np.array([[2, 3], [3, 4], [4, 5]])
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print(X_poly)
```

## Feature Selection

```python
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X, y = iris.data, iris.target

# Univariate selection
selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)
print(X_new)

# Recursive Feature Elimination
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=2)
X_rfe = rfe.fit_transform(X, y)
print(X_rfe)
```

## Feature Transformation

```python
from sklearn.preprocessing import FunctionTransformer

# Log transformation
X = np.array([[1, 2], [3, 4], [5, 6]])
log_transformer = FunctionTransformer(np.log1p)
X_log = log_transformer.fit_transform(X)
print(X_log)
```

## Pipelines

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Example pipeline for mixed data
numeric_features = ['age', 'income']
categorical_features = ['gender']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Example DataFrame
data = pd.DataFrame({'age': [25, np.nan, 35], 'income': [50000, 60000, np.nan], 'gender': ['M', 'F', 'M']})
X_processed = preprocessor.fit_transform(data)
print(X_processed)
```

## Summary

- Handle missing values, scale features, and encode categorical variables.
- Generate new features and select the most relevant ones.
- Use pipelines for reproducible and robust feature engineering workflows. 