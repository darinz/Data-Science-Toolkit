# Supervised Learning Guide

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Level](https://img.shields.io/badge/Level-Intermediate-yellow.svg)](https://github.com/yourusername/Toolkit)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-orange.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-blue.svg)](https://pandas.pydata.org/)
[![Topics](https://img.shields.io/badge/Topics-Classification%2C%20Regression%2C%20ML-orange.svg)](https://github.com/yourusername/Toolkit)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](https://github.com/yourusername/Toolkit)

A comprehensive guide to supervised learning algorithms and techniques in Python for machine learning applications.

## Table of Contents
1. [Introduction to Supervised Learning](#introduction-to-supervised-learning)
2. [Data Preparation](#data-preparation)
3. [Classification Algorithms](#classification-algorithms)
4. [Regression Algorithms](#regression-algorithms)
5. [Model Evaluation](#model-evaluation)
6. [Feature Selection](#feature-selection)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Ensemble Methods](#ensemble-methods)
9. [Model Deployment](#model-deployment)

## Introduction to Supervised Learning

### What is Supervised Learning?

Supervised learning is a type of machine learning where the algorithm learns from labeled training data to make predictions on unseen data. The goal is to learn a mapping from input features to output labels.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Basic supervised learning workflow
def supervised_learning_workflow(X, y, model):
    """
    Basic supervised learning workflow
    
    Args:
        X: Feature matrix
        y: Target variable
        model: Machine learning model
    
    Returns:
        Trained model and predictions
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    return model, y_pred, y_test
```

### Types of Supervised Learning

#### Classification
- **Binary Classification**: Two classes (e.g., spam/not spam)
- **Multi-class Classification**: Multiple classes (e.g., digit recognition)
- **Multi-label Classification**: Multiple labels per instance

#### Regression
- **Linear Regression**: Predict continuous values
- **Polynomial Regression**: Non-linear relationships
- **Time Series Regression**: Sequential data

## Data Preparation

### Data Loading and Exploration

```python
# Load sample dataset
from sklearn.datasets import load_iris, load_boston
from sklearn.datasets import make_classification, make_regression

# Classification dataset
X_class, y_class = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    n_redundant=5, random_state=42
)

# Regression dataset
X_reg, y_reg = make_regression(
    n_samples=1000, n_features=20, n_informative=15,
    noise=0.1, random_state=42
)

# Real dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Create DataFrame for exploration
df_iris = pd.DataFrame(X_iris, columns=iris.feature_names)
df_iris['target'] = y_iris
```

### Data Preprocessing

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

def preprocess_data(X, y=None, categorical_cols=None, numerical_cols=None):
    """
    Comprehensive data preprocessing pipeline
    
    Args:
        X: Feature matrix
        y: Target variable (optional)
        categorical_cols: List of categorical column indices
        numerical_cols: List of numerical column indices
    
    Returns:
        Preprocessed features and target
    """
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Create preprocessing pipeline
    preprocessors = []
    
    if numerical_cols:
        numerical_transformer = StandardScaler()
        preprocessors.append(('num', numerical_transformer, numerical_cols))
    
    if categorical_cols:
        categorical_transformer = OneHotEncoder(drop='first', sparse=False)
        preprocessors.append(('cat', categorical_transformer, categorical_cols))
    
    # Apply preprocessing
    if preprocessors:
        preprocessor = ColumnTransformer(transformers=preprocessors)
        X_processed = preprocessor.fit_transform(X_imputed)
    else:
        X_processed = X_imputed
    
    return X_processed, y
```

### Feature Engineering

```python
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def create_feature_pipeline(X, y, task='classification', n_features=10):
    """
    Create feature engineering pipeline
    
    Args:
        X: Feature matrix
        y: Target variable
        task: 'classification' or 'regression'
        n_features: Number of features to select
    
    Returns:
        Feature engineering pipeline
    """
    if task == 'classification':
        selector = SelectKBest(score_func=f_classif, k=n_features)
    else:
        selector = SelectKBest(score_func=f_regression, k=n_features)
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', selector),
        ('pca', PCA(n_components=min(n_features, X.shape[1])))
    ])
    
    return pipeline
```

## Classification Algorithms

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def logistic_regression_classification(X, y):
    """
    Logistic regression for classification
    
    Args:
        X: Feature matrix
        y: Target labels
    
    Returns:
        Trained model and evaluation metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, report, conf_matrix, y_pred_proba

# Usage
model, acc, report, conf_mat, proba = logistic_regression_classification(X_class, y_class)
print(f"Accuracy: {acc:.4f}")
print(report)
```

### Support Vector Machines (SVM)

```python
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_auc_score, roc_curve

def svm_classification(X, y, kernel='rbf'):
    """
    Support Vector Machine for classification
    
    Args:
        X: Feature matrix
        y: Target labels
        kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
    
    Returns:
        Trained model and evaluation metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if kernel == 'linear':
        model = LinearSVC(random_state=42, max_iter=1000)
    else:
        model = SVC(kernel=kernel, random_state=42, probability=True)
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Get probabilities for ROC curve
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        y_pred_proba = model.decision_function(X_test_scaled)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return model, accuracy, auc, y_pred_proba

# Usage
model, acc, auc, proba = svm_classification(X_class, y_class, kernel='rbf')
print(f"Accuracy: {acc:.4f}, AUC: {auc:.4f}")
```

### Decision Trees

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import precision_recall_curve

def decision_tree_classification(X, y, max_depth=None):
    """
    Decision Tree for classification
    
    Args:
        X: Feature matrix
        y: Target labels
        max_depth: Maximum depth of the tree
    
    Returns:
        Trained model and evaluation metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = DecisionTreeClassifier(
        max_depth=max_depth, 
        random_state=42,
        min_samples_split=5,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = model.feature_importances_
    
    return model, accuracy, feature_importance, y_pred_proba

# Usage
model, acc, importance, proba = decision_tree_classification(X_class, y_class, max_depth=5)
print(f"Accuracy: {acc:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(range(len(importance)), importance)
plt.title('Feature Importance')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.show()
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

def random_forest_classification(X, y, n_estimators=100):
    """
    Random Forest for classification
    
    Args:
        X: Feature matrix
        y: Target labels
        n_estimators: Number of trees
    
    Returns:
        Trained model and evaluation metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    feature_importance = model.feature_importances_
    
    return model, accuracy, feature_importance, y_pred_proba

# Usage
model, acc, importance, proba = random_forest_classification(X_class, y_class)
print(f"Accuracy: {acc:.4f}")
```

## Regression Algorithms

### Linear Regression

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error

def linear_regression_model(X, y, model_type='linear'):
    """
    Linear regression models
    
    Args:
        X: Feature matrix
        y: Target values
        model_type: 'linear', 'ridge', or 'lasso'
    
    Returns:
        Trained model and evaluation metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Choose model
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'lasso':
        model = Lasso(alpha=1.0)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, mae, r2, y_pred

# Usage
model, mse, mae, r2, pred = linear_regression_model(X_reg, y_reg, 'ridge')
print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
```

### Support Vector Regression

```python
from sklearn.svm import SVR

def svr_regression(X, y, kernel='rbf'):
    """
    Support Vector Regression
    
    Args:
        X: Feature matrix
        y: Target values
        kernel: Kernel type
    
    Returns:
        Trained model and evaluation metrics
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = SVR(kernel=kernel, C=1.0, epsilon=0.1)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mse, mae, r2, y_pred

# Usage
model, mse, mae, r2, pred = svr_regression(X_reg, y_reg, 'rbf')
print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
```

## Model Evaluation

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)

def evaluate_classification_model(y_true, y_pred, y_pred_proba=None):
    """
    Comprehensive classification model evaluation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # ROC AUC (if probabilities available)
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    return metrics

def plot_classification_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Plot classification evaluation metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # ROC curve
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
    
    # Precision-Recall curve
    if y_pred_proba is not None:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        axes[1, 0].plot(recall, precision)
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
    
    # Metrics bar plot
    metrics = evaluate_classification_model(y_true, y_pred, y_pred_proba)
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
    metric_values = [metrics['accuracy'], metrics['precision'], 
                    metrics['recall'], metrics['f1']]
    
    axes[1, 1].bar(metric_names, metric_values)
    axes[1, 1].set_title('Classification Metrics')
    axes[1, 1].set_ylabel('Score')
    
    plt.tight_layout()
    plt.show()
```

### Regression Metrics

```python
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

def evaluate_regression_model(y_true, y_pred):
    """
    Comprehensive regression model evaluation
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
    
    return metrics

def plot_regression_metrics(y_true, y_pred):
    """
    Plot regression evaluation metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
    axes[0, 0].plot([y_true.min(), y_true.max()], 
                   [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Actual vs Predicted')
    
    # Residuals
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    
    # Residuals histogram
    axes[1, 0].hist(residuals, bins=30, alpha=0.7)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residuals Distribution')
    
    # Metrics bar plot
    metrics = evaluate_regression_model(y_true, y_pred)
    metric_names = ['MSE', 'RMSE', 'MAE', 'R²']
    metric_values = [metrics['mse'], metrics['rmse'], 
                    metrics['mae'], metrics['r2']]
    
    axes[1, 1].bar(metric_names, metric_values)
    axes[1, 1].set_title('Regression Metrics')
    axes[1, 1].set_ylabel('Score')
    
    plt.tight_layout()
    plt.show()
```

## Feature Selection

### Filter Methods

```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, mutual_info_classif,
    mutual_info_regression, chi2
)

def filter_feature_selection(X, y, task='classification', method='f_test', k=10):
    """
    Filter-based feature selection
    
    Args:
        X: Feature matrix
        y: Target variable
        task: 'classification' or 'regression'
        method: 'f_test', 'mutual_info', or 'chi2'
        k: Number of features to select
    
    Returns:
        Selected features and scores
    """
    if task == 'classification':
        if method == 'f_test':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'chi2':
            selector = SelectKBest(score_func=chi2, k=k)
    else:
        if method == 'f_test':
            selector = SelectKBest(score_func=f_regression, k=k)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
    
    # Fit and transform
    X_selected = selector.fit_transform(X, y)
    feature_scores = selector.scores_
    selected_features = selector.get_support()
    
    return X_selected, feature_scores, selected_features
```

### Wrapper Methods

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression

def wrapper_feature_selection(X, y, estimator=None, n_features=10):
    """
    Wrapper-based feature selection using Recursive Feature Elimination
    
    Args:
        X: Feature matrix
        y: Target variable
        estimator: Base estimator
        n_features: Number of features to select
    
    Returns:
        Selected features and ranking
    """
    if estimator is None:
        estimator = LogisticRegression(random_state=42)
    
    # RFE
    rfe = RFE(estimator=estimator, n_features_to_select=n_features)
    X_selected = rfe.fit_transform(X, y)
    
    # Get feature ranking
    feature_ranking = rfe.ranking_
    selected_features = rfe.support_
    
    return X_selected, feature_ranking, selected_features

def cross_validation_feature_selection(X, y, estimator=None):
    """
    Cross-validation based feature selection
    
    Args:
        X: Feature matrix
        y: Target variable
        estimator: Base estimator
    
    Returns:
        Optimal number of features and selected features
    """
    if estimator is None:
        estimator = LogisticRegression(random_state=42)
    
    # RFECV
    rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring='accuracy')
    X_selected = rfecv.fit_transform(X, y)
    
    optimal_features = rfecv.n_features_
    selected_features = rfecv.support_
    
    return X_selected, optimal_features, selected_features
```

## Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

def grid_search_tuning(X, y, model, param_grid, cv=5):
    """
    Grid search for hyperparameter tuning
    
    Args:
        X: Feature matrix
        y: Target variable
        model: Base model
        param_grid: Parameter grid
        cv: Number of cross-validation folds
    
    Returns:
        Best model and results
    """
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_score_, grid_search.cv_results_

# Example usage
param_grid = {
    'model__C': [0.1, 1, 10, 100],
    'model__gamma': [0.001, 0.01, 0.1, 1],
    'model__kernel': ['rbf', 'linear']
}

best_model, best_score, cv_results = grid_search_tuning(
    X_class, y_class, SVC(), param_grid
)
print(f"Best score: {best_score:.4f}")
print(f"Best parameters: {best_model.get_params()}")
```

### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def random_search_tuning(X, y, model, param_distributions, n_iter=100, cv=5):
    """
    Random search for hyperparameter tuning
    
    Args:
        X: Feature matrix
        y: Target variable
        model: Base model
        param_distributions: Parameter distributions
        n_iter: Number of iterations
        cv: Number of cross-validation folds
    
    Returns:
        Best model and results
    """
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Random search
    random_search = RandomizedSearchCV(
        pipeline, param_distributions, n_iter=n_iter, cv=cv, 
        scoring='accuracy', n_jobs=-1, random_state=42
    )
    random_search.fit(X, y)
    
    return random_search.best_estimator_, random_search.best_score_, random_search.cv_results_

# Example usage
param_distributions = {
    'model__C': uniform(0.1, 100),
    'model__gamma': uniform(0.001, 1),
    'model__kernel': ['rbf', 'linear']
}

best_model, best_score, cv_results = random_search_tuning(
    X_class, y_class, SVC(), param_distributions
)
```

## Ensemble Methods

### Voting Classifier

```python
from sklearn.ensemble import VotingClassifier

def create_voting_classifier(X, y):
    """
    Create voting classifier ensemble
    
    Args:
        X: Feature matrix
        y: Target labels
    
    Returns:
        Trained voting classifier
    """
    # Define base classifiers
    clf1 = LogisticRegression(random_state=42)
    clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf3 = SVC(probability=True, random_state=42)
    
    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)],
        voting='soft'
    )
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ensemble
    voting_clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = voting_clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return voting_clf, accuracy
```

### Stacking

```python
from sklearn.ensemble import StackingClassifier, StackingRegressor

def create_stacking_classifier(X, y):
    """
    Create stacking classifier ensemble
    
    Args:
        X: Feature matrix
        y: Target labels
    
    Returns:
        Trained stacking classifier
    """
    # Define base estimators
    estimators = [
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ]
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    # Split and train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ensemble
    stacking_clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = stacking_clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return stacking_clf, accuracy
```

## Model Deployment

### Model Persistence

```python
import joblib
import pickle

def save_model(model, filepath, method='joblib'):
    """
    Save trained model to file
    
    Args:
        model: Trained model
        filepath: Path to save model
        method: 'joblib' or 'pickle'
    """
    if method == 'joblib':
        joblib.dump(model, filepath)
    elif method == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)

def load_model(filepath, method='joblib'):
    """
    Load trained model from file
    
    Args:
        filepath: Path to model file
        method: 'joblib' or 'pickle'
    
    Returns:
        Loaded model
    """
    if method == 'joblib':
        return joblib.load(filepath)
    elif method == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)

# Usage
save_model(best_model, 'best_model.joblib')
loaded_model = load_model('best_model.joblib')
```

### Model Pipeline

```python
def create_complete_pipeline(X, y, task='classification'):
    """
    Create complete machine learning pipeline
    
    Args:
        X: Feature matrix
        y: Target variable
        task: 'classification' or 'regression'
    
    Returns:
        Complete pipeline
    """
    if task == 'classification':
        # Classification pipeline
        estimators = [
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(random_state=42))
        ]
        pipeline = Pipeline(estimators)
        
        # Define parameter grid
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l1', 'l2']
        }
        
    else:
        # Regression pipeline
        estimators = [
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]
        pipeline = Pipeline(estimators)
        
        # Define parameter grid
        param_grid = {}
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring='accuracy' if task == 'classification' else 'r2'
    )
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_

# Usage
complete_pipeline = create_complete_pipeline(X_class, y_class, 'classification')
save_model(complete_pipeline, 'complete_pipeline.joblib')
```

## Best Practices

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

def robust_model_evaluation(X, y, model, cv_folds=5):
    """
    Robust model evaluation using cross-validation
    
    Args:
        X: Feature matrix
        y: Target variable
        model: Machine learning model
        cv_folds: Number of cross-validation folds
    
    Returns:
        Cross-validation scores
    """
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Cross-validation
    cv_scores = cross_val_score(
        pipeline, X, y, cv=cv_folds, scoring='accuracy'
    )
    
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_scores
```

### Model Interpretability

```python
def interpret_model(model, feature_names, X_sample):
    """
    Model interpretation for linear models
    
    Args:
        model: Trained model
        feature_names: List of feature names
        X_sample: Sample data for interpretation
    """
    if hasattr(model, 'coef_'):
        # Linear model coefficients
        coefficients = model.coef_
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients
        })
        importance_df = importance_df.sort_values('coefficient', key=abs, ascending=False)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['coefficient'])
        plt.title('Feature Importance (Coefficients)')
        plt.xlabel('Coefficient Value')
        plt.show()
        
        return importance_df
```

## Exercises

1. **Binary Classification**: Build a spam detection model using different algorithms.
2. **Multi-class Classification**: Create a digit recognition system using the MNIST dataset.
3. **Regression**: Predict house prices using various regression techniques.
4. **Feature Selection**: Implement different feature selection methods and compare their performance.
5. **Ensemble Methods**: Build an ensemble model that outperforms individual models.

## Next Steps

After mastering supervised learning, explore:
- [Unsupervised Learning](unsupervised_learning_guide.md)
- [Model Evaluation](model_evaluation_guide.md)
- [Hyperparameter Tuning](hyperparameter_tuning_guide.md)
- [Deep Learning with Neural Networks](neural_networks_guide.md) 