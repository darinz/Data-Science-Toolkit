# Machine Learning Basics with scikit-learn

A comprehensive guide to fundamental machine learning concepts using scikit-learn.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
3. [Model Training](#model-training)
4. [Evaluation](#evaluation)
5. [Best Practices](#best-practices)

## Introduction

Machine learning enables computers to learn from data without explicit programming. scikit-learn is the most popular Python ML library.

### Types of ML:
- **Supervised**: Learning from labeled data (classification, regression)
- **Unsupervised**: Learning from unlabeled data (clustering, dimensionality reduction)
- **Reinforcement**: Learning through environment interaction

## Data Preparation

### Loading and Exploring Data

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, model_selection, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Load datasets
iris = datasets.load_iris()
breast_cancer = datasets.load_breast_cancer()
boston = datasets.load_boston()

# Explore data
def explore_dataset(dataset, name):
    print(f"\n=== {name} Dataset ===")
    print(f"Shape: {dataset.data.shape}")
    print(f"Features: {dataset.feature_names}")
    print(f"Target: {dataset.target_names}")
    
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target
    
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nBasic statistics:")
    print(df.describe())
    
    return df

iris_df = explore_dataset(iris, "Iris")
```

### Data Preprocessing

```python
# Handle missing values
def handle_missing_values(df):
    # Fill with mean for numeric columns
    df_filled = df.fillna(df.mean())
    return df_filled

# Feature scaling
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# Train-test split
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, 
                           random_state=random_state, stratify=y)

# Example usage
X = iris_df.drop('target', axis=1)
y = iris_df['target']
X_train, X_test, y_train, y_test = split_data(X, y)
X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
```

## Model Training

### Basic Classification Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def train_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True),
        'K-Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average='weighted')
        recall = metrics.recall_score(y_test, y_pred, average='weighted')
        f1 = metrics.f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    
    return results

# Train models
model_results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
```

### Model Comparison

```python
def compare_models(results):
    # Extract metrics
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    f1_scores = [results[name]['f1_score'] for name in model_names]
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = ax1.bar(model_names, accuracies, alpha=0.7, color='skyblue')
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='x', rotation=45)
    
    # F1-Score comparison
    bars2 = ax2.bar(model_names, f1_scores, alpha=0.7, color='lightcoral')
    ax2.set_title('F1-Score Comparison')
    ax2.set_ylabel('F1-Score')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bars, ax in [(bars1, ax1), (bars2, ax2)]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Find best model
    best_model_name = model_names[np.argmax(f1_scores)]
    print(f"\nBest model: {best_model_name}")
    
    return best_model_name

best_model = compare_models(model_results)
```

## Evaluation

### Cross-Validation

```python
def cross_validation_analysis(X, y):
    from sklearn.model_selection import cross_val_score
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42)
    }
    
    cv_results = {}
    
    for name, model in models.items():
        # 5-fold cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        cv_results[name] = {
            'scores': cv_scores,
            'mean': cv_scores.mean(),
            'std': cv_scores.std()
        }
        
        print(f"{name}:")
        print(f"  CV Scores: {cv_scores}")
        print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return cv_results

cv_results = cross_validation_analysis(X, y)
```

### Comprehensive Evaluation

```python
def comprehensive_evaluation(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Classification report
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(12, 5))
    
    # Confusion matrix plot
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        plt.subplot(1, 2, 2)
        feature_importance = pd.DataFrame({
            'feature': iris.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Feature Importance')
        plt.xlabel('Importance')
    
    plt.tight_layout()
    plt.show()
    
    return y_pred, y_pred_proba

# Evaluate best model
best_model_instance = model_results[best_model]['model']
y_pred, y_pred_proba = comprehensive_evaluation(best_model_instance, X_test_scaled, y_test)
```

### Overfitting Analysis

```python
def overfitting_analysis(X, y):
    # Test different model complexities
    max_depths = [1, 2, 3, 5, 10, 15, 20, None]
    
    train_scores = []
    test_scores = []
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    for depth in max_depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        
        # Train and evaluate
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
        
        print(f"Depth {depth}: Train={train_score:.4f}, Test={test_score:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    depth_labels = [str(d) if d is not None else 'None' for d in max_depths]
    x_pos = range(len(depth_labels))
    
    plt.plot(x_pos, train_scores, 'o-', label='Training Score', linewidth=2, markersize=8)
    plt.plot(x_pos, test_scores, 's-', label='Test Score', linewidth=2, markersize=8)
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Overfitting vs Underfitting Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(x_pos, depth_labels)
    
    # Find optimal depth
    optimal_idx = np.argmax(test_scores)
    optimal_depth = max_depths[optimal_idx]
    print(f"\nOptimal max_depth: {optimal_depth}")
    
    plt.show()
    
    return max_depths, train_scores, test_scores

overfitting_results = overfitting_analysis(X, y)
```

## Best Practices

### 1. Pipeline Creation

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def create_pipeline():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    return pipeline

# Create and use pipeline
pipeline = create_pipeline()
pipeline.fit(X_train, y_train)
pipeline_accuracy = pipeline.score(X_test, y_test)
print(f"Pipeline accuracy: {pipeline_accuracy:.4f}")
```

### 2. Model Persistence

```python
import joblib

def save_load_model(model, filename='model.pkl'):
    # Save model
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")
    
    # Load model
    loaded_model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    
    # Test loaded model
    loaded_accuracy = loaded_model.score(X_test, y_test)
    print(f"Loaded model accuracy: {loaded_accuracy:.4f}")
    
    return loaded_model

# Save and load model
loaded_model = save_load_model(pipeline)
```

### 3. Error Analysis

```python
def analyze_errors(y_true, y_pred, X_test):
    # Find incorrect predictions
    incorrect_mask = y_true != y_pred
    incorrect_indices = np.where(incorrect_mask)[0]
    
    print(f"Total predictions: {len(y_true)}")
    print(f"Incorrect predictions: {len(incorrect_indices)}")
    print(f"Error rate: {len(incorrect_indices) / len(y_true):.4f}")
    
    if len(incorrect_indices) > 0:
        print(f"\nIncorrect predictions:")
        for idx in incorrect_indices[:5]:  # Show first 5 errors
            true_label = iris.target_names[y_true[idx]]
            pred_label = iris.target_names[y_pred[idx]]
            print(f"  Sample {idx}: True={true_label}, Predicted={pred_label}")

# Analyze errors
analyze_errors(y_test, y_pred, X_test)
```

## Summary

### Key Concepts Covered:

1. **Data Preparation**: Loading, exploring, and preprocessing data
2. **Model Training**: Training multiple classification algorithms
3. **Evaluation**: Using various metrics and cross-validation
4. **Overfitting**: Understanding and detecting overfitting
5. **Best Practices**: Pipelines, model persistence, error analysis

### Next Steps:

- Explore supervised learning algorithms in detail
- Learn unsupervised learning techniques
- Master feature engineering
- Practice with real-world datasets
- Learn advanced techniques like ensemble methods

This foundation prepares you for more advanced machine learning concepts and real-world applications. 