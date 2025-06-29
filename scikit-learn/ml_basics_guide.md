# Machine Learning Basics with scikit-learn

A comprehensive guide to fundamental machine learning concepts using scikit-learn, covering data preparation, model training, evaluation, and best practices.

## Table of Contents

1. [Introduction to Machine Learning](#introduction-to-machine-learning)
2. [Setting Up Your Environment](#setting-up-your-environment)
3. [Understanding Data](#understanding-data)
4. [Data Preprocessing](#data-preprocessing)
5. [Train-Test Split](#train-test-split)
6. [Model Training and Evaluation](#model-training-and-evaluation)
7. [Cross-Validation](#cross-validation)
8. [Overfitting and Underfitting](#overfitting-and-underfitting)
9. [Model Evaluation Metrics](#model-evaluation-metrics)
10. [Best Practices](#best-practices)

## Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. scikit-learn is one of the most popular Python libraries for machine learning.

### Types of Machine Learning

1. **Supervised Learning**: Learning from labeled data
   - Classification: Predicting categories/classes
   - Regression: Predicting continuous values

2. **Unsupervised Learning**: Learning from unlabeled data
   - Clustering: Grouping similar data points
   - Dimensionality Reduction: Reducing data complexity

3. **Reinforcement Learning**: Learning through interaction with environment

## Setting Up Your Environment

```python
# Import essential libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, model_selection, metrics
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

## Understanding Data

### Loading Built-in Datasets

scikit-learn provides several built-in datasets for learning and testing:

```python
# Load different types of datasets
from sklearn.datasets import load_iris, load_breast_cancer, load_boston, load_digits

# Classification datasets
iris = load_iris()
breast_cancer = load_breast_cancer()
digits = load_digits()

# Regression dataset
boston = load_boston()

print(f"Iris dataset shape: {iris.data.shape}")
print(f"Breast cancer dataset shape: {breast_cancer.data.shape}")
print(f"Digits dataset shape: {digits.data.shape}")
print(f"Boston dataset shape: {boston.data.shape}")
```

### Exploring Dataset Structure

```python
def explore_dataset(dataset, name):
    """Explore dataset structure and basic statistics"""
    print(f"\n=== {name} Dataset ===")
    print(f"Shape: {dataset.data.shape}")
    print(f"Target shape: {dataset.target.shape}")
    print(f"Feature names: {dataset.feature_names}")
    print(f"Target names: {dataset.target_names}")
    
    # Convert to DataFrame for easier exploration
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target
    
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nBasic statistics:")
    print(df.describe())
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    return df

# Explore Iris dataset
iris_df = explore_dataset(iris, "Iris")
```

### Data Visualization

```python
def visualize_dataset(df, target_col='target', dataset_name="Dataset"):
    """Create comprehensive visualizations for dataset exploration"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{dataset_name} Dataset Analysis', fontsize=16)
    
    # 1. Target distribution
    axes[0, 0].hist(df[target_col], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Target Distribution')
    axes[0, 0].set_xlabel('Target')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=axes[0, 1], fmt='.2f')
    axes[0, 1].set_title('Correlation Matrix')
    
    # 3. Feature distributions
    feature_cols = [col for col in df.columns if col != target_col]
    for i, col in enumerate(feature_cols[:4]):  # Show first 4 features
        row, col_idx = divmod(i, 2)
        axes[row, col_idx + 2].hist(df[col], bins=20, alpha=0.7, edgecolor='black')
        axes[row, col_idx + 2].set_title(f'{col} Distribution')
        axes[row, col_idx + 2].set_xlabel(col)
        axes[row, col_idx + 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # 4. Pair plot for smaller datasets
    if len(feature_cols) <= 6:
        sns.pairplot(df, hue=target_col, diag_kind='hist')
        plt.show()

# Visualize Iris dataset
visualize_dataset(iris_df, 'target', 'Iris')
```

## Data Preprocessing

### Handling Missing Values

```python
def handle_missing_values(df):
    """Demonstrate different strategies for handling missing values"""
    
    # Create a sample dataset with missing values
    sample_df = df.copy()
    
    # Introduce some missing values randomly
    np.random.seed(42)
    missing_mask = np.random.random(sample_df.shape) < 0.1
    sample_df[missing_mask] = np.nan
    
    print("Original dataset shape:", df.shape)
    print("Dataset with missing values shape:", sample_df.shape)
    print("Missing values count:")
    print(sample_df.isnull().sum())
    
    # Strategy 1: Remove rows with missing values
    df_drop = sample_df.dropna()
    print(f"\nAfter dropping rows with missing values: {df_drop.shape}")
    
    # Strategy 2: Fill with mean (for numeric columns)
    df_fill_mean = sample_df.fillna(sample_df.mean())
    print(f"After filling with mean: {df_fill_mean.shape}")
    
    # Strategy 3: Fill with median
    df_fill_median = sample_df.fillna(sample_df.median())
    print(f"After filling with median: {df_fill_median.shape}")
    
    # Strategy 4: Forward fill
    df_fill_ffill = sample_df.fillna(method='ffill')
    print(f"After forward fill: {df_fill_ffill.shape}")
    
    return df_fill_mean  # Return the mean-filled dataset

# Handle missing values
iris_clean = handle_missing_values(iris_df)
```

### Feature Scaling

```python
def demonstrate_scaling(X, y):
    """Demonstrate different scaling techniques"""
    
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    
    # Split data first
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Different scaling methods
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }
    
    scaled_data = {}
    
    for name, scaler in scalers.items():
        # Fit on training data
        X_train_scaled = scaler.fit_transform(X_train)
        # Transform test data
        X_test_scaled = scaler.transform(X_test)
        
        scaled_data[name] = {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'scaler': scaler
        }
        
        print(f"\n{name}:")
        print(f"Training data - Mean: {X_train_scaled.mean():.3f}, Std: {X_train_scaled.std():.3f}")
        print(f"Training data - Min: {X_train_scaled.min():.3f}, Max: {X_train_scaled.max():.3f}")
    
    return scaled_data

# Demonstrate scaling
X = iris_df.drop('target', axis=1)
y = iris_df['target']
scaled_data = demonstrate_scaling(X, y)
```

### Encoding Categorical Variables

```python
def demonstrate_encoding():
    """Demonstrate different encoding techniques for categorical variables"""
    
    # Create sample categorical data
    categorical_data = pd.DataFrame({
        'color': ['red', 'blue', 'green', 'red', 'blue'] * 20,
        'size': ['small', 'medium', 'large', 'small', 'medium'] * 20,
        'brand': ['A', 'B', 'C', 'A', 'B'] * 20,
        'target': np.random.randint(0, 2, 100)
    })
    
    print("Original categorical data:")
    print(categorical_data.head())
    print(f"\nShape: {categorical_data.shape}")
    
    # 1. Label Encoding
    from sklearn.preprocessing import LabelEncoder
    
    label_encoded = categorical_data.copy()
    label_encoders = {}
    
    for column in ['color', 'size', 'brand']:
        le = LabelEncoder()
        label_encoded[column] = le.fit_transform(categorical_data[column])
        label_encoders[column] = le
    
    print("\nLabel Encoded:")
    print(label_encoded.head())
    
    # 2. One-Hot Encoding
    from sklearn.preprocessing import OneHotEncoder
    
    onehot_encoded = pd.get_dummies(categorical_data, columns=['color', 'size', 'brand'])
    
    print("\nOne-Hot Encoded:")
    print(onehot_encoded.head())
    print(f"Shape after one-hot encoding: {onehot_encoded.shape}")
    
    # 3. Ordinal Encoding (for ordinal categories)
    from sklearn.preprocessing import OrdinalEncoder
    
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoded = categorical_data.copy()
    ordinal_encoded[['color', 'size', 'brand']] = ordinal_encoder.fit_transform(
        categorical_data[['color', 'size', 'brand']]
    )
    
    print("\nOrdinal Encoded:")
    print(ordinal_encoded.head())
    
    return label_encoded, onehot_encoded, ordinal_encoded

# Demonstrate encoding
label_encoded, onehot_encoded, ordinal_encoded = demonstrate_encoding()
```

## Train-Test Split

```python
def demonstrate_train_test_split(X, y):
    """Demonstrate different train-test split strategies"""
    
    # 1. Basic train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Basic split:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training set target distribution: {np.bincount(y_train)}")
    print(f"Test set target distribution: {np.bincount(y_test)}")
    
    # 2. Stratified split (for classification)
    X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nStratified split:")
    print(f"Training set target distribution: {np.bincount(y_train_strat)}")
    print(f"Test set target distribution: {np.bincount(y_test_strat)}")
    
    # 3. Different split ratios
    split_ratios = [0.1, 0.2, 0.3, 0.4]
    
    for ratio in split_ratios:
        X_train_ratio, X_test_ratio, y_train_ratio, y_test_ratio = train_test_split(
            X, y, test_size=ratio, random_state=42, stratify=y
        )
        print(f"\nSplit ratio {ratio}:")
        print(f"Training: {X_train_ratio.shape[0]}, Test: {X_test_ratio.shape[0]}")
    
    return X_train, X_test, y_train, y_test

# Demonstrate train-test split
X_train, X_test, y_train, y_test = demonstrate_train_test_split(X, y)
```

## Model Training and Evaluation

### Basic Model Training

```python
def train_basic_models(X_train, X_test, y_train, y_test):
    """Train and evaluate basic classification models"""
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
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
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"{name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    
    return results

# Train basic models
model_results = train_basic_models(X_train, X_test, y_train, y_test)
```

### Model Comparison

```python
def compare_models(results):
    """Compare model performance and create visualizations"""
    
    # Extract metrics for comparison
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    precisions = [results[name]['precision'] for name in model_names]
    recalls = [results[name]['recall'] for name in model_names]
    f1_scores = [results[name]['f1_score'] for name in model_names]
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    metrics_data = [accuracies, precisions, recalls, f1_scores]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for i, (metric_data, metric_name) in enumerate(zip(metrics_data, metric_names)):
        row, col = divmod(i, 2)
        bars = axes[row, col].bar(model_names, metric_data, alpha=0.7, color='skyblue')
        axes[row, col].set_title(f'{metric_name} Comparison')
        axes[row, col].set_ylabel(metric_name)
        axes[row, col].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_data):
            axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Create summary table
    comparison_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'Precision': precisions,
        'Recall': recalls,
        'F1-Score': f1_scores
    })
    
    print("\nModel Performance Summary:")
    print(comparison_df.round(4))
    
    # Find best model
    best_model_name = comparison_df.loc[comparison_df['F1-Score'].idxmax(), 'Model']
    print(f"\nBest model based on F1-Score: {best_model_name}")
    
    return comparison_df

# Compare models
comparison_df = compare_models(model_results)
```

## Cross-Validation

```python
def demonstrate_cross_validation(X, y):
    """Demonstrate different cross-validation strategies"""
    
    # Define models to test
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42)
    }
    
    # Different CV strategies
    cv_strategies = {
        '5-Fold CV': 5,
        '10-Fold CV': 10,
        'Stratified 5-Fold CV': model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        'Leave-One-Out CV': model_selection.LeaveOneOut()
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n=== {model_name} ===")
        model_results = {}
        
        for cv_name, cv_strategy in cv_strategies.items():
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='accuracy')
            
            model_results[cv_name] = {
                'scores': cv_scores,
                'mean': cv_scores.mean(),
                'std': cv_scores.std()
            }
            
            print(f"{cv_name}:")
            print(f"  Scores: {cv_scores}")
            print(f"  Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        results[model_name] = model_results
    
    # Visualize CV results
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5))
    fig.suptitle('Cross-Validation Results Comparison', fontsize=16)
    
    for i, (model_name, model_results) in enumerate(results.items()):
        cv_names = list(model_results.keys())
        means = [model_results[cv]['mean'] for cv in cv_names]
        stds = [model_results[cv]['std'] for cv in cv_names]
        
        bars = axes[i].bar(cv_names, means, yerr=stds, capsize=5, alpha=0.7, color='lightcoral')
        axes[i].set_title(f'{model_name}')
        axes[i].set_ylabel('Accuracy')
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_ylim(0, 1)
        
        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{mean:.3f}\n±{std:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    return results

# Demonstrate cross-validation
cv_results = demonstrate_cross_validation(X, y)
```

## Overfitting and Underfitting

```python
def demonstrate_overfitting_underfitting(X, y):
    """Demonstrate overfitting and underfitting with different model complexities"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Test different model complexities
    # Decision Tree with different max_depths
    max_depths = [1, 2, 3, 5, 10, 15, 20, None]
    
    train_scores = []
    test_scores = []
    
    for depth in max_depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Calculate scores
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)
        
        print(f"Max Depth: {depth}, Train Score: {train_score:.4f}, Test Score: {test_score:.4f}")
    
    # Plot learning curves
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Model complexity vs performance
    plt.subplot(1, 2, 1)
    depth_labels = [str(d) if d is not None else 'None' for d in max_depths]
    x_pos = range(len(depth_labels))
    
    plt.plot(x_pos, train_scores, 'o-', label='Training Score', linewidth=2, markersize=8)
    plt.plot(x_pos, test_scores, 's-', label='Test Score', linewidth=2, markersize=8)
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Model Complexity vs Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(x_pos, depth_labels)
    
    # Plot 2: Overfitting/Underfitting visualization
    plt.subplot(1, 2, 2)
    plt.plot(x_pos, train_scores, 'o-', label='Training Score', linewidth=2, markersize=8)
    plt.plot(x_pos, test_scores, 's-', label='Test Score', linewidth=2, markersize=8)
    plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='Good Performance')
    plt.xlabel('Model Complexity (Max Depth)')
    plt.ylabel('Accuracy')
    plt.title('Overfitting vs Underfitting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(x_pos, depth_labels)
    
    # Add annotations
    plt.annotate('Underfitting', xy=(0, train_scores[0]), xytext=(1, 0.6),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    plt.annotate('Overfitting', xy=(len(max_depths)-1, train_scores[-1]), xytext=(len(max_depths)-2, 0.6),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal complexity
    optimal_idx = np.argmax(test_scores)
    optimal_depth = max_depths[optimal_idx]
    print(f"\nOptimal max_depth: {optimal_depth}")
    print(f"Optimal test score: {test_scores[optimal_idx]:.4f}")
    
    return max_depths, train_scores, test_scores

# Demonstrate overfitting/underfitting
depths, train_scores, test_scores = demonstrate_overfitting_underfitting(X, y)
```

## Model Evaluation Metrics

```python
def comprehensive_model_evaluation(X_train, X_test, y_train, y_test):
    """Demonstrate comprehensive model evaluation with multiple metrics"""
    
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # 1. Classification Report
    print("Classification Report:")
    print(metrics.classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # 2. Confusion Matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(15, 5))
    
    # Confusion Matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 3. ROC Curve (for binary classification, we'll use one-vs-rest)
    plt.subplot(1, 3, 2)
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    
    # Binarize the output
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]
    
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    y_score = classifier.fit(X_train, label_binarize(y_train, classes=[0, 1, 2])).predict_proba(X_test)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    colors = ['blue', 'red', 'green']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'ROC curve of class {iris.target_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    
    # 4. Precision-Recall Curve
    plt.subplot(1, 3, 3)
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = metrics.precision_recall_curve(y_bin[:, i], y_score[:, i])
        avg_precision[i] = metrics.average_precision_score(y_bin[:, i], y_score[:, i])
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                label=f'Precision-Recall curve of class {iris.target_names[i]} (AP = {avg_precision[i]:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    
    plt.tight_layout()
    plt.show()
    
    # 5. Additional metrics
    print(f"\nAdditional Metrics:")
    print(f"Accuracy: {metrics.accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy: {metrics.balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"Hamming Loss: {metrics.hamming_loss(y_test, y_pred):.4f}")
    print(f"Jaccard Score: {metrics.jaccard_score(y_test, y_pred, average='weighted'):.4f}")
    
    # 6. Feature importance (for tree-based models)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': iris.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        print(f"\nFeature Importance:")
        print(feature_importance)
    
    return model, y_pred, y_pred_proba

# Comprehensive evaluation
best_model, y_pred, y_pred_proba = comprehensive_model_evaluation(X_train, X_test, y_train, y_test)
```

## Best Practices

### 1. Data Preprocessing Pipeline

```python
def create_preprocessing_pipeline():
    """Create a comprehensive preprocessing pipeline"""
    
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    
    # Define numeric and categorical features
    numeric_features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    categorical_features = []  # No categorical features in iris dataset
    
    # Create preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

# Create preprocessing pipeline
preprocessor = create_preprocessing_pipeline()
```

### 2. Complete ML Pipeline

```python
def create_complete_pipeline():
    """Create a complete ML pipeline with preprocessing and model"""
    
    # Create the complete pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Train the pipeline
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"Pipeline Accuracy: {accuracy:.4f}")
    
    return pipeline

# Create and test complete pipeline
ml_pipeline = create_complete_pipeline()
```

### 3. Model Persistence

```python
def demonstrate_model_persistence(model, filename='model.pkl'):
    """Demonstrate how to save and load models"""
    
    import joblib
    
    # Save the model
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")
    
    # Load the model
    loaded_model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    
    # Test the loaded model
    y_pred_loaded = loaded_model.predict(X_test)
    accuracy_loaded = metrics.accuracy_score(y_test, y_pred_loaded)
    print(f"Loaded model accuracy: {accuracy_loaded:.4f}")
    
    return loaded_model

# Demonstrate model persistence
loaded_model = demonstrate_model_persistence(ml_pipeline)
```

### 4. Error Analysis

```python
def analyze_errors(y_true, y_pred, X_test):
    """Analyze prediction errors to understand model weaknesses"""
    
    # Find incorrect predictions
    incorrect_mask = y_true != y_pred
    incorrect_indices = np.where(incorrect_mask)[0]
    
    print(f"Total predictions: {len(y_true)}")
    print(f"Incorrect predictions: {len(incorrect_indices)}")
    print(f"Error rate: {len(incorrect_indices) / len(y_true):.4f}")
    
    if len(incorrect_indices) > 0:
        print(f"\nIncorrect predictions:")
        for idx in incorrect_indices:
            true_label = iris.target_names[y_true[idx]]
            pred_label = iris.target_names[y_pred[idx]]
            features = X_test.iloc[idx] if hasattr(X_test, 'iloc') else X_test[idx]
            print(f"  Sample {idx}: True={true_label}, Predicted={pred_label}")
            print(f"    Features: {features}")
    
    # Analyze error patterns
    error_confusion = metrics.confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(error_confusion, annot=True, fmt='d', cmap='Reds',
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title('Error Analysis - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Analyze errors
analyze_errors(y_test, y_pred, X_test)
```

## Summary

This comprehensive guide covers the fundamental concepts of machine learning with scikit-learn:

### Key Takeaways:

1. **Data Understanding**: Always explore your data before modeling
2. **Preprocessing**: Handle missing values, scale features, and encode categorical variables
3. **Train-Test Split**: Use stratified splits for classification problems
4. **Model Selection**: Compare multiple algorithms and use cross-validation
5. **Evaluation**: Use appropriate metrics for your problem type
6. **Overfitting**: Monitor training vs test performance
7. **Pipelines**: Create reproducible workflows
8. **Persistence**: Save and load models for deployment

### Next Steps:

- Explore supervised learning algorithms in detail
- Learn about unsupervised learning techniques
- Master feature engineering and selection
- Practice with real-world datasets
- Learn advanced techniques like ensemble methods and hyperparameter tuning

This foundation will prepare you for more advanced machine learning concepts and real-world applications. 