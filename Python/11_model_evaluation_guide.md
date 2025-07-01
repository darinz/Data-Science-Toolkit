# Python Model Evaluation Guide

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Evaluation](https://img.shields.io/badge/Model-Evaluation-green.svg)](https://scikit-learn.org/stable/modules/model_evaluation.html)

A comprehensive guide to Model Evaluation in Python for data science and machine learning applications.

## Table of Contents

1. [Introduction to Model Evaluation](#introduction-to-model-evaluation)
2. [Classification Metrics](#classification-metrics)
3. [Regression Metrics](#regression-metrics)
4. [Cross-Validation](#cross-validation)
5. [Bias-Variance Tradeoff](#bias-variance-tradeoff)
6. [Model Comparison](#model-comparison)
7. [Feature Importance](#feature-importance)
8. [Model Interpretability](#model-interpretability)
9. [Best Practices](#best-practices)

## Introduction to Model Evaluation

Model evaluation is crucial for assessing model performance and making informed decisions about model selection and deployment.

### Evaluation Goals

- Assess model performance accurately
- Compare different models
- Detect overfitting and underfitting
- Understand model behavior
- Make deployment decisions

### Basic Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_auc_score, confusion_matrix, classification_report,
                           mean_squared_error, mean_absolute_error, r2_score)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import warnings
warnings.filterwarnings('ignore')

# Sample datasets
np.random.seed(42)

# Classification dataset
X_clf = np.random.randn(1000, 10)
y_clf = (X_clf[:, 0] + X_clf[:, 1] > 0).astype(int)

# Regression dataset
X_reg = np.random.randn(1000, 10)
y_reg = X_reg[:, 0] * 2 + X_reg[:, 1] * 3 + np.random.normal(0, 0.1, 1000)

# Split data
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
```

## Classification Metrics

### Basic Classification Metrics

```python
def evaluate_classification_model(y_true, y_pred, y_prob=None):
    """Evaluate classification model with multiple metrics."""
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC AUC (if probabilities available)
    auc = None
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    
    # Print results
    print("=== CLASSIFICATION EVALUATION ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if auc:
        print(f"ROC AUC: {auc:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm
    }

# Train and evaluate classification model
clf_model = LogisticRegression(random_state=42)
clf_model.fit(X_clf_train, y_clf_train)
y_clf_pred = clf_model.predict(X_clf_test)
y_clf_prob = clf_model.predict_proba(X_clf_test)[:, 1]

# Evaluate
clf_results = evaluate_classification_model(y_clf_test, y_clf_pred, y_clf_prob)
```

### Advanced Classification Metrics

```python
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

def advanced_classification_metrics(y_true, y_prob):
    """Calculate advanced classification metrics."""
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    
    # Plot curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    ax1.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_roc:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True)
    
    # Precision-Recall Curve
    ax2.plot(recall, precision, label=f'PR Curve (AP = {auc_pr:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'roc_auc': auc_roc,
        'pr_auc': auc_pr,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall
    }

# Calculate advanced metrics
advanced_metrics = advanced_classification_metrics(y_clf_test, y_clf_prob)
```

## Regression Metrics

### Basic Regression Metrics

```python
def evaluate_regression_model(y_true, y_pred):
    """Evaluate regression model with multiple metrics."""
    
    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean absolute percentage error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Print results
    print("=== REGRESSION EVALUATION ===")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid(True)
    plt.show()
    
    # Residual plot
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True)
    plt.show()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

# Train and evaluate regression model
reg_model = LinearRegression()
reg_model.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_model.predict(X_reg_test)

# Evaluate
reg_results = evaluate_regression_model(y_reg_test, y_reg_pred)
```

### Advanced Regression Metrics

```python
from sklearn.metrics import explained_variance_score, max_error

def advanced_regression_metrics(y_true, y_pred):
    """Calculate advanced regression metrics."""
    
    # Explained variance score
    explained_var = explained_variance_score(y_true, y_pred)
    
    # Maximum error
    max_err = max_error(y_true, y_pred)
    
    # Median absolute error
    from sklearn.metrics import median_absolute_error
    medae = median_absolute_error(y_true, y_pred)
    
    # Print advanced metrics
    print("=== ADVANCED REGRESSION METRICS ===")
    print(f"Explained Variance Score: {explained_var:.4f}")
    print(f"Maximum Error: {max_err:.4f}")
    print(f"Median Absolute Error: {medae:.4f}")
    
    # Distribution of residuals
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Residual histogram
    ax1.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Residuals')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Residuals')
    ax1.grid(True)
    
    # Q-Q plot for normality
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'explained_variance': explained_var,
        'max_error': max_err,
        'median_ae': medae,
        'residuals': residuals
    }

# Calculate advanced regression metrics
advanced_reg_metrics = advanced_regression_metrics(y_reg_test, y_reg_pred)
```

## Cross-Validation

### K-Fold Cross-Validation

```python
def perform_cross_validation(X, y, model, cv=5, scoring='accuracy'):
    """Perform k-fold cross-validation."""
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    # Print results
    print(f"=== {cv}-FOLD CROSS-VALIDATION RESULTS ===")
    print(f"Mean {scoring}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Individual scores: {cv_scores}")
    
    # Plot cross-validation scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, cv + 1), cv_scores, 'bo-')
    plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
    plt.fill_between(range(1, cv + 1), 
                     cv_scores.mean() - cv_scores.std(),
                     cv_scores.mean() + cv_scores.std(), 
                     alpha=0.2, color='r')
    plt.xlabel('Fold')
    plt.ylabel(f'{scoring.title()} Score')
    plt.title(f'{cv}-Fold Cross-Validation Results')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return cv_scores

# Perform cross-validation for classification
clf_cv_scores = perform_cross_validation(X_clf, y_clf, LogisticRegression(), cv=5, scoring='accuracy')

# Perform cross-validation for regression
reg_cv_scores = perform_cross_validation(X_reg, y_reg, LinearRegression(), cv=5, scoring='r2')
```

### Stratified Cross-Validation

```python
from sklearn.model_selection import StratifiedKFold

def stratified_cross_validation(X, y, model, cv=5):
    """Perform stratified k-fold cross-validation for classification."""
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Train model
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_val_fold)
        
        # Calculate score
        score = accuracy_score(y_val_fold, y_pred_fold)
        scores.append(score)
        
        print(f"Fold {fold}: {score:.4f}")
    
    print(f"\nMean Accuracy: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
    
    return scores

# Perform stratified cross-validation
stratified_scores = stratified_cross_validation(X_clf, y_clf, LogisticRegression())
```

## Bias-Variance Tradeoff

### Learning Curves

```python
from sklearn.model_selection import learning_curve

def plot_learning_curves(X, y, model, title="Learning Curves"):
    """Plot learning curves to understand bias-variance tradeoff."""
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    
    return train_sizes, train_mean, val_mean

# Plot learning curves for different models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

for name, model in models.items():
    plot_learning_curves(X_clf, y_clf, model, f"Learning Curves - {name}")
```

## Model Comparison

### Model Comparison Framework

```python
def compare_models(X_train, X_test, y_train, y_test, models, task='classification'):
    """Compare multiple models and their performance."""
    
    results = {}
    
    for name, model in models.items():
        print(f"\n=== EVALUATING {name.upper()} ===")
        
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        if task == 'classification':
            # Classification metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
        elif task == 'regression':
            # Regression metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            print(f"R²: {r2:.4f}")
    
    return results

# Compare classification models
clf_models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42)
}

clf_comparison = compare_models(X_clf_train, X_clf_test, y_clf_train, y_clf_test, 
                               clf_models, task='classification')

# Compare regression models
reg_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': SVR()
}

reg_comparison = compare_models(X_reg_train, X_reg_test, y_reg_train, y_reg_test, 
                               reg_models, task='regression')
```

### Visualization of Model Comparison

```python
def visualize_model_comparison(results, metric='accuracy'):
    """Visualize model comparison results."""
    
    # Extract metric values
    models = list(results.keys())
    values = [results[model][metric] for model in models]
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color=['skyblue', 'lightgreen', 'lightcoral'])
    plt.xlabel('Models')
    plt.ylabel(metric.title())
    plt.title(f'Model Comparison - {metric.title()}')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# Visualize classification comparison
visualize_model_comparison(clf_comparison, metric='accuracy')

# Visualize regression comparison
visualize_model_comparison(reg_comparison, metric='r2')
```

## Feature Importance

### Feature Importance Analysis

```python
def analyze_feature_importance(X, y, model, feature_names=None):
    """Analyze feature importance for tree-based models."""
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Train model
    model.fit(X, y)
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        print("Model doesn't support feature importance analysis")
        return None
    
    # Create feature importance DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Analysis')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    print("Top 10 Most Important Features:")
    print(feature_importance_df.head(10))
    
    return feature_importance_df

# Analyze feature importance for classification
clf_importance = analyze_feature_importance(X_clf, y_clf, 
                                          RandomForestClassifier(n_estimators=100, random_state=42))

# Analyze feature importance for regression
reg_importance = analyze_feature_importance(X_reg, y_reg, 
                                          RandomForestRegressor(n_estimators=100, random_state=42))
```

## Model Interpretability

### SHAP Values

```python
def analyze_shap_values(X, y, model, feature_names=None):
    """Analyze SHAP values for model interpretability."""
    
    try:
        import shap
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Train model
        model.fit(X, y)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Plot SHAP summary
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.show()
        
        # Plot SHAP dependence plots for top features
        feature_importance = np.abs(shap_values).mean(0)
        top_features = np.argsort(feature_importance)[-3:]  # Top 3 features
        
        for i, feature_idx in enumerate(top_features):
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(feature_idx, shap_values, X, 
                               feature_names=feature_names, show=False)
            plt.title(f'SHAP Dependence Plot - {feature_names[feature_idx]}')
            plt.tight_layout()
            plt.show()
        
        return shap_values
        
    except ImportError:
        print("SHAP library not installed. Install with: pip install shap")
        return None

# Analyze SHAP values for Random Forest
shap_values = analyze_shap_values(X_clf, y_clf, 
                                RandomForestClassifier(n_estimators=100, random_state=42))
```

## Best Practices

### Complete Model Evaluation Pipeline

```python
class ModelEvaluationPipeline:
    """Complete model evaluation pipeline."""
    
    def __init__(self):
        self.results = {}
        self.models = {}
        
    def evaluate_classification(self, X_train, X_test, y_train, y_test, models):
        """Evaluate classification models."""
        
        results = {}
        
        for name, model in models.items():
            print(f"\n{'='*50}")
            print(f"EVALUATING {name.upper()}")
            print(f"{'='*50}")
            
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Basic evaluation
            basic_metrics = evaluate_classification_model(y_test, y_pred, y_prob)
            
            # Advanced metrics
            if y_prob is not None:
                advanced_metrics = advanced_classification_metrics(y_test, y_prob)
                basic_metrics.update(advanced_metrics)
            
            # Cross-validation
            cv_scores = perform_cross_validation(X_train, y_train, model, cv=5, scoring='accuracy')
            basic_metrics['cv_scores'] = cv_scores
            
            # Learning curves
            plot_learning_curves(X_train, y_train, model, f"Learning Curves - {name}")
            
            # Feature importance
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                importance_df = analyze_feature_importance(X_train, y_train, model)
                basic_metrics['feature_importance'] = importance_df
            
            results[name] = basic_metrics
            self.models[name] = model
        
        self.results['classification'] = results
        return results
    
    def evaluate_regression(self, X_train, X_test, y_train, y_test, models):
        """Evaluate regression models."""
        
        results = {}
        
        for name, model in models.items():
            print(f"\n{'='*50}")
            print(f"EVALUATING {name.upper()}")
            print(f"{'='*50}")
            
            # Train model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Basic evaluation
            basic_metrics = evaluate_regression_model(y_test, y_pred)
            
            # Advanced metrics
            advanced_metrics = advanced_regression_metrics(y_test, y_pred)
            basic_metrics.update(advanced_metrics)
            
            # Cross-validation
            cv_scores = perform_cross_validation(X_train, y_train, model, cv=5, scoring='r2')
            basic_metrics['cv_scores'] = cv_scores
            
            # Learning curves
            plot_learning_curves(X_train, y_train, model, f"Learning Curves - {name}")
            
            # Feature importance
            if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
                importance_df = analyze_feature_importance(X_train, y_train, model)
                basic_metrics['feature_importance'] = importance_df
            
            results[name] = basic_metrics
            self.models[name] = model
        
        self.results['regression'] = results
        return results
    
    def generate_report(self):
        """Generate comprehensive evaluation report."""
        
        report = "=== MODEL EVALUATION REPORT ===\n"
        
        if 'classification' in self.results:
            report += "\n--- CLASSIFICATION RESULTS ---\n"
            for model_name, metrics in self.results['classification'].items():
                report += f"\n{model_name}:\n"
                report += f"  Accuracy: {metrics['accuracy']:.4f}\n"
                report += f"  F1-Score: {metrics['f1_score']:.4f}\n"
                report += f"  CV Mean: {metrics['cv_scores'].mean():.4f}\n"
        
        if 'regression' in self.results:
            report += "\n--- REGRESSION RESULTS ---\n"
            for model_name, metrics in self.results['regression'].items():
                report += f"\n{model_name}:\n"
                report += f"  R² Score: {metrics['r2']:.4f}\n"
                report += f"  RMSE: {metrics['rmse']:.4f}\n"
                report += f"  CV Mean: {metrics['cv_scores'].mean():.4f}\n"
        
        return report

# Use the evaluation pipeline
pipeline = ModelEvaluationPipeline()

# Evaluate classification models
clf_models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

clf_results = pipeline.evaluate_classification(X_clf_train, X_clf_test, 
                                             y_clf_train, y_clf_test, clf_models)

# Evaluate regression models
reg_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'SVR': SVR()
}

reg_results = pipeline.evaluate_regression(X_reg_train, X_reg_test, 
                                         y_reg_train, y_reg_test, reg_models)

# Generate final report
final_report = pipeline.generate_report()
print(final_report)
```

## Summary

Model evaluation is essential for building reliable machine learning systems:

- **Classification Metrics**: Accuracy, precision, recall, F1-score, ROC AUC
- **Regression Metrics**: MSE, RMSE, MAE, R², MAPE
- **Cross-Validation**: K-fold, stratified, learning curves
- **Model Comparison**: Systematic comparison of multiple models
- **Feature Importance**: Understanding model decisions
- **Interpretability**: SHAP values and model explanations

Mastering model evaluation will help you build better, more reliable models.

## Next Steps

- Practice evaluation on real-world datasets
- Explore advanced evaluation techniques
- Learn about model deployment evaluation
- Study A/B testing for model comparison

---

**Happy Model Evaluation!** 📊✨ 