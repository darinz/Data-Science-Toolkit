# Supervised Learning with scikit-learn

A comprehensive guide to supervised learning algorithms including classification and regression techniques.

## Table of Contents

1. [Classification Algorithms](#classification-algorithms)
2. [Regression Algorithms](#regression-algorithms)
3. [Model Evaluation](#model-evaluation)
4. [Hyperparameter Tuning](#hyperparameter-tuning)
5. [Ensemble Methods](#ensemble-methods)
6. [Feature Selection](#feature-selection)

## Classification Algorithms

### 1. Logistic Regression

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, model_selection, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# Load datasets
iris = datasets.load_iris()
breast_cancer = datasets.load_breast_cancer()

def logistic_regression_example():
    """Demonstrate logistic regression for binary classification"""
    
    # Use breast cancer dataset (binary classification)
    X = breast_cancer.data
    y = breast_cancer.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = lr.predict(X_test_scaled)
    y_pred_proba = lr.predict_proba(X_test_scaled)
    
    # Evaluate
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)
    
    print("Logistic Regression Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'feature': breast_cancer.feature_names,
        'coefficient': np.abs(lr.coef_[0])
    }).sort_values('coefficient', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return lr, scaler, y_pred, y_pred_proba

lr_model, lr_scaler, lr_pred, lr_proba = logistic_regression_example()
```

### 2. Support Vector Machine (SVM)

```python
from sklearn.svm import SVC, LinearSVC

def svm_example():
    """Demonstrate SVM for classification"""
    
    # Use iris dataset for multiclass classification
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train different SVM models
    svm_models = {
        'Linear SVM': SVC(kernel='linear', random_state=42),
        'RBF SVM': SVC(kernel='rbf', random_state=42),
        'Polynomial SVM': SVC(kernel='poly', degree=3, random_state=42)
    }
    
    results = {}
    
    for name, model in svm_models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        accuracy = metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test, y_pred, average='weighted')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'predictions': y_pred
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    
    return results

svm_results = svm_example()
```

### 3. Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

def random_forest_example():
    """Demonstrate Random Forest for classification"""
    
    # Use iris dataset
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)
    
    # Evaluate
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    
    print("Random Forest Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': iris.feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    print(feature_importance)
    
    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()
    
    return rf, y_pred, y_pred_proba

rf_model, rf_pred, rf_proba = random_forest_example()
```

### 4. K-Nearest Neighbors

```python
from sklearn.neighbors import KNeighborsClassifier

def knn_example():
    """Demonstrate K-Nearest Neighbors for classification"""
    
    # Use iris dataset
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features (important for KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test different k values
    k_values = [1, 3, 5, 7, 9, 11, 13, 15]
    accuracies = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        accuracy = knn.score(X_test_scaled, y_test)
        accuracies.append(accuracy)
        print(f"k={k}: Accuracy={accuracy:.4f}")
    
    # Find optimal k
    optimal_k = k_values[np.argmax(accuracies)]
    print(f"\nOptimal k: {optimal_k}")
    
    # Train with optimal k
    optimal_knn = KNeighborsClassifier(n_neighbors=optimal_k)
    optimal_knn.fit(X_train_scaled, y_train)
    
    y_pred = optimal_knn.predict(X_test_scaled)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    
    print(f"Optimal KNN Accuracy: {accuracy:.4f}")
    
    # Plot k vs accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('k (Number of Neighbors)')
    plt.ylabel('Accuracy')
    plt.title('K-Nearest Neighbors: k vs Accuracy')
    plt.grid(True, alpha=0.3)
    plt.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
    plt.legend()
    plt.show()
    
    return optimal_knn, scaler, y_pred

knn_model, knn_scaler, knn_pred = knn_example()
```

## Regression Algorithms

### 1. Linear Regression

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score

def linear_regression_example():
    """Demonstrate linear regression"""
    
    # Load boston housing dataset
    boston = load_boston()
    X = boston.data
    y = boston.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train linear regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = lr.predict(X_test_scaled)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("Linear Regression Results:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance (coefficients)
    feature_importance = pd.DataFrame({
        'feature': boston.feature_names,
        'coefficient': np.abs(lr.coef_)
    }).sort_values('coefficient', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return lr, scaler, y_pred

lr_reg_model, lr_reg_scaler, lr_reg_pred = linear_regression_example()
```

### 2. Ridge and Lasso Regression

```python
def ridge_lasso_example():
    """Demonstrate Ridge and Lasso regression"""
    
    # Load boston housing dataset
    boston = load_boston()
    X = boston.data
    y = boston.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train different models
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }
        
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R² Score: {r2:.4f}")
    
    return results

regression_results = ridge_lasso_example()
```

### 3. Random Forest Regression

```python
from sklearn.ensemble import RandomForestRegressor

def random_forest_regression_example():
    """Demonstrate Random Forest for regression"""
    
    # Load boston housing dataset
    boston = load_boston()
    X = boston.data
    y = boston.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print("Random Forest Regression Results:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': boston.feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    print(feature_importance)
    
    # Visualize predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Random Forest Regression: Predicted vs Actual')
    plt.tight_layout()
    plt.show()
    
    return rf, y_pred

rf_reg_model, rf_reg_pred = random_forest_regression_example()
```

## Model Evaluation

### Classification Metrics

```python
def classification_evaluation(y_true, y_pred, y_pred_proba=None):
    """Comprehensive classification evaluation"""
    
    # Basic metrics
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    f1 = metrics.f1_score(y_true, y_pred, average='weighted')
    
    print("Classification Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 5))
    
    # Confusion matrix plot
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Classification report
    plt.subplot(1, 2, 2)
    report = metrics.classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    
    # Create a simple text plot for classification report
    plt.text(0.1, 0.9, 'Classification Report:', fontsize=12, fontweight='bold')
    plt.text(0.1, 0.8, f'Accuracy: {accuracy:.4f}', fontsize=10)
    plt.text(0.1, 0.7, f'Precision: {precision:.4f}', fontsize=10)
    plt.text(0.1, 0.6, f'Recall: {recall:.4f}', fontsize=10)
    plt.text(0.1, 0.5, f'F1-Score: {f1:.4f}', fontsize=10)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

# Example evaluation
classification_metrics = classification_evaluation(y_test, rf_pred)
```

### Regression Metrics

```python
def regression_evaluation(y_true, y_pred):
    """Comprehensive regression evaluation"""
    
    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("Regression Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Residuals plot
    residuals = y_true - y_pred
    
    plt.figure(figsize=(15, 5))
    
    # Predicted vs Actual
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual')
    
    # Residuals vs Predicted
    plt.subplot(1, 3, 2)
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    
    # Residuals histogram
    plt.subplot(1, 3, 3)
    plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'residuals': residuals
    }

# Example evaluation
regression_metrics = regression_evaluation(y_test, rf_reg_pred)
```

## Hyperparameter Tuning

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

def grid_search_example():
    """Demonstrate grid search for hyperparameter tuning"""
    
    # Use iris dataset
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grid for Random Forest
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create base model
    base_model = RandomForestClassifier(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best parameters and score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    test_accuracy = best_model.score(X_test_scaled, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Results summary
    results_df = pd.DataFrame(grid_search.cv_results_)
    print(f"\nTop 5 parameter combinations:")
    print(results_df[['params', 'mean_test_score']].sort_values('mean_test_score', ascending=False).head())
    
    return grid_search, best_model

grid_search_result, best_model = grid_search_example()
```

### Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def random_search_example():
    """Demonstrate random search for hyperparameter tuning"""
    
    # Use iris dataset
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter distributions for Random Forest
    param_distributions = {
        'n_estimators': randint(50, 300),
        'max_depth': [None] + list(range(5, 50)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Create base model
    base_model = RandomForestClassifier(random_state=42)
    
    # Perform random search
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=50,  # Number of parameter settings sampled
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Fit random search
    random_search.fit(X_train_scaled, y_train)
    
    # Get best parameters and score
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    
    # Evaluate on test set
    best_model = random_search.best_estimator_
    test_accuracy = best_model.score(X_test_scaled, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    return random_search, best_model

random_search_result, best_model_random = random_search_example()
```

## Ensemble Methods

### 1. Voting Classifier

```python
from sklearn.ensemble import VotingClassifier

def voting_classifier_example():
    """Demonstrate voting classifier"""
    
    # Use iris dataset
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create base models
    models = [
        ('lr', LogisticRegression(random_state=42, max_iter=1000)),
        ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
        ('svm', SVC(random_state=42, probability=True))
    ]
    
    # Create voting classifier
    voting_clf = VotingClassifier(estimators=models, voting='soft')
    
    # Train voting classifier
    voting_clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = voting_clf.predict(X_test_scaled)
    
    # Evaluate
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    
    print("Voting Classifier Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Compare with individual models
    print(f"\nIndividual Model Results:")
    for name, model in models:
        model.fit(X_train_scaled, y_train)
        y_pred_individual = model.predict(X_test_scaled)
        accuracy_individual = metrics.accuracy_score(y_test, y_pred_individual)
        print(f"{name}: Accuracy = {accuracy_individual:.4f}")
    
    return voting_clf, y_pred

voting_model, voting_pred = voting_classifier_example()
```

### 2. Bagging

```python
from sklearn.ensemble import BaggingClassifier

def bagging_example():
    """Demonstrate bagging classifier"""
    
    # Use iris dataset
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create base estimator
    base_estimator = DecisionTreeClassifier(random_state=42)
    
    # Create bagging classifier
    bagging_clf = BaggingClassifier(
        base_estimator=base_estimator,
        n_estimators=10,
        random_state=42
    )
    
    # Train bagging classifier
    bagging_clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = bagging_clf.predict(X_test_scaled)
    
    # Evaluate
    accuracy = metrics.accuracy_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred, average='weighted')
    
    print("Bagging Classifier Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Compare with base estimator
    base_estimator.fit(X_train_scaled, y_train)
    y_pred_base = base_estimator.predict(X_test_scaled)
    accuracy_base = metrics.accuracy_score(y_test, y_pred_base)
    
    print(f"\nBase Estimator (Decision Tree) Accuracy: {accuracy_base:.4f}")
    print(f"Improvement: {accuracy - accuracy_base:.4f}")
    
    return bagging_clf, y_pred

bagging_model, bagging_pred = bagging_example()
```

## Feature Selection

### 1. Univariate Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

def univariate_feature_selection():
    """Demonstrate univariate feature selection"""
    
    # Use iris dataset for classification
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Select top k features
    k_values = [1, 2, 3, 4]
    results = {}
    
    for k in k_values:
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Train model
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        model.fit(X_train_selected, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_selected)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        
        results[k] = {
            'accuracy': accuracy,
            'selected_features': selector.get_support(),
            'feature_scores': selector.scores_
        }
        
        print(f"k={k}: Accuracy={accuracy:.4f}")
    
    # Show feature scores
    feature_scores = pd.DataFrame({
        'feature': iris.feature_names,
        'score': results[4]['feature_scores']
    }).sort_values('score', ascending=False)
    
    print(f"\nFeature Selection Scores:")
    print(feature_scores)
    
    return results

feature_selection_results = univariate_feature_selection()
```

### 2. Recursive Feature Elimination

```python
from sklearn.feature_selection import RFE

def recursive_feature_elimination():
    """Demonstrate recursive feature elimination"""
    
    # Use iris dataset
    X = iris.data
    y = iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create base estimator
    estimator = RandomForestClassifier(random_state=42, n_estimators=100)
    
    # Perform RFE
    rfe = RFE(estimator=estimator, n_features_to_select=2)
    rfe.fit(X_train_scaled, y_train)
    
    # Get selected features
    selected_features = rfe.get_support()
    feature_ranking = rfe.ranking_
    
    print("Recursive Feature Elimination Results:")
    print(f"Selected features: {selected_features}")
    print(f"Feature ranking: {feature_ranking}")
    
    # Show which features were selected
    selected_feature_names = [iris.feature_names[i] for i in range(len(iris.feature_names)) if selected_features[i]]
    print(f"Selected feature names: {selected_feature_names}")
    
    # Transform data
    X_train_rfe = rfe.transform(X_train_scaled)
    X_test_rfe = rfe.transform(X_test_scaled)
    
    # Train model with selected features
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train_rfe, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_rfe)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    
    print(f"Accuracy with selected features: {accuracy:.4f}")
    
    return rfe, selected_features, accuracy

rfe_result, selected_features, rfe_accuracy = recursive_feature_elimination()
```

## Summary

### Key Concepts Covered:

1. **Classification Algorithms**: Logistic Regression, SVM, Random Forest, KNN
2. **Regression Algorithms**: Linear Regression, Ridge, Lasso, Random Forest
3. **Model Evaluation**: Comprehensive metrics for both classification and regression
4. **Hyperparameter Tuning**: Grid search and random search techniques
5. **Ensemble Methods**: Voting classifiers and bagging
6. **Feature Selection**: Univariate selection and recursive feature elimination

### Best Practices:

1. **Always scale features** for algorithms sensitive to scale (SVM, KNN, Neural Networks)
2. **Use cross-validation** for reliable model evaluation
3. **Compare multiple algorithms** before choosing the best one
4. **Tune hyperparameters** systematically using grid or random search
5. **Consider ensemble methods** for improved performance
6. **Select relevant features** to reduce complexity and improve performance

### Next Steps:

- Explore advanced ensemble methods (Boosting, Stacking)
- Learn about neural networks and deep learning
- Practice with real-world datasets
- Master model deployment and production considerations 