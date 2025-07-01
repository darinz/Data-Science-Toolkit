# Python Hyperparameter Tuning Guide

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Optimization-green.svg)](https://optuna.org/)

A comprehensive guide to Hyperparameter Tuning in Python for data science and machine learning applications.

## Table of Contents

1. [Introduction to Hyperparameter Tuning](#introduction-to-hyperparameter-tuning)
2. [Grid Search](#grid-search)
3. [Random Search](#random-search)
4. [Bayesian Optimization](#bayesian-optimization)
5. [Cross-Validation Strategies](#cross-validation-strategies)
6. [Advanced Techniques](#advanced-techniques)
7. [Best Practices](#best-practices)

## Introduction to Hyperparameter Tuning

Hyperparameter tuning is the process of finding the optimal hyperparameters for a machine learning model.

### Why Hyperparameter Tuning Matters

- **Model Performance**: Better hyperparameters lead to better performance
- **Generalization**: Proper tuning prevents overfitting
- **Efficiency**: Optimized models are more efficient
- **Robustness**: Well-tuned models are more reliable

### Basic Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer
import optuna
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
from sklearn.model_selection import train_test_split
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
```

## Grid Search

### Basic Grid Search

```python
def perform_grid_search(X_train, y_train, model, param_grid, cv=5, scoring='accuracy'):
    """Perform grid search for hyperparameter tuning."""
    
    # Create grid search object
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Print results
    print("=== GRID SEARCH RESULTS ===")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Best estimator: {grid_search.best_estimator_}")
    
    # Plot results
    plot_grid_search_results(grid_search, param_grid)
    
    return grid_search

def plot_grid_search_results(grid_search, param_grid):
    """Plot grid search results."""
    
    # Get parameter names
    param_names = list(param_grid.keys())
    
    if len(param_names) == 2:
        # 2D heatmap for 2 parameters
        param1, param2 = param_names
        scores = grid_search.cv_results_['mean_test_score'].reshape(
            len(param_grid[param1]), len(param_grid[param2])
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(scores, annot=True, fmt='.4f', 
                   xticklabels=param_grid[param2], 
                   yticklabels=param_grid[param1])
        plt.xlabel(param2)
        plt.ylabel(param1)
        plt.title('Grid Search Results')
        plt.show()
    
    # Plot individual parameter effects
    for param_name in param_names:
        param_values = param_grid[param_name]
        param_scores = []
        
        for value in param_values:
            # Find scores for this parameter value
            mask = grid_search.cv_results_['param_' + param_name] == value
            scores = grid_search.cv_results_['mean_test_score'][mask]
            param_scores.append(np.mean(scores))
        
        plt.figure(figsize=(8, 6))
        plt.plot(param_values, param_scores, 'bo-')
        plt.xlabel(param_name)
        plt.ylabel('Cross-validation Score')
        plt.title(f'Effect of {param_name} on Performance')
        plt.grid(True)
        plt.show()

# Example: Grid search for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

rf_model = RandomForestClassifier(random_state=42)
rf_grid_search = perform_grid_search(X_clf_train, y_clf_train, rf_model, rf_param_grid)
```

### Grid Search for Different Models

```python
def compare_grid_search_models(X_train, y_train, models_and_params, cv=5):
    """Compare multiple models using grid search."""
    
    results = {}
    
    for model_name, (model, param_grid) in models_and_params.items():
        print(f"\n{'='*50}")
        print(f"GRID SEARCH FOR {model_name.upper()}")
        print(f"{'='*50}")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        results[model_name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best score: {grid_search.best_score_:.4f}")
    
    return results

# Define models and their parameter grids
models_and_params = {
    'Random Forest': (
        RandomForestClassifier(random_state=42),
        {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, None],
            'min_samples_split': [2, 5]
        }
    ),
    'SVM': (
        SVC(random_state=42),
        {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    ),
    'Logistic Regression': (
        LogisticRegression(random_state=42),
        {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    )
}

# Compare models
grid_search_results = compare_grid_search_models(X_clf_train, y_clf_train, models_and_params)
```

## Random Search

### Basic Random Search

```python
def perform_random_search(X_train, y_train, model, param_distributions, n_iter=100, cv=5):
    """Perform random search for hyperparameter tuning."""
    
    # Create random search object
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Fit random search
    random_search.fit(X_train, y_train)
    
    # Print results
    print("=== RANDOM SEARCH RESULTS ===")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")
    print(f"Best estimator: {random_search.best_estimator_}")
    
    # Plot results
    plot_random_search_results(random_search)
    
    return random_search

def plot_random_search_results(random_search):
    """Plot random search results."""
    
    # Plot score distribution
    scores = random_search.cv_results_['mean_test_score']
    
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(random_search.best_score_, color='red', linestyle='--', 
                label=f'Best Score: {random_search.best_score_:.4f}')
    plt.xlabel('Cross-validation Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Random Search Scores')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot score vs iteration
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(scores)), scores, 'bo-', alpha=0.6)
    plt.axhline(random_search.best_score_, color='red', linestyle='--', 
                label=f'Best Score: {random_search.best_score_:.4f}')
    plt.xlabel('Iteration')
    plt.ylabel('Cross-validation Score')
    plt.title('Random Search Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example: Random search for Random Forest
rf_param_distributions = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf_model = RandomForestClassifier(random_state=42)
rf_random_search = perform_random_search(X_clf_train, y_clf_train, rf_model, 
                                       rf_param_distributions, n_iter=50)
```

## Bayesian Optimization

### Using Optuna

```python
def objective_function(trial, X_train, y_train, cv=5):
    """Objective function for Optuna optimization."""
    
    # Define hyperparameters to optimize
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    }
    
    # Create model
    model = RandomForestClassifier(**params, random_state=42)
    
    # Perform cross-validation
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    return scores.mean()

def perform_bayesian_optimization(X_train, y_train, n_trials=100):
    """Perform Bayesian optimization using Optuna."""
    
    # Create study
    study = optuna.create_study(direction='maximize')
    
    # Optimize
    study.optimize(lambda trial: objective_function(trial, X_train, y_train), 
                  n_trials=n_trials)
    
    # Print results
    print("=== BAYESIAN OPTIMIZATION RESULTS ===")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_trial.value:.4f}")
    print(f"Best parameters: {study.best_trial.params}")
    
    # Plot optimization history
    plot_optimization_history(study)
    
    return study

def plot_optimization_history(study):
    """Plot optimization history."""
    
    # Plot optimization history
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title('Optimization History')
    
    plt.subplot(2, 2, 2)
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title('Parameter Importances')
    
    plt.subplot(2, 2, 3)
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.title('Parallel Coordinate Plot')
    
    plt.subplot(2, 2, 4)
    optuna.visualization.matplotlib.plot_contour(study, params=['n_estimators', 'max_depth'])
    plt.title('Contour Plot')
    
    plt.tight_layout()
    plt.show()

# Perform Bayesian optimization
bayesian_study = perform_bayesian_optimization(X_clf_train, y_clf_train, n_trials=50)
```

### Advanced Bayesian Optimization

```python
def advanced_bayesian_optimization(X_train, y_train, model_type='random_forest', n_trials=100):
    """Advanced Bayesian optimization for different model types."""
    
    def objective_rf(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
        model = RandomForestClassifier(**params, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        return scores.mean()
    
    def objective_svm(trial):
        params = {
            'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
            'gamma': trial.suggest_float('gamma', 1e-4, 1e1, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly'])
        }
        model = SVC(**params, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        return scores.mean()
    
    def objective_lr(trial):
        params = {
            'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga'])
        }
        model = LogisticRegression(**params, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        return scores.mean()
    
    # Select objective function
    if model_type == 'random_forest':
        objective = objective_rf
    elif model_type == 'svm':
        objective = objective_svm
    elif model_type == 'logistic_regression':
        objective = objective_lr
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study

# Perform advanced Bayesian optimization
rf_study = advanced_bayesian_optimization(X_clf_train, y_clf_train, 'random_forest', n_trials=50)
svm_study = advanced_bayesian_optimization(X_clf_train, y_clf_train, 'svm', n_trials=50)
```

## Cross-Validation Strategies

### Nested Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

def nested_cross_validation(X, y, model, param_grid, outer_cv=5, inner_cv=3):
    """Perform nested cross-validation for unbiased evaluation."""
    
    outer_scores = []
    best_params_list = []
    
    # Outer cross-validation
    outer_cv_splitter = KFold(n_splits=outer_cv, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv_splitter.split(X), 1):
        print(f"Outer fold {fold}/{outer_cv}")
        
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Inner cross-validation for hyperparameter tuning
        inner_grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='accuracy',
            n_jobs=-1
        )
        
        inner_grid_search.fit(X_train_fold, y_train_fold)
        
        # Evaluate on outer fold
        best_model = inner_grid_search.best_estimator_
        score = best_model.score(X_test_fold, y_test_fold)
        
        outer_scores.append(score)
        best_params_list.append(inner_grid_search.best_params_)
        
        print(f"  Best params: {inner_grid_search.best_params_}")
        print(f"  Score: {score:.4f}")
    
    # Print results
    print(f"\nNested CV Results:")
    print(f"Mean score: {np.mean(outer_scores):.4f} (+/- {np.std(outer_scores) * 2:.4f})")
    print(f"Individual scores: {outer_scores}")
    
    return outer_scores, best_params_list

# Perform nested cross-validation
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

rf_model = RandomForestClassifier(random_state=42)
nested_scores, nested_params = nested_cross_validation(
    X_clf, y_clf, rf_model, rf_param_grid, outer_cv=5, inner_cv=3
)
```

### Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cross_validation(X, y, model, param_grid, n_splits=5):
    """Perform time series cross-validation."""
    
    # Create time series splitter
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Perform grid search with time series CV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print("=== TIME SERIES CV RESULTS ===")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_:.4f}")
    
    return grid_search

# Example with time series data (using our sample data for demonstration)
ts_grid_search = time_series_cross_validation(X_clf, y_clf, rf_model, rf_param_grid)
```

## Advanced Techniques

### Multi-Objective Optimization

```python
def multi_objective_optimization(X_train, y_train, n_trials=100):
    """Multi-objective optimization considering both accuracy and training time."""
    
    def objective(trial):
        import time
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20)
        }
        
        model = RandomForestClassifier(**params, random_state=42)
        
        # Measure training time
        start_time = time.time()
        scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
        training_time = time.time() - start_time
        
        # Return both objectives (accuracy and negative training time)
        return scores.mean(), -training_time
    
    # Create multi-objective study
    study = optuna.create_study(directions=['maximize', 'maximize'])
    study.optimize(objective, n_trials=n_trials)
    
    # Plot Pareto front
    plt.figure(figsize=(10, 6))
    pareto_front = optuna.visualization.matplotlib.plot_pareto_front(study)
    plt.title('Pareto Front: Accuracy vs Training Time')
    plt.xlabel('Accuracy')
    plt.ylabel('Training Time (negative)')
    plt.show()
    
    return study

# Perform multi-objective optimization
multi_obj_study = multi_objective_optimization(X_clf_train, y_clf_train, n_trials=50)
```

### Ensemble Hyperparameter Tuning

```python
from sklearn.ensemble import VotingClassifier

def ensemble_hyperparameter_tuning(X_train, y_train, X_test, y_test):
    """Tune hyperparameters for ensemble methods."""
    
    # Define base models with their parameter grids
    base_models = {
        'rf': (RandomForestClassifier(random_state=42), {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, None]
        }),
        'svm': (SVC(probability=True, random_state=42), {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear']
        }),
        'lr': (LogisticRegression(random_state=42), {
            'C': [0.1, 1, 10],
            'penalty': ['l1', 'l2']
        })
    }
    
    # Tune each base model
    tuned_models = {}
    for name, (model, param_grid) in base_models.items():
        print(f"Tuning {name}...")
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        tuned_models[name] = grid_search.best_estimator_
        print(f"Best {name} score: {grid_search.best_score_:.4f}")
    
    # Create ensemble with different voting strategies
    ensemble_results = {}
    
    for voting in ['hard', 'soft']:
        ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in tuned_models.items()],
            voting=voting
        )
        
        ensemble.fit(X_train, y_train)
        score = ensemble.score(X_test, y_test)
        ensemble_results[voting] = score
        print(f"Ensemble ({voting} voting) score: {score:.4f}")
    
    return tuned_models, ensemble_results

# Perform ensemble hyperparameter tuning
tuned_models, ensemble_results = ensemble_hyperparameter_tuning(
    X_clf_train, y_clf_train, X_clf_test, y_clf_test
)
```

## Best Practices

### Complete Hyperparameter Tuning Pipeline

```python
class HyperparameterTuningPipeline:
    """Complete hyperparameter tuning pipeline."""
    
    def __init__(self):
        self.best_models = {}
        self.tuning_results = {}
        
    def tune_classification_models(self, X_train, y_train, X_test, y_test):
        """Tune multiple classification models."""
        
        # Define models and their parameter spaces
        models_config = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'param_grid': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'SVM': {
                'model': SVC(random_state=42),
                'param_grid': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42),
                'param_grid': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
        }
        
        results = {}
        
        for model_name, config in models_config.items():
            print(f"\n{'='*50}")
            print(f"TUNING {model_name.upper()}")
            print(f"{'='*50}")
            
            # Grid search
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['param_grid'],
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Evaluate on test set
            test_score = grid_search.score(X_test, y_test)
            
            results[model_name] = {
                'best_params': grid_search.best_params_,
                'cv_score': grid_search.best_score_,
                'test_score': test_score,
                'best_model': grid_search.best_estimator_
            }
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"CV Score: {grid_search.best_score_:.4f}")
            print(f"Test Score: {test_score:.4f}")
        
        self.tuning_results['classification'] = results
        return results
    
    def tune_regression_models(self, X_train, y_train, X_test, y_test):
        """Tune multiple regression models."""
        
        # Define models and their parameter spaces
        models_config = {
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'param_grid': {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'Ridge': {
                'model': Ridge(),
                'param_grid': {
                    'alpha': [0.1, 1, 10, 100, 1000]
                }
            },
            'Lasso': {
                'model': Lasso(),
                'param_grid': {
                    'alpha': [0.1, 1, 10, 100, 1000]
                }
            }
        }
        
        results = {}
        
        for model_name, config in models_config.items():
            print(f"\n{'='*50}")
            print(f"TUNING {model_name.upper()}")
            print(f"{'='*50}")
            
            # Grid search
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['param_grid'],
                cv=5,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Evaluate on test set
            test_score = grid_search.score(X_test, y_test)
            
            results[model_name] = {
                'best_params': grid_search.best_params_,
                'cv_score': grid_search.best_score_,
                'test_score': test_score,
                'best_model': grid_search.best_estimator_
            }
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"CV Score: {grid_search.best_score_:.4f}")
            print(f"Test Score: {test_score:.4f}")
        
        self.tuning_results['regression'] = results
        return results
    
    def generate_report(self):
        """Generate comprehensive tuning report."""
        
        report = "=== HYPERPARAMETER TUNING REPORT ===\n"
        
        for task, results in self.tuning_results.items():
            report += f"\n--- {task.upper()} RESULTS ---\n"
            
            for model_name, result in results.items():
                report += f"\n{model_name}:\n"
                report += f"  Best Parameters: {result['best_params']}\n"
                report += f"  CV Score: {result['cv_score']:.4f}\n"
                report += f"  Test Score: {result['test_score']:.4f}\n"
        
        return report

# Use the tuning pipeline
tuning_pipeline = HyperparameterTuningPipeline()

# Tune classification models
clf_tuning_results = tuning_pipeline.tune_classification_models(
    X_clf_train, y_clf_train, X_clf_test, y_clf_test
)

# Tune regression models
reg_tuning_results = tuning_pipeline.tune_regression_models(
    X_reg_train, y_reg_train, X_reg_test, y_reg_test
)

# Generate final report
final_report = tuning_pipeline.generate_report()
print(final_report)
```

## Summary

Hyperparameter tuning is essential for optimal model performance:

- **Grid Search**: Systematic exploration of parameter space
- **Random Search**: Efficient exploration for high-dimensional spaces
- **Bayesian Optimization**: Intelligent search using probabilistic models
- **Cross-Validation**: Proper evaluation strategies
- **Advanced Techniques**: Multi-objective optimization and ensemble tuning
- **Best Practices**: Systematic approach and comprehensive evaluation

Mastering hyperparameter tuning will significantly improve your model performance.

## Next Steps

- Practice tuning on real-world datasets
- Explore advanced optimization libraries
- Learn about automated hyperparameter tuning
- Study model selection strategies

---

**Happy Hyperparameter Tuning!** ⚙️✨ 