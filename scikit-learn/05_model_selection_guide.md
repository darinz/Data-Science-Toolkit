# Model Selection with scikit-learn

A comprehensive guide to model selection: cross-validation, hyperparameter tuning, model comparison, and best practices.

## Table of Contents

1. [Introduction](#introduction)
2. [Cross-Validation](#cross-validation)
3. [Hyperparameter Tuning](#hyperparameter-tuning)
4. [Model Comparison](#model-comparison)
5. [Ensemble Methods](#ensemble-methods)
6. [Best Practices](#best-practices)

## Introduction

Model selection is the process of choosing the best model and its parameters for a given dataset and task.

## Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, LeaveOneOut
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X, y = iris.data, iris.target
model = LogisticRegression(max_iter=1000)

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)
print('5-Fold CV Scores:', scores)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5)
skf_scores = cross_val_score(model, X, y, cv=skf)
print('Stratified K-Fold Scores:', skf_scores)

# Leave-One-Out
loo = LeaveOneOut()
loo_scores = cross_val_score(model, X, y, cv=loo)
print('Leave-One-Out CV Mean Score:', loo_scores.mean())
```

## Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import randint

param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)
print('Best Params (Grid Search):', grid_search.best_params_)

param_dist = {'n_estimators': randint(50, 200), 'max_depth': [None, 10, 20]}
random_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=10, cv=5, random_state=42)
random_search.fit(X, y)
print('Best Params (Random Search):', random_search.best_params_)
```

## Model Comparison

```python
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'{name}: Accuracy = {acc:.4f}')
```

## Ensemble Methods

```python
from sklearn.ensemble import VotingClassifier

voting = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier(random_state=42)),
    ('svc', SVC(probability=True))
], voting='soft')
voting.fit(X_train, y_train)
print('Voting Classifier Accuracy:', voting.score(X_test, y_test))
```

## Best Practices

- Use cross-validation for reliable model evaluation.
- Tune hyperparameters with grid/random search.
- Compare multiple models before final selection.
- Use ensemble methods for improved performance.
- Avoid data leakage by splitting data before preprocessing. 