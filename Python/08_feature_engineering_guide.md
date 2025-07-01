# Python Feature Engineering Guide

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-blue.svg)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)

A comprehensive guide to Feature Engineering in Python for data science and machine learning applications.

## Table of Contents

1. [Introduction to Feature Engineering](#introduction-to-feature-engineering)
2. [Feature Creation](#feature-creation)
3. [Feature Transformation](#feature-transformation)
4. [Feature Selection](#feature-selection)
5. [Categorical Encoding](#categorical-encoding)
6. [Time Series Features](#time-series-features)
7. [Text Feature Engineering](#text-feature-engineering)
8. [Domain-Specific Features](#domain-specific-features)
9. [Feature Scaling](#feature-scaling)
10. [Best Practices](#best-practices)

## Introduction to Feature Engineering

Feature engineering is the process of creating new features or modifying existing ones to improve model performance.

### Why Feature Engineering Matters

- **Model Performance**: Better features lead to better predictions
- **Interpretability**: Engineered features can be more meaningful
- **Data Efficiency**: Reduces need for more data
- **Domain Knowledge**: Incorporates business understanding

### Basic Setup

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Sample dataset
np.random.seed(42)
data = {
    'age': np.random.normal(35, 10, 1000),
    'salary': np.random.normal(60000, 15000, 1000),
    'experience': np.random.normal(8, 3, 1000),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
    'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing'], 1000),
    'join_date': pd.date_range('2015-01-01', periods=1000, freq='D'),
    'satisfaction': np.random.uniform(1, 10, 1000)
}

df = pd.DataFrame(data)
```

## Feature Creation

### Mathematical Transformations

```python
def create_mathematical_features(df):
    """Create mathematical transformations of numerical features."""
    
    df_new = df.copy()
    
    # Square and cube transformations
    df_new['age_squared'] = df['age'] ** 2
    df_new['age_cubed'] = df['age'] ** 3
    df_new['salary_squared'] = df['salary'] ** 2
    
    # Square root transformations
    df_new['age_sqrt'] = np.sqrt(np.abs(df['age']))
    df_new['salary_sqrt'] = np.sqrt(df['salary'])
    
    # Logarithmic transformations
    df_new['salary_log'] = np.log1p(df['salary'])
    df_new['experience_log'] = np.log1p(df['experience'])
    
    # Reciprocal transformations
    df_new['age_reciprocal'] = 1 / (df['age'] + 1)  # Add 1 to avoid division by zero
    
    # Ratio features
    df_new['salary_per_year'] = df['salary'] / (df['age'] - 18)  # Assuming work starts at 18
    df_new['experience_per_year'] = df['experience'] / (df['age'] - 18)
    
    return df_new

# Create mathematical features
df_math = create_mathematical_features(df)
print("Mathematical features created:")
print(df_math.columns.tolist())
```

### Interaction Features

```python
def create_interaction_features(df):
    """Create interaction features between variables."""
    
    df_new = df.copy()
    
    # Two-way interactions
    df_new['age_experience'] = df['age'] * df['experience']
    df_new['age_salary'] = df['age'] * df['salary']
    df_new['experience_salary'] = df['experience'] * df['salary']
    
    # Polynomial interactions
    df_new['age_experience_squared'] = df['age'] * (df['experience'] ** 2)
    df_new['age_squared_experience'] = (df['age'] ** 2) * df['experience']
    
    # Ratio interactions
    df_new['salary_experience_ratio'] = df['salary'] / (df['experience'] + 1)
    df_new['age_experience_ratio'] = df['age'] / (df['experience'] + 1)
    
    return df_new

# Create interaction features
df_interactions = create_interaction_features(df)
print("Interaction features created:")
print(df_interactions.columns.tolist())
```

### Binning and Discretization

```python
def create_binned_features(df):
    """Create binned features for numerical variables."""
    
    df_new = df.copy()
    
    # Age bins
    df_new['age_bins'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 100], 
                               labels=['Young', 'Early Career', 'Mid Career', 'Senior'])
    
    # Salary bins
    df_new['salary_bins'] = pd.cut(df['salary'], bins=5, labels=['Low', 'Below Avg', 'Average', 'Above Avg', 'High'])
    
    # Experience bins
    df_new['experience_bins'] = pd.cut(df['experience'], bins=[0, 3, 7, 12, 50], 
                                      labels=['Junior', 'Mid', 'Senior', 'Expert'])
    
    # Satisfaction bins
    df_new['satisfaction_bins'] = pd.cut(df['satisfaction'], bins=[0, 4, 7, 10], 
                                        labels=['Low', 'Medium', 'High'])
    
    return df_new

# Create binned features
df_binned = create_binned_features(df)
print("Binned features created:")
print(df_binned[['age_bins', 'salary_bins', 'experience_bins', 'satisfaction_bins']].head())
```

## Feature Transformation

### Scaling and Normalization

```python
def scale_features(df, method='standard'):
    """Scale numerical features using different methods."""
    
    df_scaled = df.copy()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == 'standard':
        scaler = StandardScaler()
        df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
    elif method == 'minmax':
        scaler = MinMaxScaler()
        df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
    elif method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
    elif method == 'log':
        # Log transformation for positive values
        for col in numerical_cols:
            if (df[col] > 0).all():
                df_scaled[col] = np.log1p(df[col])
    
    return df_scaled

# Scale features
df_scaled = scale_features(df, method='standard')
print("Features scaled using StandardScaler")
```

### Power Transformations

```python
from sklearn.preprocessing import PowerTransformer

def apply_power_transformations(df):
    """Apply power transformations to numerical features."""
    
    df_transformed = df.copy()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Yeo-Johnson transformation
    pt = PowerTransformer(method='yeo-johnson')
    df_transformed[numerical_cols] = pt.fit_transform(df[numerical_cols])
    
    return df_transformed

# Apply power transformations
df_power = apply_power_transformations(df)
print("Power transformations applied")
```

## Feature Selection

### Statistical Feature Selection

```python
def select_features_statistical(df, target_col, k=10):
    """Select features using statistical methods."""
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Remove non-numeric columns for statistical selection
    X_numeric = X.select_dtypes(include=[np.number])
    
    # F-regression for regression problems
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X_numeric, y)
    
    # Get selected feature names
    selected_features = X_numeric.columns[selector.get_support()].tolist()
    feature_scores = selector.scores_[selector.get_support()]
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'Feature': selected_features,
        'Score': feature_scores
    }).sort_values('Score', ascending=False)
    
    return feature_importance, X_selected

# Select features (assuming 'satisfaction' as target)
feature_importance, X_selected = select_features_statistical(df, 'satisfaction', k=5)
print("Top features by statistical selection:")
print(feature_importance)
```

### Correlation-Based Selection

```python
def select_features_correlation(df, threshold=0.8):
    """Remove highly correlated features."""
    
    # Calculate correlation matrix
    correlation_matrix = df.corr().abs()
    
    # Find highly correlated features
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to drop
    to_drop = [column for column in upper_triangle.columns 
               if any(upper_triangle[column] > threshold)]
    
    # Drop highly correlated features
    df_selected = df.drop(columns=to_drop)
    
    return df_selected, to_drop

# Remove correlated features
df_uncorrelated, dropped_features = select_features_correlation(df, threshold=0.8)
print(f"Features dropped due to high correlation: {dropped_features}")
```

### PCA for Feature Reduction

```python
def reduce_features_pca(df, n_components=0.95):
    """Reduce features using Principal Component Analysis."""
    
    # Select numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[numerical_cols])
    
    # Apply PCA
    if n_components < 1:
        pca = PCA(n_components=n_components)
    else:
        pca = PCA(n_components=int(n_components))
    
    X_pca = pca.fit_transform(X_scaled)
    
    # Create DataFrame with PCA components
    pca_df = pd.DataFrame(
        X_pca,
        columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
    )
    
    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"Number of components: {X_pca.shape[1]}")
    print(f"Explained variance: {cumulative_variance[-1]:.3f}")
    
    return pca_df, pca

# Reduce features using PCA
pca_features, pca_model = reduce_features_pca(df, n_components=0.95)
print("PCA features created:")
print(pca_features.head())
```

## Categorical Encoding

### Label Encoding

```python
def encode_categorical_features(df, method='label'):
    """Encode categorical features using different methods."""
    
    df_encoded = df.copy()
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if method == 'label':
        # Label encoding
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col])
            
    elif method == 'onehot':
        # One-hot encoding
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
    elif method == 'target':
        # Target encoding (for supervised learning)
        from sklearn.model_selection import KFold
        from category_encoders import TargetEncoder
        
        te = TargetEncoder(cols=categorical_cols.tolist())
        df_encoded[categorical_cols] = te.fit_transform(df[categorical_cols], df['satisfaction'])
        
    elif method == 'frequency':
        # Frequency encoding
        for col in categorical_cols:
            frequency_map = df[col].value_counts(normalize=True).to_dict()
            df_encoded[col] = df[col].map(frequency_map)
    
    return df_encoded

# Encode categorical features
df_encoded = encode_categorical_features(df, method='onehot')
print("Categorical features encoded using one-hot encoding")
print(f"Shape after encoding: {df_encoded.shape}")
```

## Time Series Features

### Date and Time Features

```python
def create_time_features(df, date_column='join_date'):
    """Create time-based features from date columns."""
    
    df_time = df.copy()
    
    # Extract date components
    df_time[f'{date_column}_year'] = df[date_column].dt.year
    df_time[f'{date_column}_month'] = df[date_column].dt.month
    df_time[f'{date_column}_day'] = df[date_column].dt.day
    df_time[f'{date_column}_dayofweek'] = df[date_column].dt.dayofweek
    df_time[f'{date_column}_quarter'] = df[date_column].dt.quarter
    
    # Cyclical encoding for periodic features
    df_time[f'{date_column}_month_sin'] = np.sin(2 * np.pi * df[date_column].dt.month / 12)
    df_time[f'{date_column}_month_cos'] = np.cos(2 * np.pi * df[date_column].dt.month / 12)
    df_time[f'{date_column}_dayofweek_sin'] = np.sin(2 * np.pi * df[date_column].dt.dayofweek / 7)
    df_time[f'{date_column}_dayofweek_cos'] = np.cos(2 * np.pi * df[date_column].dt.dayofweek / 7)
    
    # Time-based features
    df_time[f'{date_column}_is_weekend'] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)
    df_time[f'{date_column}_is_month_start'] = df[date_column].dt.is_month_start.astype(int)
    df_time[f'{date_column}_is_month_end'] = df[date_column].dt.is_month_end.astype(int)
    
    # Days since reference date
    reference_date = df[date_column].min()
    df_time[f'{date_column}_days_since_start'] = (df[date_column] - reference_date).dt.days
    
    return df_time

# Create time features
df_time_features = create_time_features(df)
print("Time-based features created:")
print(df_time_features.columns.tolist())
```

## Text Feature Engineering

### Basic Text Features

```python
def create_text_features(df, text_column):
    """Create features from text data."""
    
    df_text = df.copy()
    
    # Length features
    df_text[f'{text_column}_length'] = df[text_column].str.len()
    df_text[f'{text_column}_word_count'] = df[text_column].str.split().str.len()
    df_text[f'{text_column}_char_count'] = df[text_column].str.replace(' ', '').str.len()
    
    # Statistical features
    df_text[f'{text_column}_avg_word_length'] = df[text_column].str.split().apply(
        lambda x: np.mean([len(word) for word in x]) if x else 0
    )
    
    # Special character features
    df_text[f'{text_column}_special_char_count'] = df[text_column].str.count(r'[^a-zA-Z0-9\s]')
    df_text[f'{text_column}_digit_count'] = df[text_column].str.count(r'\d')
    df_text[f'{text_column}_uppercase_count'] = df[text_column].str.count(r'[A-Z]')
    
    # Sentiment features (if text is available)
    try:
        from textblob import TextBlob
        df_text[f'{text_column}_sentiment'] = df[text_column].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
    except ImportError:
        print("TextBlob not available for sentiment analysis")
    
    return df_text

# Example with department names as text
df_text_features = create_text_features(df, 'department')
print("Text features created:")
print(df_text_features.columns.tolist())
```

## Domain-Specific Features

### Business Logic Features

```python
def create_business_features(df):
    """Create domain-specific business features."""
    
    df_business = df.copy()
    
    # Salary-related features
    df_business['salary_percentile'] = df['salary'].rank(pct=True)
    df_business['salary_ratio_to_mean'] = df['salary'] / df['salary'].mean()
    df_business['salary_ratio_to_median'] = df['salary'] / df['salary'].median()
    
    # Experience-related features
    df_business['experience_percentile'] = df['experience'].rank(pct=True)
    df_business['years_per_education'] = df['experience'] / (df['age'] - 18)
    
    # Department-specific features
    dept_salary_means = df.groupby('department')['salary'].mean()
    df_business['salary_vs_dept_mean'] = df['salary'] / df['department'].map(dept_salary_means)
    
    # Education level encoding
    education_levels = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    df_business['education_level'] = df['education'].map(education_levels)
    
    # Seniority features
    df_business['is_senior'] = ((df['age'] > 40) & (df['experience'] > 10)).astype(int)
    df_business['is_junior'] = ((df['age'] < 30) & (df['experience'] < 5)).astype(int)
    
    return df_business

# Create business features
df_business = create_business_features(df)
print("Business-specific features created:")
print(df_business.columns.tolist())
```

## Feature Scaling

### Advanced Scaling Techniques

```python
def advanced_feature_scaling(df, method='robust'):
    """Apply advanced scaling techniques."""
    
    df_scaled = df.copy()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if method == 'robust':
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        
    elif method == 'quantile':
        from sklearn.preprocessing import QuantileTransformer
        scaler = QuantileTransformer(output_distribution='normal')
        
    elif method == 'maxabs':
        from sklearn.preprocessing import MaxAbsScaler
        scaler = MaxAbsScaler()
        
    elif method == 'yeojohnson':
        from sklearn.preprocessing import PowerTransformer
        scaler = PowerTransformer(method='yeo-johnson')
    
    df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df_scaled, scaler

# Apply advanced scaling
df_advanced_scaled, scaler = advanced_feature_scaling(df, method='robust')
print("Advanced scaling applied")
```

## Best Practices

### Feature Engineering Pipeline

```python
class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline."""
    
    def __init__(self):
        self.transformers = {}
        self.feature_names = []
        
    def fit_transform(self, df, target_col=None):
        """Apply complete feature engineering pipeline."""
        
        df_engineered = df.copy()
        
        # 1. Handle missing values
        df_engineered = self._handle_missing_values(df_engineered)
        
        # 2. Create mathematical features
        df_engineered = self._create_mathematical_features(df_engineered)
        
        # 3. Create interaction features
        df_engineered = self._create_interaction_features(df_engineered)
        
        # 4. Encode categorical features
        df_engineered = self._encode_categorical_features(df_engineered)
        
        # 5. Scale features
        df_engineered = self._scale_features(df_engineered)
        
        # 6. Select features
        if target_col:
            df_engineered = self._select_features(df_engineered, target_col)
        
        self.feature_names = df_engineered.columns.tolist()
        
        return df_engineered
    
    def _handle_missing_values(self, df):
        """Handle missing values."""
        # Fill numeric with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        # Fill categorical with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        
        return df
    
    def _create_mathematical_features(self, df):
        """Create mathematical features."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_log'] = np.log1p(np.abs(df[col]))
        
        return df
    
    def _create_interaction_features(self, df):
        """Create interaction features."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for i, col1 in enumerate(numerical_cols):
            for col2 in numerical_cols[i+1:]:
                df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
        
        return df
    
    def _encode_categorical_features(self, df):
        """Encode categorical features."""
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.transformers[f'label_encoder_{col}'] = le
        
        return df
    
    def _scale_features(self, df):
        """Scale features."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        self.transformers['scaler'] = scaler
        
        return df
    
    def _select_features(self, df, target_col):
        """Select features."""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        selector = SelectKBest(score_func=f_regression, k=min(20, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        df_selected = df[selected_features + [target_col]]
        
        self.transformers['feature_selector'] = selector
        
        return df_selected

# Use the pipeline
pipeline = FeatureEngineeringPipeline()
df_engineered = pipeline.fit_transform(df, target_col='satisfaction')
print(f"Final engineered dataset shape: {df_engineered.shape}")
print(f"Features created: {len(pipeline.feature_names)}")
```

## Summary

Feature engineering is crucial for machine learning success:

- **Feature Creation**: Mathematical transformations, interactions, and domain-specific features
- **Feature Transformation**: Scaling, normalization, and power transformations
- **Feature Selection**: Statistical methods, correlation analysis, and dimensionality reduction
- **Categorical Encoding**: Label, one-hot, target, and frequency encoding
- **Time Series Features**: Date components, cyclical encoding, and time-based features
- **Text Features**: Length, statistical, and sentiment features
- **Best Practices**: Pipeline approach, validation, and documentation

Mastering feature engineering will significantly improve your model performance and interpretability.

## Next Steps

- Practice feature engineering on real datasets
- Explore automated feature engineering libraries
- Learn domain-specific feature engineering techniques
- Study feature importance and interpretability methods

---

**Happy Feature Engineering!** 🔧✨ 