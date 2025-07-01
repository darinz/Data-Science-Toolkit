# scikit-learn Tutorials

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green.svg)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-blue.svg)](https://matplotlib.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Conda](https://img.shields.io/badge/Conda-Environment-green.svg)](https://docs.conda.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

A comprehensive collection of scikit-learn tutorials covering machine learning fundamentals, algorithms, and practical applications.

## What's Included

### Machine Learning Fundamentals
- **Supervised Learning** - Classification and regression algorithms
- **Unsupervised Learning** - Clustering and dimensionality reduction
- **Model Evaluation** - Cross-validation, metrics, and hyperparameter tuning
- **Feature Engineering** - Preprocessing, scaling, and feature selection
- **Pipeline Construction** - Building end-to-end ML workflows

### Algorithms Covered
- **Classification**: Logistic Regression, SVM, Random Forest, Naive Bayes
- **Regression**: Linear Regression, Ridge, Lasso, Elastic Net
- **Clustering**: K-Means, DBSCAN, Hierarchical Clustering
- **Dimensionality Reduction**: PCA, t-SNE, LDA
- **Ensemble Methods**: Random Forest, Gradient Boosting, Voting

### Practical Applications
- **Data Preprocessing** - Handling missing values, outliers, and categorical data
- **Model Selection** - Choosing the right algorithm for your problem
- **Hyperparameter Tuning** - Grid search, random search, and Bayesian optimization
- **Feature Selection** - Identifying the most important features
- **Model Deployment** - Saving and loading trained models

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Basic understanding of Python and statistics
- Familiarity with NumPy and pandas (covered in other tutorials)

### Installation

**Using Conda (Recommended):**
```bash
cd scikit-learn
conda env create -f environment.yml
conda activate sklearn-tutorials
```

**Using pip:**
```bash
cd scikit-learn
pip install -r requirements.txt
```

### Verify Installation
```python
import sklearn
print(f"scikit-learn version: {sklearn.__version__}")
```

## File Structure

```
scikit-learn/
├── 01_ml_basics_guide.md                 # Machine learning fundamentals
├── 02_supervised_learning_guide.md       # Classification and regression
├── 03_unsupervised_learning_guide.md     # Clustering and dimensionality reduction
├── 04_feature_engineering_guide.md       # Data preprocessing and feature engineering
├── 05_model_selection_guide.md           # Model evaluation and selection
├── 06_real_world_applications_guide.md   # Practical applications and case studies
├── environment.yml                    # Conda environment configuration
├── requirements.txt                   # pip dependencies
└── README.md                          # This file
```

## Tutorial Structure

### 1. Machine Learning Basics (`01_ml_basics_guide.md`)
- Understanding supervised vs unsupervised learning
- Train-test split and cross-validation
- Model evaluation metrics
- Overfitting and underfitting
- Bias-variance tradeoff
- **Read:** `01_ml_basics_guide.md`

### 2. Supervised Learning (`02_supervised_learning_guide.md`)
- Classification algorithms (Logistic Regression, SVM, Random Forest)
- Regression algorithms (Linear Regression, Ridge, Lasso)
- Model evaluation and comparison
- Feature importance analysis
- **Read:** `02_supervised_learning_guide.md`

### 3. Unsupervised Learning (`03_unsupervised_learning_guide.md`)
- Clustering algorithms (K-Means, DBSCAN)
- Dimensionality reduction (PCA, t-SNE)
- Association rule learning
- Anomaly detection
- **Read:** `03_unsupervised_learning_guide.md`

### 4. Feature Engineering (`04_feature_engineering_guide.md`)
- Data preprocessing techniques
- Feature scaling and normalization
- Handling categorical variables
- Feature selection methods
- **Read:** `04_feature_engineering_guide.md`

### 5. Model Selection and Tuning (`05_model_selection_guide.md`)
- Cross-validation strategies
- Hyperparameter tuning (Grid Search, Random Search)
- Model comparison and selection
- Ensemble methods
- **Read:** `05_model_selection_guide.md`

### 6. Real-world Applications (`06_real_world_applications_guide.md`)
- Customer segmentation
- Predictive maintenance
- Sentiment analysis
- Recommendation systems
- **Read:** `06_real_world_applications_guide.md`

## Running the Tutorials

### Reading the Guides
All tutorials are provided as comprehensive markdown guides:

```bash
# Read guides in your preferred markdown viewer
# Or open them in Jupyter Lab/Notebook for better formatting
jupyter lab 01_ml_basics_guide.md
jupyter lab 02_supervised_learning_guide.md
jupyter lab 03_unsupervised_learning_guide.md
jupyter lab 04_feature_engineering_guide.md
jupyter lab 05_model_selection_guide.md
jupyter lab 06_real_world_applications_guide.md
```

### Interactive Learning
Create your own notebooks to follow along with the guides:

```bash
# Start Jupyter Lab
jupyter lab

# Create new notebooks for each topic
# Copy code examples from the guides
# Experiment with different parameters and datasets
```

## Environment Management

### Conda Commands

```bash
# Create new environment
conda env create -f environment.yml

# Activate environment
conda activate sklearn-tutorials

# Update environment (when dependencies change)
conda env update -f environment.yml --prune

# Remove environment
conda remove --name sklearn-tutorials --all

# List all environments
conda env list

# Deactivate current environment
conda deactivate
```

### pip Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Check installed packages
pip list

# Export current environment
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
```

## Learning Path

### Beginner Path
1. **Start with ML Basics** - Understand fundamental concepts
2. **Learn Supervised Learning** - Master classification and regression
3. **Practice Feature Engineering** - Prepare data for modeling
4. **Explore Model Selection** - Choose and tune the best models

### Intermediate Path
1. **Dive into Unsupervised Learning** - Discover patterns in data
2. **Master Advanced Techniques** - Ensemble methods and optimization
3. **Work with Real Datasets** - Apply ML to practical problems
4. **Optimize Performance** - Improve model accuracy and efficiency

### Advanced Path
1. **Custom Algorithms** - Implement custom ML algorithms
2. **Production Deployment** - Deploy models to production
3. **Research Applications** - Apply ML to research problems
4. **Specialized Domains** - Domain-specific ML applications

## Key Concepts Covered

### Machine Learning Fundamentals
- **Supervised vs Unsupervised** - Understanding learning paradigms
- **Training and Testing** - Data splitting and validation
- **Overfitting** - Recognizing and preventing overfitting
- **Bias-Variance Tradeoff** - Balancing model complexity

### Algorithm Categories
- **Linear Models** - Simple, interpretable algorithms
- **Tree-based Models** - Non-linear, feature importance
- **Support Vector Machines** - Kernel methods and margins
- **Neural Networks** - Deep learning foundations
- **Ensemble Methods** - Combining multiple models

### Model Evaluation
- **Cross-validation** - Robust model assessment
- **Metrics** - Accuracy, precision, recall, F1-score
- **Hyperparameter Tuning** - Optimizing model parameters
- **Feature Selection** - Identifying important features

### Data Preprocessing
- **Scaling** - Normalization and standardization
- **Encoding** - Categorical variable handling
- **Missing Values** - Imputation strategies
- **Outlier Detection** - Identifying and handling outliers

## Common Use Cases

### Classification Example
```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Regression Example
```python
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
boston = load_boston()
X, y = boston.data, boston.target

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict and evaluate
y_pred = model.predict(X)
print(f"R² Score: {r2_score(y, y_pred):.3f}")
print(f"MSE: {mean_squared_error(y, y_pred):.3f}")
```

### Clustering Example
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate data
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Perform clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X)

# Visualize
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='x', s=200, linewidths=3)
plt.show()
```

## Integration with Other Libraries

### NumPy and pandas Integration
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load data with pandas
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Preprocess with scikit-learn
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
clf = RandomForestClassifier()
clf.fit(X_scaled, y)
```

### Matplotlib Integration
```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X = iris.data

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Visualize
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()
```

## Additional Resources

### Official Documentation
- [scikit-learn Official Documentation](https://scikit-learn.org/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [scikit-learn Examples](https://scikit-learn.org/stable/auto_examples/index.html)
- [scikit-learn API Reference](https://scikit-learn.org/stable/modules/classes.html)

### Learning Resources
- [scikit-learn GitHub Repository](https://github.com/scikit-learn/scikit-learn)
- [scikit-learn Community](https://scikit-learn.org/community/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)

### Recommended Books
- "Introduction to Machine Learning with Python" by Andreas Müller
- "Hands-On Machine Learning" by Aurélien Géron
- "Python Machine Learning" by Sebastian Raschka

## Contributing

Found an error or have a suggestion? Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

**Happy Machine Learning!** 