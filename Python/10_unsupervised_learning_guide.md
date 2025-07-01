# Python Unsupervised Learning Guide

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-orange.svg)](https://scikit-learn.org/)
[![Clustering](https://img.shields.io/badge/Clustering-Algorithms-green.svg)](https://scikit-learn.org/stable/modules/clustering.html)

A comprehensive guide to Unsupervised Learning in Python for data science and machine learning applications.

## Table of Contents

1. [Introduction to Unsupervised Learning](#introduction-to-unsupervised-learning)
2. [Clustering Algorithms](#clustering-algorithms)
3. [Dimensionality Reduction](#dimensionality-reduction)
4. [Association Rule Learning](#association-rule-learning)
5. [Anomaly Detection](#anomaly-detection)
6. [Density Estimation](#density-estimation)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Best Practices](#best-practices)

## Introduction to Unsupervised Learning

Unsupervised learning finds hidden patterns in data without labeled outputs.

### Types of Unsupervised Learning

- **Clustering**: Group similar data points
- **Dimensionality Reduction**: Reduce feature space
- **Association Rules**: Find relationships between variables
- **Anomaly Detection**: Identify unusual patterns

### Basic Setup

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# Sample dataset
np.random.seed(42)
n_samples = 1000

# Generate clusters
cluster1 = np.random.normal([2, 2], 0.5, (n_samples//3, 2))
cluster2 = np.random.normal([8, 3], 0.8, (n_samples//3, 2))
cluster3 = np.random.normal([5, 8], 0.6, (n_samples//3, 2))

X = np.vstack([cluster1, cluster2, cluster3])
df = pd.DataFrame(X, columns=['feature1', 'feature2'])

plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title('Sample Dataset for Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```

## Clustering Algorithms

### K-Means Clustering

```python
def kmeans_clustering(X, n_clusters=3):
    """Perform K-means clustering with elbow method."""
    
    # Find optimal number of clusters
    inertias = []
    silhouette_scores = []
    K_range = range(2, 11)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    # Plot elbow curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    
    ax2.plot(K_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    
    plt.tight_layout()
    plt.show()
    
    # Optimal clustering
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Final clustering
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = final_kmeans.fit_predict(X)
    
    return labels, final_kmeans

# Perform K-means clustering
labels, kmeans_model = kmeans_clustering(X)

# Visualize results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
```

### DBSCAN Clustering

```python
def dbscan_clustering(X, eps=0.5, min_samples=5):
    """Perform DBSCAN clustering."""
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    # Count clusters (excluding noise)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    
    # Visualize results
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title('DBSCAN Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    
    return labels, dbscan

# Perform DBSCAN clustering
dbscan_labels, dbscan_model = dbscan_clustering(X, eps=0.8, min_samples=10)
```

### Hierarchical Clustering

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def hierarchical_clustering(X, n_clusters=3):
    """Perform hierarchical clustering."""
    
    # Create linkage matrix
    linkage_matrix = linkage(X, method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()
    
    # Cut dendrogram to get clusters
    labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Visualize results
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title('Hierarchical Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    
    return labels

# Perform hierarchical clustering
hierarchical_labels = hierarchical_clustering(X, n_clusters=3)
```

### Gaussian Mixture Models

```python
from sklearn.mixture import GaussianMixture

def gmm_clustering(X, n_components=3):
    """Perform Gaussian Mixture Model clustering."""
    
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    labels = gmm.fit_predict(X)
    
    # Plot results
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title('Gaussian Mixture Model Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    
    # Plot probability contours
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    Z = gmm.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    plt.title('GMM with Probability Contours')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    
    return labels, gmm

# Perform GMM clustering
gmm_labels, gmm_model = gmm_clustering(X, n_components=3)
```

## Dimensionality Reduction

### Principal Component Analysis (PCA)

```python
def pca_analysis(X, n_components=None):
    """Perform Principal Component Analysis."""
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    if n_components is None:
        pca = PCA()
    else:
        pca = PCA(n_components=n_components)
    
    X_pca = pca.fit_transform(X_scaled)
    
    # Explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Plot explained variance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Explained Variance by Component')
    
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Explained Variance')
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(X_pca.shape[1])],
        index=[f'Feature{i+1}' for i in range(X.shape[1])]
    )
    
    print("Component Loadings:")
    print(loadings)
    
    return X_pca, pca, loadings

# Perform PCA
X_pca, pca_model, loadings = pca_analysis(X, n_components=2)

# Visualize PCA results
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
plt.title('PCA Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
```

### t-SNE for Visualization

```python
def tsne_visualization(X, perplexity=30):
    """Perform t-SNE for dimensionality reduction and visualization."""
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6)
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()
    
    return X_tsne

# Perform t-SNE
X_tsne = tsne_visualization(X, perplexity=30)
```

### Non-negative Matrix Factorization (NMF)

```python
def nmf_decomposition(X, n_components=2):
    """Perform Non-negative Matrix Factorization."""
    
    # Ensure non-negative data
    X_non_negative = X - X.min()
    
    nmf = NMF(n_components=n_components, random_state=42)
    X_nmf = nmf.fit_transform(X_non_negative)
    
    plt.scatter(X_nmf[:, 0], X_nmf[:, 1], alpha=0.6)
    plt.title('NMF Results')
    plt.xlabel('NMF Component 1')
    plt.ylabel('NMF Component 2')
    plt.show()
    
    return X_nmf, nmf

# Perform NMF
X_nmf, nmf_model = nmf_decomposition(X, n_components=2)
```

## Association Rule Learning

### Apriori Algorithm

```python
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def association_rule_mining(transactions, min_support=0.01, min_confidence=0.5):
    """Perform association rule mining using Apriori algorithm."""
    
    # Convert transactions to one-hot encoding
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    
    # Find frequent itemsets
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # Sort by lift
    rules = rules.sort_values('lift', ascending=False)
    
    print("Top Association Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())
    
    return rules, frequent_itemsets

# Example transactions
transactions = [
    ['milk', 'bread', 'butter'],
    ['bread', 'diapers'],
    ['milk', 'diapers', 'beer', 'eggs'],
    ['milk', 'bread', 'diapers'],
    ['bread', 'diapers', 'beer']
]

# Perform association rule mining
rules, frequent_itemsets = association_rule_mining(transactions, min_support=0.3, min_confidence=0.6)
```

## Anomaly Detection

### Isolation Forest

```python
from sklearn.ensemble import IsolationForest

def isolation_forest_detection(X, contamination=0.1):
    """Detect anomalies using Isolation Forest."""
    
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    labels = iso_forest.fit_predict(X)
    
    # Separate normal and anomalous points
    normal_points = X[labels == 1]
    anomalous_points = X[labels == -1]
    
    # Visualize results
    plt.scatter(normal_points[:, 0], normal_points[:, 1], 
               c='blue', alpha=0.6, label='Normal')
    plt.scatter(anomalous_points[:, 0], anomalous_points[:, 1], 
               c='red', alpha=0.6, label='Anomaly')
    plt.title('Isolation Forest Anomaly Detection')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    
    return labels, iso_forest

# Perform anomaly detection
anomaly_labels, iso_model = isolation_forest_detection(X, contamination=0.05)
```

### Local Outlier Factor (LOF)

```python
from sklearn.neighbors import LocalOutlierFactor

def lof_detection(X, contamination=0.1):
    """Detect anomalies using Local Outlier Factor."""
    
    lof = LocalOutlierFactor(contamination=contamination)
    labels = lof.fit_predict(X)
    
    # Separate normal and anomalous points
    normal_points = X[labels == 1]
    anomalous_points = X[labels == -1]
    
    # Visualize results
    plt.scatter(normal_points[:, 0], normal_points[:, 1], 
               c='blue', alpha=0.6, label='Normal')
    plt.scatter(anomalous_points[:, 0], anomalous_points[:, 1], 
               c='red', alpha=0.6, label='Anomaly')
    plt.title('Local Outlier Factor Anomaly Detection')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
    
    return labels, lof

# Perform LOF detection
lof_labels, lof_model = lof_detection(X, contamination=0.05)
```

## Density Estimation

### Kernel Density Estimation

```python
from sklearn.neighbors import KernelDensity

def kernel_density_estimation(X, bandwidth=0.5):
    """Perform kernel density estimation."""
    
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(X)
    
    # Create grid for visualization
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Calculate density
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    density = np.exp(kde.score_samples(grid_points))
    density = density.reshape(xx.shape)
    
    # Plot density
    plt.contourf(xx, yy, density, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6, c='red')
    plt.title('Kernel Density Estimation')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Density')
    plt.show()
    
    return kde

# Perform KDE
kde_model = kernel_density_estimation(X, bandwidth=0.3)
```

## Evaluation Metrics

### Clustering Evaluation

```python
def evaluate_clustering(X, labels):
    """Evaluate clustering results using multiple metrics."""
    
    # Silhouette Score
    silhouette_avg = silhouette_score(X, labels)
    
    # Calinski-Harabasz Index
    calinski_score = calinski_harabasz_score(X, labels)
    
    # Davies-Bouldin Index
    from sklearn.metrics import davies_bouldin_score
    davies_score = davies_bouldin_score(X, labels)
    
    print("Clustering Evaluation Metrics:")
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    print(f"Calinski-Harabasz Index: {calinski_score:.3f}")
    print(f"Davies-Bouldin Index: {davies_score:.3f}")
    
    return {
        'silhouette': silhouette_avg,
        'calinski_harabasz': calinski_score,
        'davies_bouldin': davies_score
    }

# Evaluate clustering
evaluation_results = evaluate_clustering(X, kmeans_model.labels_)
```

### Dimensionality Reduction Evaluation

```python
def evaluate_dimensionality_reduction(X_original, X_reduced):
    """Evaluate dimensionality reduction quality."""
    
    # Reconstruction error (for PCA)
    from sklearn.metrics import mean_squared_error
    
    # For demonstration, we'll use a simple approach
    # In practice, you'd reconstruct and compare
    
    print("Dimensionality Reduction Evaluation:")
    print(f"Original dimensions: {X_original.shape[1]}")
    print(f"Reduced dimensions: {X_reduced.shape[1]}")
    print(f"Compression ratio: {X_reduced.shape[1]/X_original.shape[1]:.2f}")
    
    return {
        'original_dim': X_original.shape[1],
        'reduced_dim': X_reduced.shape[1],
        'compression_ratio': X_reduced.shape[1]/X_original.shape[1]
    }

# Evaluate dimensionality reduction
reduction_evaluation = evaluate_dimensionality_reduction(X, X_pca)
```

## Best Practices

### Complete Unsupervised Learning Pipeline

```python
class UnsupervisedLearningPipeline:
    """Complete unsupervised learning pipeline."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def clustering_pipeline(self, X, methods=['kmeans', 'dbscan', 'hierarchical']):
        """Complete clustering pipeline."""
        
        results = {}
        
        for method in methods:
            if method == 'kmeans':
                labels, model = kmeans_clustering(X)
                results['kmeans'] = {'labels': labels, 'model': model}
                
            elif method == 'dbscan':
                labels, model = dbscan_clustering(X)
                results['dbscan'] = {'labels': labels, 'model': model}
                
            elif method == 'hierarchical':
                labels = hierarchical_clustering(X)
                results['hierarchical'] = {'labels': labels}
        
        self.results['clustering'] = results
        return results
    
    def dimensionality_reduction_pipeline(self, X, methods=['pca', 'tsne', 'nmf']):
        """Complete dimensionality reduction pipeline."""
        
        results = {}
        
        for method in methods:
            if method == 'pca':
                X_reduced, model, loadings = pca_analysis(X)
                results['pca'] = {'X_reduced': X_reduced, 'model': model, 'loadings': loadings}
                
            elif method == 'tsne':
                X_reduced = tsne_visualization(X)
                results['tsne'] = {'X_reduced': X_reduced}
                
            elif method == 'nmf':
                X_reduced, model = nmf_decomposition(X)
                results['nmf'] = {'X_reduced': X_reduced, 'model': model}
        
        self.results['dimensionality_reduction'] = results
        return results
    
    def anomaly_detection_pipeline(self, X, methods=['isolation_forest', 'lof']):
        """Complete anomaly detection pipeline."""
        
        results = {}
        
        for method in methods:
            if method == 'isolation_forest':
                labels, model = isolation_forest_detection(X)
                results['isolation_forest'] = {'labels': labels, 'model': model}
                
            elif method == 'lof':
                labels, model = lof_detection(X)
                results['lof'] = {'labels': labels, 'model': model}
        
        self.results['anomaly_detection'] = results
        return results
    
    def generate_report(self):
        """Generate comprehensive report."""
        
        report = "=== UNSUPERVISED LEARNING REPORT ===\n"
        
        if 'clustering' in self.results:
            report += "\n--- CLUSTERING RESULTS ---\n"
            for method, result in self.results['clustering'].items():
                n_clusters = len(set(result['labels']))
                report += f"{method.upper()}: {n_clusters} clusters\n"
        
        if 'dimensionality_reduction' in self.results:
            report += "\n--- DIMENSIONALITY REDUCTION RESULTS ---\n"
            for method, result in self.results['dimensionality_reduction'].items():
                if 'X_reduced' in result:
                    report += f"{method.upper()}: {result['X_reduced'].shape[1]} components\n"
        
        if 'anomaly_detection' in self.results:
            report += "\n--- ANOMALY DETECTION RESULTS ---\n"
            for method, result in self.results['anomaly_detection'].items():
                n_anomalies = sum(result['labels'] == -1)
                report += f"{method.upper()}: {n_anomalies} anomalies detected\n"
        
        return report

# Use the pipeline
pipeline = UnsupervisedLearningPipeline()

# Run clustering
clustering_results = pipeline.clustering_pipeline(X, methods=['kmeans', 'dbscan'])

# Run dimensionality reduction
reduction_results = pipeline.dimensionality_reduction_pipeline(X, methods=['pca', 'tsne'])

# Run anomaly detection
anomaly_results = pipeline.anomaly_detection_pipeline(X, methods=['isolation_forest'])

# Generate report
report = pipeline.generate_report()
print(report)
```

## Summary

Unsupervised learning provides powerful tools for data exploration:

- **Clustering**: Group similar data points using K-means, DBSCAN, hierarchical clustering
- **Dimensionality Reduction**: Reduce feature space with PCA, t-SNE, NMF
- **Association Rules**: Find relationships between variables
- **Anomaly Detection**: Identify unusual patterns
- **Density Estimation**: Model data distributions
- **Evaluation**: Use appropriate metrics for each task

Mastering unsupervised learning will help you discover hidden patterns and insights in your data.

## Next Steps

- Practice with real-world datasets
- Explore advanced clustering algorithms
- Learn about deep learning for unsupervised learning
- Study interpretability techniques for unsupervised models

---

**Happy Unsupervised Learning!** 🔍✨ 