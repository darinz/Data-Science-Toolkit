# Unsupervised Learning with scikit-learn

A comprehensive guide to unsupervised learning techniques including clustering, dimensionality reduction, and association rule learning.

## Table of Contents

1. [Clustering Algorithms](#clustering-algorithms)
2. [Dimensionality Reduction](#dimensionality-reduction)
3. [Association Rule Learning](#association-rule-learning)
4. [Anomaly Detection](#anomaly-detection)
5. [Model Evaluation](#model-evaluation)
6. [Real-world Applications](#real-world-applications)

## Clustering Algorithms

### 1. K-Means Clustering

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, cluster, decomposition, manifold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Load datasets
iris = datasets.load_iris()
wine = datasets.load_wine()
breast_cancer = datasets.load_breast_cancer()

def kmeans_example():
    """Demonstrate K-Means clustering"""
    
    # Use iris dataset
    X = iris.data
    y_true = iris.target
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test different numbers of clusters
    k_values = range(2, 11)
    inertias = []
    silhouette_scores = []
    
    for k in k_values:
        # Create and fit K-Means model
        kmeans = cluster.KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        
        # Calculate metrics
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        
        print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
    
    # Plot elbow curve and silhouette scores
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Elbow curve
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette scores
    ax2.plot(k_values, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal k (highest silhouette score)
    optimal_k = k_values[np.argmax(silhouette_scores)]
    print(f"\nOptimal k based on silhouette score: {optimal_k}")
    
    # Fit optimal model
    optimal_kmeans = cluster.KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    optimal_kmeans.fit(X_scaled)
    
    # Visualize clusters (using first two features)
    plt.figure(figsize=(12, 5))
    
    # True labels
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.title('True Labels')
    plt.colorbar(scatter)
    
    # Predicted clusters
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=optimal_kmeans.labels_, cmap='viridis', alpha=0.7)
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.title(f'K-Means Clusters (k={optimal_k})')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()
    
    return optimal_kmeans, optimal_k

kmeans_model, optimal_k = kmeans_example()
```

### 2. Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def hierarchical_clustering_example():
    """Demonstrate hierarchical clustering"""
    
    # Use iris dataset
    X = iris.data
    y_true = iris.target
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create linkage matrix
    linkage_matrix = linkage(X_scaled, method='ward')
    
    # Plot dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, leaf_rotation=90, leaf_font_size=8)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()
    
    # Test different numbers of clusters
    n_clusters_values = [2, 3, 4, 5]
    results = {}
    
    for n_clusters in n_clusters_values:
        # Create hierarchical clustering model
        hc = AgglomerativeClustering(n_clusters=n_clusters)
        hc.fit(X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, hc.labels_)
        calinski = calinski_harabasz_score(X_scaled, hc.labels_)
        davies = davies_bouldin_score(X_scaled, hc.labels_)
        
        results[n_clusters] = {
            'labels': hc.labels_,
            'silhouette': silhouette,
            'calinski': calinski,
            'davies': davies
        }
        
        print(f"n_clusters={n_clusters}:")
        print(f"  Silhouette Score: {silhouette:.3f}")
        print(f"  Calinski-Harabasz Score: {calinski:.3f}")
        print(f"  Davies-Bouldin Score: {davies:.3f}")
    
    # Find optimal number of clusters
    optimal_n = max(results.keys(), key=lambda k: results[k]['silhouette'])
    print(f"\nOptimal number of clusters: {optimal_n}")
    
    # Visualize optimal clustering
    optimal_labels = results[optimal_n]['labels']
    
    plt.figure(figsize=(12, 5))
    
    # True labels
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.title('True Labels')
    plt.colorbar(scatter)
    
    # Hierarchical clusters
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=optimal_labels, cmap='viridis', alpha=0.7)
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.title(f'Hierarchical Clusters (n={optimal_n})')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()
    
    return results, optimal_n

hc_results, optimal_hc_n = hierarchical_clustering_example()
```

### 3. DBSCAN

```python
from sklearn.cluster import DBSCAN

def dbscan_example():
    """Demonstrate DBSCAN clustering"""
    
    # Use iris dataset
    X = iris.data
    y_true = iris.target
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test different parameters
    eps_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    min_samples_values = [2, 3, 5, 10]
    
    results = {}
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            # Create DBSCAN model
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(X_scaled)
            
            # Get number of clusters (excluding noise points)
            n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
            n_noise = list(dbscan.labels_).count(-1)
            
            # Calculate metrics (only if we have more than 1 cluster)
            if n_clusters > 1:
                silhouette = silhouette_score(X_scaled, dbscan.labels_)
            else:
                silhouette = -1
            
            results[(eps, min_samples)] = {
                'labels': dbscan.labels_,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': silhouette
            }
            
            print(f"eps={eps}, min_samples={min_samples}:")
            print(f"  Clusters: {n_clusters}, Noise points: {n_noise}")
            print(f"  Silhouette Score: {silhouette:.3f}")
    
    # Find best parameters (highest silhouette score, reasonable number of clusters)
    valid_results = {k: v for k, v in results.items() if v['n_clusters'] > 1 and v['n_clusters'] <= 10}
    if valid_results:
        best_params = max(valid_results.keys(), key=lambda k: valid_results[k]['silhouette'])
        best_result = valid_results[best_params]
        
        print(f"\nBest parameters: eps={best_params[0]}, min_samples={best_params[1]}")
        print(f"Best result: {best_result['n_clusters']} clusters, silhouette={best_result['silhouette']:.3f}")
        
        # Visualize best clustering
        best_labels = best_result['labels']
        
        plt.figure(figsize=(12, 5))
        
        # True labels
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis', alpha=0.7)
        plt.xlabel('Feature 1 (scaled)')
        plt.ylabel('Feature 2 (scaled)')
        plt.title('True Labels')
        plt.colorbar(scatter)
        
        # DBSCAN clusters
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=best_labels, cmap='viridis', alpha=0.7)
        plt.xlabel('Feature 1 (scaled)')
        plt.ylabel('Feature 2 (scaled)')
        plt.title(f'DBSCAN Clusters (eps={best_params[0]}, min_samples={best_params[1]})')
        plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.show()
        
        return best_params, best_result
    
    return None, None

best_dbscan_params, best_dbscan_result = dbscan_example()
```

### 4. Gaussian Mixture Models

```python
from sklearn.mixture import GaussianMixture

def gmm_example():
    """Demonstrate Gaussian Mixture Models"""
    
    # Use iris dataset
    X = iris.data
    y_true = iris.target
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test different numbers of components
    n_components_values = range(2, 11)
    results = {}
    
    for n_components in n_components_values:
        # Create GMM model
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X_scaled)
        
        # Get predictions
        labels = gmm.predict(X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels)
        bic = gmm.bic(X_scaled)
        aic = gmm.aic(X_scaled)
        
        results[n_components] = {
            'labels': labels,
            'silhouette': silhouette,
            'bic': bic,
            'aic': aic,
            'model': gmm
        }
        
        print(f"n_components={n_components}:")
        print(f"  Silhouette Score: {silhouette:.3f}")
        print(f"  BIC: {bic:.3f}")
        print(f"  AIC: {aic:.3f}")
    
    # Find optimal number of components
    optimal_n = max(results.keys(), key=lambda k: results[k]['silhouette'])
    print(f"\nOptimal number of components: {optimal_n}")
    
    # Visualize optimal clustering
    optimal_labels = results[optimal_n]['labels']
    
    plt.figure(figsize=(12, 5))
    
    # True labels
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.title('True Labels')
    plt.colorbar(scatter)
    
    # GMM clusters
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=optimal_labels, cmap='viridis', alpha=0.7)
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.title(f'GMM Clusters (n_components={optimal_n})')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()
    
    return results, optimal_n

gmm_results, optimal_gmm_n = gmm_example()
```

## Dimensionality Reduction

### 1. Principal Component Analysis (PCA)

```python
def pca_example():
    """Demonstrate Principal Component Analysis"""
    
    # Use iris dataset
    X = iris.data
    y_true = iris.target
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = decomposition.PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Analyze explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    print("PCA Results:")
    print(f"Explained variance ratio: {explained_variance_ratio}")
    print(f"Cumulative explained variance ratio: {cumulative_variance_ratio}")
    
    # Plot explained variance
    plt.figure(figsize=(15, 5))
    
    # Explained variance ratio
    plt.subplot(1, 3, 1)
    plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio')
    
    # Cumulative explained variance
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plot of first two components
    plt.subplot(1, 3, 3)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA: First Two Components')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()
    
    # Determine optimal number of components
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    print(f"\nNumber of components needed for 95% variance: {n_components_95}")
    
    # Apply PCA with optimal number of components
    pca_optimal = decomposition.PCA(n_components=n_components_95)
    X_pca_optimal = pca_optimal.fit_transform(X_scaled)
    
    print(f"Reduced dimensions: {X_scaled.shape[1]} -> {X_pca_optimal.shape[1]}")
    
    return pca, X_pca, pca_optimal, X_pca_optimal

pca_model, X_pca, pca_optimal, X_pca_optimal = pca_example()
```

### 2. t-SNE

```python
def tsne_example():
    """Demonstrate t-SNE dimensionality reduction"""
    
    # Use iris dataset
    X = iris.data
    y_true = iris.target
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply t-SNE
    tsne = manifold.TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Visualize t-SNE results
    plt.figure(figsize=(12, 5))
    
    # Original data (first two features)
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.xlabel('Feature 1 (scaled)')
    plt.ylabel('Feature 2 (scaled)')
    plt.title('Original Data (First Two Features)')
    plt.colorbar(scatter)
    
    # t-SNE embedding
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Embedding')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.show()
    
    # Test different perplexity values
    perplexity_values = [5, 10, 20, 30, 50]
    
    plt.figure(figsize=(15, 3))
    
    for i, perplexity in enumerate(perplexity_values):
        tsne_temp = manifold.TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne_temp = tsne_temp.fit_transform(X_scaled)
        
        plt.subplot(1, len(perplexity_values), i + 1)
        scatter = plt.scatter(X_tsne_temp[:, 0], X_tsne_temp[:, 1], c=y_true, cmap='viridis', alpha=0.7)
        plt.title(f'Perplexity = {perplexity}')
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.show()
    
    return tsne, X_tsne

tsne_model, X_tsne = tsne_example()
```

### 3. UMAP

```python
try:
    import umap
    
    def umap_example():
        """Demonstrate UMAP dimensionality reduction"""
        
        # Use iris dataset
        X = iris.data
        y_true = iris.target
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply UMAP
        umap_reducer = umap.UMAP(random_state=42)
        X_umap = umap_reducer.fit_transform(X_scaled)
        
        # Visualize UMAP results
        plt.figure(figsize=(12, 5))
        
        # Original data (first two features)
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis', alpha=0.7)
        plt.xlabel('Feature 1 (scaled)')
        plt.ylabel('Feature 2 (scaled)')
        plt.title('Original Data (First Two Features)')
        plt.colorbar(scatter)
        
        # UMAP embedding
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_true, cmap='viridis', alpha=0.7)
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.title('UMAP Embedding')
        plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.show()
        
        return umap_reducer, X_umap
    
    umap_model, X_umap = umap_example()
    
except ImportError:
    print("UMAP not installed. Install with: pip install umap-learn")
```

## Association Rule Learning

```python
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def association_rules_example():
    """Demonstrate association rule learning"""
    
    # Create sample transaction data
    transactions = [
        ['milk', 'bread', 'butter'],
        ['bread', 'eggs'],
        ['milk', 'bread', 'eggs'],
        ['milk', 'bread', 'butter', 'eggs'],
        ['bread', 'butter'],
        ['milk', 'eggs'],
        ['milk', 'bread', 'butter'],
        ['bread', 'eggs', 'butter'],
        ['milk', 'bread'],
        ['milk', 'butter', 'eggs']
    ]
    
    # Convert to one-hot encoded format
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    print("Transaction Data:")
    print(df)
    
    # Find frequent itemsets
    frequent_itemsets = apriori(df, min_support=0.3, use_colnames=True)
    
    print(f"\nFrequent Itemsets (min_support=0.3):")
    print(frequent_itemsets)
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    print(f"\nAssociation Rules (min_confidence=0.5):")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    
    # Visualize rules
    plt.figure(figsize=(12, 5))
    
    # Support vs Confidence
    plt.subplot(1, 2, 1)
    plt.scatter(rules['support'], rules['confidence'], alpha=0.7)
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Support vs Confidence')
    plt.grid(True, alpha=0.3)
    
    # Support vs Lift
    plt.subplot(1, 2, 2)
    plt.scatter(rules['support'], rules['lift'], alpha=0.7)
    plt.xlabel('Support')
    plt.ylabel('Lift')
    plt.title('Support vs Lift')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Find strong rules (high lift)
    strong_rules = rules[rules['lift'] > 1.5]
    
    print(f"\nStrong Rules (lift > 1.5):")
    print(strong_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    
    return rules, strong_rules

association_rules_result, strong_rules = association_rules_example()
```

## Anomaly Detection

### 1. Isolation Forest

```python
from sklearn.ensemble import IsolationForest

def isolation_forest_example():
    """Demonstrate isolation forest for anomaly detection"""
    
    # Create sample data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (1000, 2))
    outlier_data = np.random.normal(0, 4, (50, 2))
    
    X = np.vstack([normal_data, outlier_data])
    y_true = np.hstack([np.zeros(1000), np.ones(50)])  # 0: normal, 1: outlier
    
    # Apply isolation forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    y_pred = iso_forest.fit_predict(X)
    
    # Convert predictions: -1 for outliers, 1 for normal
    y_pred_binary = (y_pred == -1).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    print("Isolation Forest Results:")
    print(classification_report(y_true, y_pred_binary, target_names=['Normal', 'Outlier']))
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Original data
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Original Data (True Labels)')
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'))
    
    # Predicted outliers
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred_binary, cmap='viridis', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Isolation Forest Predictions')
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'))
    
    plt.tight_layout()
    plt.show()
    
    return iso_forest, y_pred_binary

iso_forest_model, iso_forest_pred = isolation_forest_example()
```

### 2. Local Outlier Factor (LOF)

```python
from sklearn.neighbors import LocalOutlierFactor

def lof_example():
    """Demonstrate Local Outlier Factor for anomaly detection"""
    
    # Create sample data with outliers
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (1000, 2))
    outlier_data = np.random.normal(0, 4, (50, 2))
    
    X = np.vstack([normal_data, outlier_data])
    y_true = np.hstack([np.zeros(1000), np.ones(50)])  # 0: normal, 1: outlier
    
    # Apply LOF
    lof = LocalOutlierFactor(contamination=0.05)
    y_pred = lof.fit_predict(X)
    
    # Convert predictions: -1 for outliers, 1 for normal
    y_pred_binary = (y_pred == -1).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import classification_report
    
    print("Local Outlier Factor Results:")
    print(classification_report(y_true, y_pred_binary, target_names=['Normal', 'Outlier']))
    
    # Visualize results
    plt.figure(figsize=(12, 5))
    
    # Original data
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Original Data (True Labels)')
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'))
    
    # Predicted outliers
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred_binary, cmap='viridis', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('LOF Predictions')
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'))
    
    plt.tight_layout()
    plt.show()
    
    return lof, y_pred_binary

lof_model, lof_pred = lof_example()
```

## Model Evaluation

### Clustering Evaluation

```python
def clustering_evaluation(X, y_true, clustering_results):
    """Evaluate clustering results"""
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    evaluation_results = {}
    
    for name, labels in clustering_results.items():
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels)
        calinski = calinski_harabasz_score(X_scaled, labels)
        davies = davies_bouldin_score(X_scaled, labels)
        
        evaluation_results[name] = {
            'silhouette': silhouette,
            'calinski': calinski,
            'davies': davies
        }
        
        print(f"{name}:")
        print(f"  Silhouette Score: {silhouette:.3f}")
        print(f"  Calinski-Harabasz Score: {calinski:.3f}")
        print(f"  Davies-Bouldin Score: {davies:.3f}")
    
    # Compare results
    metrics_df = pd.DataFrame(evaluation_results).T
    
    plt.figure(figsize=(15, 5))
    
    for i, metric in enumerate(['silhouette', 'calinski', 'davies']):
        plt.subplot(1, 3, i + 1)
        bars = plt.bar(metrics_df.index, metrics_df[metric], alpha=0.7)
        plt.title(f'{metric.replace("_", " ").title()} Score')
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45)
        
        # Add value labels
        for bar, value in zip(bars, metrics_df[metric]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return evaluation_results

# Example clustering evaluation
clustering_results = {
    'K-Means': kmeans_model.labels_,
    'Hierarchical': hc_results[optimal_hc_n]['labels'],
    'GMM': gmm_results[optimal_gmm_n]['labels']
}

evaluation_results = clustering_evaluation(iris.data, iris.target, clustering_results)
```

## Real-world Applications

### 1. Customer Segmentation

```python
def customer_segmentation_example():
    """Demonstrate customer segmentation using clustering"""
    
    # Create sample customer data
    np.random.seed(42)
    n_customers = 1000
    
    # Generate customer features
    age = np.random.normal(35, 10, n_customers)
    income = np.random.normal(50000, 20000, n_customers)
    spending_score = np.random.normal(50, 20, n_customers)
    frequency = np.random.poisson(5, n_customers)
    
    # Create customer dataframe
    customer_data = pd.DataFrame({
        'age': age,
        'income': income,
        'spending_score': spending_score,
        'frequency': frequency
    })
    
    # Scale features
    scaler = StandardScaler()
    customer_data_scaled = scaler.fit_transform(customer_data)
    
    # Apply K-Means clustering
    kmeans = cluster.KMeans(n_clusters=4, random_state=42, n_init=10)
    customer_data['cluster'] = kmeans.fit_predict(customer_data_scaled)
    
    # Analyze clusters
    cluster_analysis = customer_data.groupby('cluster').agg({
        'age': ['mean', 'std'],
        'income': ['mean', 'std'],
        'spending_score': ['mean', 'std'],
        'frequency': ['mean', 'std']
    }).round(2)
    
    print("Customer Segmentation Results:")
    print(cluster_analysis)
    
    # Visualize clusters
    plt.figure(figsize=(15, 5))
    
    # Age vs Income
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(customer_data['age'], customer_data['income'], 
                         c=customer_data['cluster'], cmap='viridis', alpha=0.7)
    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.title('Age vs Income')
    plt.colorbar(scatter)
    
    # Spending Score vs Frequency
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(customer_data['spending_score'], customer_data['frequency'], 
                         c=customer_data['cluster'], cmap='viridis', alpha=0.7)
    plt.xlabel('Spending Score')
    plt.ylabel('Frequency')
    plt.title('Spending Score vs Frequency')
    plt.colorbar(scatter)
    
    # 3D plot
    from mpl_toolkits.mplot3d import Axes3D
    ax = plt.subplot(1, 3, 3, projection='3d')
    scatter = ax.scatter(customer_data['age'], customer_data['income'], customer_data['spending_score'],
                        c=customer_data['cluster'], cmap='viridis', alpha=0.7)
    ax.set_xlabel('Age')
    ax.set_ylabel('Income')
    ax.set_zlabel('Spending Score')
    ax.set_title('3D Customer Segmentation')
    
    plt.tight_layout()
    plt.show()
    
    return customer_data, kmeans

customer_segments, customer_kmeans = customer_segmentation_example()
```

### 2. Document Clustering

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def document_clustering_example():
    """Demonstrate document clustering"""
    
    # Sample documents
    documents = [
        "machine learning algorithms and data science",
        "artificial intelligence and deep learning",
        "data analysis and statistical modeling",
        "computer vision and image processing",
        "natural language processing and text mining",
        "big data and distributed computing",
        "database systems and data management",
        "software engineering and development",
        "web development and programming",
        "mobile app development and design",
        "cybersecurity and network security",
        "cloud computing and virtualization",
        "blockchain and cryptocurrency",
        "internet of things and sensors",
        "robotics and automation"
    ]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    X_tfidf = vectorizer.fit_transform(documents)
    
    # Apply K-Means clustering
    kmeans = cluster.KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_tfidf)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'document': documents,
        'cluster': cluster_labels
    })
    
    print("Document Clustering Results:")
    for cluster_id in range(3):
        print(f"\nCluster {cluster_id}:")
        cluster_docs = results_df[results_df['cluster'] == cluster_id]['document'].tolist()
        for doc in cluster_docs:
            print(f"  - {doc}")
    
    # Apply LDA for topic modeling
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(X_tfidf)
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Display topics
    print(f"\nLDA Topics:")
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-5:]]
        print(f"Topic {topic_idx}: {', '.join(top_words)}")
    
    return results_df, kmeans, lda

doc_clusters, doc_kmeans, doc_lda = document_clustering_example()
```

## Summary

### Key Concepts Covered:

1. **Clustering Algorithms**: K-Means, Hierarchical, DBSCAN, GMM
2. **Dimensionality Reduction**: PCA, t-SNE, UMAP
3. **Association Rules**: Apriori algorithm and rule generation
4. **Anomaly Detection**: Isolation Forest, LOF
5. **Evaluation Metrics**: Silhouette, Calinski-Harabasz, Davies-Bouldin
6. **Real-world Applications**: Customer segmentation, document clustering

### Best Practices:

1. **Scale your data** before applying clustering algorithms
2. **Use multiple evaluation metrics** to assess clustering quality
3. **Try different algorithms** as they work well for different data types
4. **Visualize results** to understand cluster structure
5. **Consider domain knowledge** when interpreting results
6. **Validate results** with business stakeholders

### Next Steps:

- Explore advanced clustering techniques (spectral clustering, density-based)
- Learn about deep learning for unsupervised learning
- Practice with real-world datasets
- Master model deployment and production considerations 