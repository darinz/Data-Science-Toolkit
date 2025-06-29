# Real-World Applications with scikit-learn

A comprehensive guide to real-world machine learning applications using scikit-learn.

## Table of Contents

1. [Customer Segmentation](#customer-segmentation)
2. [Predictive Maintenance](#predictive-maintenance)
3. [Sentiment Analysis](#sentiment-analysis)
4. [Recommendation Systems](#recommendation-systems)

## Customer Segmentation (Clustering)

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Simulate customer data
np.random.seed(42)
data = pd.DataFrame({
    'age': np.random.randint(18, 70, 200),
    'income': np.random.randint(20000, 120000, 200),
    'spending_score': np.random.randint(1, 100, 200)
})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=4, random_state=42)
data['cluster'] = kmeans.fit_predict(X_scaled)

plt.scatter(data['income'], data['spending_score'], c=data['cluster'], cmap='viridis')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.show()
```

## Predictive Maintenance (Classification)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Simulate equipment data
np.random.seed(42)
data = pd.DataFrame({
    'temperature': np.random.normal(70, 10, 1000),
    'vibration': np.random.normal(0.5, 0.1, 1000),
    'pressure': np.random.normal(30, 5, 1000),
    'failure': np.random.binomial(1, 0.1, 1000)
})

X = data[['temperature', 'vibration', 'pressure']]
y = data['failure']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(classification_report(y_test, y_pred))
```

## Sentiment Analysis (Text Classification)

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulate text data
texts = [
    'I love this product!', 'Worst experience ever.', 'Very happy with the service.',
    'Not satisfied.', 'Absolutely fantastic!', 'Terrible, will not buy again.'
]
labels = [1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('Sentiment Analysis Accuracy:', accuracy_score(y_test, y_pred))
```

## Recommendation Systems (Collaborative Filtering)

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Simulate user-item ratings matrix
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

# Compute item-item similarity
item_similarity = cosine_similarity(ratings.T)
print('Item-Item Similarity Matrix:\n', item_similarity)

# Recommend items for user 0 (who hasn't rated item 2)
user_idx = 0
unrated_items = np.where(ratings[user_idx] == 0)[0]
for item in unrated_items:
    # Score: weighted sum of similarities with items the user has rated
    rated_items = np.where(ratings[user_idx] > 0)[0]
    score = np.dot(item_similarity[item, rated_items], ratings[user_idx, rated_items]) / (np.sum(item_similarity[item, rated_items]) + 1e-8)
    print(f'Recommendation score for item {item}: {score:.2f}')
```

## Summary

- scikit-learn can be used for a wide range of real-world applications.
- Adapt these templates to your own datasets and business problems. 