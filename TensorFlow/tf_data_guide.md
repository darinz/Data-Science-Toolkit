# TensorFlow Data Pipelines Guide

A comprehensive guide to building efficient data pipelines using TensorFlow's tf.data API for optimal model training performance.

## Table of Contents

1. [Introduction to tf.data](#introduction-to-tfdata)
2. [Basic Data Pipeline Operations](#basic-data-pipeline-operations)
3. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
4. [Performance Optimization](#performance-optimization)
5. [Custom Data Generators](#custom-data-generators)
6. [Data Augmentation](#data-augmentation)
7. [Multi-GPU and Distributed Training](#multi-gpu-and-distributed-training)
8. [Real-World Data Pipeline Examples](#real-world-data-pipeline-examples)

## Introduction to tf.data

TensorFlow's tf.data API provides a powerful and flexible way to build efficient data pipelines. It enables you to handle large datasets, perform complex transformations, and optimize data loading for training.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Verify GPU availability
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

## Basic Data Pipeline Operations

### Creating Datasets

```python
# Create dataset from tensors
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
print(f"Dataset: {dataset}")

# Create dataset from numpy arrays
data = np.array([[1, 2], [3, 4], [5, 6]])
labels = np.array([0, 1, 0])
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
print(f"Dataset with features and labels: {dataset}")

# Create dataset from Python lists
features = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
labels = [0, 1, 0]
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
print(f"Dataset from lists: {dataset}")

# Iterate through dataset
for feature, label in dataset:
    print(f"Feature: {feature}, Label: {label}")
```

### Basic Transformations

```python
# Map transformation
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])

# Square each element
squared_dataset = dataset.map(lambda x: x * x)
print("Squared dataset:")
for element in squared_dataset:
    print(element.numpy())

# Filter transformation
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Keep only even numbers
even_dataset = dataset.filter(lambda x: x % 2 == 0)
print("Even numbers:")
for element in even_dataset:
    print(element.numpy())

# Batch transformation
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
batched_dataset = dataset.batch(3)
print("Batched dataset:")
for batch in batched_dataset:
    print(batch.numpy())

# Shuffle transformation
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
shuffled_dataset = dataset.shuffle(buffer_size=10, seed=42)
print("Shuffled dataset:")
for element in shuffled_dataset:
    print(element.numpy())
```

### Combining Operations

```python
# Create a comprehensive pipeline
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Apply multiple transformations
pipeline = dataset.map(lambda x: x * 2)  # Double each element
pipeline = pipeline.filter(lambda x: x > 10)  # Keep elements > 10
pipeline = pipeline.shuffle(buffer_size=5)  # Shuffle
pipeline = pipeline.batch(2)  # Create batches of 2

print("Pipeline result:")
for batch in pipeline:
    print(batch.numpy())

# Chain operations (more Pythonic)
pipeline = (dataset
           .map(lambda x: x * 2)
           .filter(lambda x: x > 10)
           .shuffle(buffer_size=5)
           .batch(2))

print("Chained pipeline result:")
for batch in pipeline:
    print(batch.numpy())
```

## Data Loading and Preprocessing

### Loading from Files

```python
# Create sample CSV file
import pandas as pd

# Generate sample data
data = {
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'feature3': np.random.randn(100),
    'label': np.random.randint(0, 2, 100)
}

df = pd.DataFrame(data)
df.to_csv('sample_data.csv', index=False)

# Load CSV file with tf.data
def load_csv_dataset(file_path, batch_size=32):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=batch_size,
        label_name='label',
        num_epochs=1,
        ignore_errors=True
    )
    return dataset

# Load the dataset
csv_dataset = load_csv_dataset('sample_data.csv')
print("CSV dataset:")
for batch_features, batch_labels in csv_dataset.take(1):
    print(f"Features: {batch_features}")
    print(f"Labels: {batch_labels}")
```

### Image Data Loading

```python
# Load and preprocess images
def load_and_preprocess_image(image_path):
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Resize image
    image = tf.image.resize(image, [224, 224])
    
    # Normalize pixel values
    image = tf.cast(image, tf.float32) / 255.0
    
    return image

# Create image dataset from file paths
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
labels = [0, 1, 0]

# Create dataset
image_dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

# Apply preprocessing
image_dataset = image_dataset.map(
    lambda path, label: (load_and_preprocess_image(path), label)
)

print("Image dataset:")
for image, label in image_dataset.take(1):
    print(f"Image shape: {image.shape}")
    print(f"Label: {label}")
```

### Text Data Loading

```python
# Text preprocessing function
def preprocess_text(text, max_length=100):
    # Convert to lowercase
    text = tf.strings.lower(text)
    
    # Remove punctuation
    text = tf.strings.regex_replace(text, '[^a-zA-Z0-9\s]', '')
    
    # Split into words
    words = tf.strings.split(text)
    
    # Pad or truncate to max_length
    words = words[:max_length]
    words = tf.pad(words, [[0, max_length - tf.shape(words)[0]]])
    
    return words

# Sample text data
texts = [
    "This is a sample text for preprocessing.",
    "Another example of text data.",
    "TensorFlow data pipelines are powerful."
]
labels = [0, 1, 0]

# Create text dataset
text_dataset = tf.data.Dataset.from_tensor_slices((texts, labels))

# Apply preprocessing
text_dataset = text_dataset.map(
    lambda text, label: (preprocess_text(text), label)
)

print("Text dataset:")
for text, label in text_dataset.take(1):
    print(f"Processed text: {text}")
    print(f"Label: {label}")
```

## Performance Optimization

### Prefetching

```python
# Dataset without prefetching
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
dataset = dataset.map(lambda x: x * 2)
dataset = dataset.batch(2)

# Add prefetching for better performance
dataset = dataset.prefetch(tf.data.AUTOTUNE)

print("Dataset with prefetching:")
for batch in dataset:
    print(batch.numpy())
```

### Caching

```python
# Cache dataset to avoid repeated preprocessing
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])

# Apply expensive preprocessing
dataset = dataset.map(lambda x: x * x + tf.math.sqrt(tf.cast(x, tf.float32)))

# Cache the result
dataset = dataset.cache()

# Apply additional transformations
dataset = dataset.shuffle(5).batch(2)

print("Cached dataset:")
for batch in dataset:
    print(batch.numpy())
```

### Parallel Processing

```python
# Parallel map operations
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Apply map with parallel processing
dataset = dataset.map(
    lambda x: x * x,
    num_parallel_calls=tf.data.AUTOTUNE
)

# Apply filter with parallel processing
dataset = dataset.filter(
    lambda x: x > 10,
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset = dataset.batch(2)

print("Parallel processed dataset:")
for batch in dataset:
    print(batch.numpy())
```

### Memory Optimization

```python
# Use interleave for memory-efficient processing
def load_file(file_path):
    # Simulate file loading
    return tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])

# List of file paths
file_paths = ['file1.txt', 'file2.txt', 'file3.txt']

# Interleave datasets for memory efficiency
dataset = tf.data.Dataset.from_tensor_slices(file_paths)
dataset = dataset.interleave(
    load_file,
    cycle_length=2,  # Number of files to process in parallel
    block_length=3,  # Number of consecutive elements from each file
    num_parallel_calls=tf.data.AUTOTUNE
)

print("Interleaved dataset:")
for element in dataset:
    print(element.numpy())
```

## Custom Data Generators

### Custom Generator Function

```python
# Custom generator function
def custom_generator():
    for i in range(10):
        # Generate features
        features = np.random.randn(5)
        # Generate label
        label = np.random.randint(0, 2)
        yield features, label

# Create dataset from generator
dataset = tf.data.Dataset.from_generator(
    custom_generator,
    output_signature=(
        tf.TensorSpec(shape=(5,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)

# Apply transformations
dataset = dataset.batch(2).prefetch(tf.data.AUTOTUNE)

print("Custom generator dataset:")
for batch_features, batch_labels in dataset:
    print(f"Features: {batch_features}")
    print(f"Labels: {batch_labels}")
```

### Custom Layer for Data Processing

```python
# Custom preprocessing layer
class CustomPreprocessingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomPreprocessingLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        # Apply custom preprocessing
        normalized = tf.cast(inputs, tf.float32) / 255.0
        return normalized

# Use in data pipeline
def preprocess_with_custom_layer(data, label):
    preprocessor = CustomPreprocessingLayer()
    processed_data = preprocessor(data)
    return processed_data, label

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((
    np.random.randint(0, 256, (100, 28, 28, 3)),
    np.random.randint(0, 10, 100)
))

# Apply custom preprocessing
dataset = dataset.map(preprocess_with_custom_layer)
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)

print("Dataset with custom preprocessing:")
for batch_data, batch_labels in dataset.take(1):
    print(f"Data shape: {batch_data.shape}")
    print(f"Data range: [{tf.reduce_min(batch_data)}, {tf.reduce_max(batch_data)}]")
    print(f"Labels: {batch_labels}")
```

## Data Augmentation

### Image Augmentation

```python
# Image augmentation functions
def augment_image(image, label):
    # Random flip
    image = tf.image.random_flip_left_right(image)
    
    # Random rotation
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    
    # Random brightness
    image = tf.image.random_brightness(image, 0.2)
    
    # Random contrast
    image = tf.image.random_contrast(image, 0.8, 1.2)
    
    # Ensure pixel values are in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label

# Create image dataset
image_dataset = tf.data.Dataset.from_tensor_slices((
    np.random.rand(100, 224, 224, 3),
    np.random.randint(0, 10, 100)
))

# Apply augmentation
augmented_dataset = image_dataset.map(
    augment_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

augmented_dataset = augmented_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

print("Augmented image dataset:")
for batch_images, batch_labels in augmented_dataset.take(1):
    print(f"Image batch shape: {batch_images.shape}")
    print(f"Labels: {batch_labels}")
```

### Text Augmentation

```python
# Text augmentation functions
def augment_text(text, label):
    # Random word dropout
    words = tf.strings.split(text)
    
    # Randomly drop some words
    mask = tf.random.uniform(tf.shape(words), 0, 1) > 0.1
    words = tf.boolean_mask(words, mask)
    
    # Rejoin words
    augmented_text = tf.strings.reduce_join(words, separator=' ')
    
    return augmented_text, label

# Create text dataset
texts = [
    "This is a sample text for augmentation",
    "Another example of text data",
    "TensorFlow data pipelines are powerful"
]
labels = [0, 1, 0]

text_dataset = tf.data.Dataset.from_tensor_slices((texts, labels))

# Apply augmentation
augmented_text_dataset = text_dataset.map(
    augment_text,
    num_parallel_calls=tf.data.AUTOTUNE
)

print("Augmented text dataset:")
for text, label in augmented_text_dataset:
    print(f"Text: {text}")
    print(f"Label: {label}")
```

## Multi-GPU and Distributed Training

### Multi-GPU Data Pipeline

```python
# Multi-GPU data pipeline
def create_multi_gpu_pipeline(batch_size_per_replica, num_replicas=2):
    # Create base dataset
    dataset = tf.data.Dataset.from_tensor_slices((
        np.random.randn(1000, 784),
        np.random.randint(0, 10, 1000)
    ))
    
    # Apply preprocessing
    dataset = dataset.map(
        lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Shuffle and batch
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size_per_replica * num_replicas)
    
    # Prefetch
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Create multi-GPU pipeline
multi_gpu_dataset = create_multi_gpu_pipeline(batch_size_per_replica=32, num_replicas=2)

print("Multi-GPU dataset:")
for batch_features, batch_labels in multi_gpu_dataset.take(1):
    print(f"Features shape: {batch_features.shape}")
    print(f"Labels shape: {batch_labels.shape}")
```

### Distributed Training Pipeline

```python
# Distributed training pipeline
def create_distributed_pipeline(strategy, batch_size_per_replica):
    with strategy.scope():
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            np.random.randn(1000, 784),
            np.random.randint(0, 10, 1000)
        ))
        
        # Apply preprocessing
        dataset = dataset.map(
            lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Shuffle and batch
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size_per_replica)
        
        # Prefetch
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Distribute dataset
        distributed_dataset = strategy.experimental_distribute_dataset(dataset)
        
        return distributed_dataset

# Create distributed strategy
strategy = tf.distribute.MirroredStrategy()

# Create distributed pipeline
distributed_dataset = create_distributed_pipeline(strategy, batch_size_per_replica=32)

print("Distributed dataset created with strategy:", strategy)
```

## Real-World Data Pipeline Examples

### Image Classification Pipeline

```python
# Complete image classification pipeline
def create_image_classification_pipeline(image_paths, labels, batch_size=32):
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    
    # Load and preprocess images
    def load_and_preprocess_image(path, label):
        # Load image
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        
        # Resize
        image = tf.image.resize(image, [224, 224])
        
        # Normalize
        image = tf.cast(image, tf.float32) / 255.0
        
        return image, label
    
    # Apply preprocessing
    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Apply augmentation (for training)
    def augment_for_training(image, label):
        # Random flip
        image = tf.image.random_flip_left_right(image)
        
        # Random rotation
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        
        # Random brightness and contrast
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # Ensure pixel values are in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    dataset = dataset.map(
        augment_for_training,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Shuffle, batch, and prefetch
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Example usage
image_paths = ['path/to/image1.jpg', 'path/to/image2.jpg', 'path/to/image3.jpg']
labels = [0, 1, 0]

image_pipeline = create_image_classification_pipeline(image_paths, labels)
print("Image classification pipeline created")
```

### Text Classification Pipeline

```python
# Complete text classification pipeline
def create_text_classification_pipeline(texts, labels, vocab_size=1000, max_length=100, batch_size=32):
    # Tokenize texts
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_length, padding='post', truncating='post'
    )
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((padded_sequences, labels))
    
    # Apply preprocessing
    def preprocess_text(sequence, label):
        # Convert to float32
        sequence = tf.cast(sequence, tf.float32)
        label = tf.cast(label, tf.int32)
        return sequence, label
    
    dataset = dataset.map(
        preprocess_text,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Shuffle, batch, and prefetch
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, tokenizer

# Example usage
texts = [
    "I love this movie, it's amazing!",
    "This is the worst film I've ever seen.",
    "Great acting and wonderful story."
]
labels = [1, 0, 1]

text_pipeline, tokenizer = create_text_classification_pipeline(texts, labels)
print("Text classification pipeline created")
```

### Time Series Pipeline

```python
# Complete time series pipeline
def create_timeseries_pipeline(data, labels, sequence_length=50, batch_size=32):
    # Create sliding windows
    def create_sequences(data, labels, sequence_length):
        sequences = []
        sequence_labels = []
        
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length])
            sequence_labels.append(labels[i + sequence_length])
        
        return np.array(sequences), np.array(sequence_labels)
    
    # Generate sequences
    X, y = create_sequences(data, labels, sequence_length)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Apply preprocessing
    def preprocess_timeseries(sequence, label):
        # Normalize sequence
        sequence = tf.cast(sequence, tf.float32)
        sequence = (sequence - tf.reduce_mean(sequence)) / tf.math.reduce_std(sequence)
        
        # Convert label
        label = tf.cast(label, tf.float32)
        
        return sequence, label
    
    dataset = dataset.map(
        preprocess_timeseries,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Shuffle, batch, and prefetch
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Example usage
# Generate sample time series data
timeseries_data = np.random.randn(1000, 10)  # 1000 timesteps, 10 features
timeseries_labels = np.random.randn(1000)

timeseries_pipeline = create_timeseries_pipeline(timeseries_data, timeseries_labels)
print("Time series pipeline created")
```

## Summary

- tf.data provides efficient data pipeline operations
- Use prefetching, caching, and parallel processing for performance
- Custom generators and layers enable flexible data processing
- Data augmentation improves model generalization
- Multi-GPU and distributed pipelines scale training
- Real-world pipelines combine multiple operations for specific tasks

## Next Steps

- Explore TensorFlow Extended (TFX) for production pipelines
- Learn about data validation and monitoring
- Practice with large-scale datasets
- Implement custom data formats and sources 