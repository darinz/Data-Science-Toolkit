# Advanced TensorFlow Techniques Guide

A comprehensive guide to advanced TensorFlow techniques including custom training loops, model deployment, optimization, and production-ready implementations.

## Table of Contents

1. [Custom Training Loops](#custom-training-loops)
2. [Model Optimization](#model-optimization)
3. [Model Deployment](#model-deployment)
4. [TensorFlow Serving](#tensorflow-serving)
5. [Model Compression](#model-compression)
6. [Advanced Architectures](#advanced-architectures)
7. [Performance Profiling](#performance-profiling)
8. [Production Best Practices](#production-best-practices)

## Custom Training Loops

### Basic Custom Training Loop

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Create a simple model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Define loss function and optimizer
loss_fn = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

# Define metrics
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

# Generate sample data
X_train = np.random.random((1000, 784))
y_train = np.random.randint(0, 10, (1000,))
X_val = np.random.random((200, 784))
y_val = np.random.randint(0, 10, (200,))

# Custom training step
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
    
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    
    train_acc_metric.update_state(y, logits)
    return loss_value

# Custom validation step
@tf.function
def val_step(x, y):
    val_logits = model(x, training=False)
    val_acc_metric.update_state(y, val_logits)

# Training loop
epochs = 10
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    # Training
    for step, (x_batch, y_batch) in enumerate(zip(X_train, y_train)):
        if step % 100 == 0:
            loss = train_step(x_batch, y_batch)
            print(f"Step {step}: loss = {loss:.4f}, accuracy = {train_acc_metric.result():.4f}")
    
    # Validation
    for x_batch, y_batch in zip(X_val, y_val):
        val_step(x_batch, y_batch)
    
    print(f"Training accuracy: {train_acc_metric.result():.4f}")
    print(f"Validation accuracy: {val_acc_metric.result():.4f}")
    
    # Reset metrics
    train_acc_metric.reset_states()
    val_acc_metric.reset_states()
```

### Advanced Custom Training Loop

```python
# Advanced custom training loop with callbacks
class CustomTrainingLoop:
    def __init__(self, model, loss_fn, optimizer, metrics=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics or []
        self.train_loss = keras.metrics.Mean()
        self.val_loss = keras.metrics.Mean()
    
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.loss_fn(y, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss.update_state(loss)
        for metric in self.metrics:
            metric.update_state(y, predictions)
        
        return loss
    
    @tf.function
    def val_step(self, x, y):
        predictions = self.model(x, training=False)
        loss = self.loss_fn(y, predictions)
        
        self.val_loss.update_state(loss)
        for metric in self.metrics:
            metric.update_state(y, predictions)
        
        return loss
    
    def train(self, train_dataset, val_dataset, epochs, callbacks=None):
        callbacks = callbacks or []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            for x_batch, y_batch in train_dataset:
                self.train_step(x_batch, y_batch)
            
            # Validation
            for x_batch, y_batch in val_dataset:
                self.val_step(x_batch, y_batch)
            
            # Log metrics
            print(f"Training loss: {self.train_loss.result():.4f}")
            print(f"Validation loss: {self.val_loss.result():.4f}")
            for metric in self.metrics:
                print(f"{metric.name}: {metric.result():.4f}")
            
            # Reset metrics
            self.train_loss.reset_states()
            self.val_loss.reset_states()
            for metric in self.metrics:
                metric.reset_states()
            
            # Execute callbacks
            for callback in callbacks:
                callback.on_epoch_end(epoch)

# Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

# Initialize training loop
training_loop = CustomTrainingLoop(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

# Train the model
training_loop.train(train_dataset, val_dataset, epochs=5)
```

## Model Optimization

### Mixed Precision Training

```python
# Enable mixed precision
policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)

print(f"Compute dtype: {policy.compute_dtype}")
print(f"Variable dtype: {policy.variable_dtype}")

# Create model with mixed precision
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile with mixed precision
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with mixed precision
model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))
```

### Model Quantization

```python
# Post-training quantization
def representative_dataset():
    for i in range(100):
        yield [np.random.random((1, 784)).astype(np.float32)]

# Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

# Save quantized model
with open('quantized_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Quantized model saved as quantized_model.tflite")
```

### Model Pruning

```python
# Model pruning
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

# Apply pruning to the model
model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

# Compile the pruned model
model_for_pruning.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with pruning
model_for_pruning.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

# Strip pruning wrapper
final_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
```

## Model Deployment

### Model Export

```python
# Export model in SavedModel format
model.save('saved_model')

# Export with custom signatures
class CustomModel(keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.dense = layers.Dense(10, activation='softmax')
    
    def call(self, inputs):
        return self.dense(inputs)
    
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 784], dtype=tf.float32)])
    def predict(self, inputs):
        return self.call(inputs)

custom_model = CustomModel()
custom_model.build((None, 784))

# Save with custom signature
tf.saved_model.save(custom_model, 'custom_model', signatures={
    'predict': custom_model.predict
})
```

### TensorFlow Lite Conversion

```python
# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Load and run TFLite model
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference
input_data = np.random.random((1, 784)).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(f"TFLite model output: {output_data}")
```

## TensorFlow Serving

### Model Serving Setup

```python
# Prepare model for serving
model_version = 1
export_path = f'./serving_model/{model_version}'

# Save model for serving
tf.saved_model.save(model, export_path)

print(f"Model saved to {export_path}")

# Model serving configuration
serving_config = {
    'model_config_list': {
        'config': [{
            'name': 'my_model',
            'base_path': './serving_model',
            'model_platform': 'tensorflow'
        }]
    }
}

# Save serving config
import json
with open('serving_config.json', 'w') as f:
    json.dump(serving_config, f)
```

### REST API Client

```python
import requests
import json

# REST API client for TensorFlow Serving
class TFServingClient:
    def __init__(self, base_url='http://localhost:8501'):
        self.base_url = base_url
    
    def predict(self, model_name, data, version=None):
        url = f"{self.base_url}/v1/models/{model_name}"
        if version:
            url += f"/versions/{version}"
        url += ":predict"
        
        payload = {
            'instances': data.tolist()
        }
        
        response = requests.post(url, json=payload)
        return response.json()

# Example usage
client = TFServingClient()
sample_data = np.random.random((1, 784))

try:
    result = client.predict('my_model', sample_data)
    print(f"Prediction result: {result}")
except requests.exceptions.ConnectionError:
    print("TensorFlow Serving not running. Start with: tensorflow_model_server")
```

## Model Compression

### Knowledge Distillation

```python
# Teacher model (larger, more accurate)
teacher_model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

teacher_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train teacher model
teacher_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Student model (smaller, to be distilled)
student_model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(784,)),
    layers.Dense(10, activation='softmax')
])

# Knowledge distillation loss
class DistillationLoss(keras.losses.Loss):
    def __init__(self, temperature=3.0, alpha=0.7, **kwargs):
        super(DistillationLoss, self).__init__(**kwargs)
        self.temperature = temperature
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        # Hard target loss
        hard_loss = keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        
        # Soft target loss (knowledge distillation)
        soft_targets = teacher_model.predict(X_train, verbose=0)
        soft_loss = keras.losses.categorical_crossentropy(
            soft_targets / self.temperature,
            y_pred / self.temperature
        ) * (self.temperature ** 2)
        
        return self.alpha * hard_loss + (1 - self.alpha) * soft_loss

# Compile student model with distillation loss
student_model.compile(
    optimizer='adam',
    loss=DistillationLoss(),
    metrics=['accuracy']
)

# Train student model with knowledge distillation
student_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
```

### Model Clustering

```python
# Model clustering for weight sharing
cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

clustering_params = {
    'number_of_clusters': 16,
    'cluster_centroids_init': CentroidInitialization.LINEAR
}

# Apply clustering
model_for_clustering = cluster_weights(model, **clustering_params)

# Compile clustered model
model_for_clustering.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train clustered model
model_for_clustering.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))

# Strip clustering wrapper
final_clustered_model = tfmot.clustering.keras.strip_clustering(model_for_clustering)
```

## Advanced Architectures

### Custom Layer with Gradient Computation

```python
# Custom layer with manual gradient computation
class CustomDenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(CustomDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)
    
    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
    
    def call(self, inputs):
        output = tf.matmul(inputs, self.kernel) + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output
    
    def get_config(self):
        config = super(CustomDenseLayer, self).get_config()
        config.update({
            'units': self.units,
            'activation': keras.activations.serialize(self.activation)
        })
        return config

# Use custom layer
custom_model = keras.Sequential([
    CustomDenseLayer(64, activation='relu', input_shape=(784,)),
    CustomDenseLayer(32, activation='relu'),
    CustomDenseLayer(10, activation='softmax')
])

custom_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

custom_model.fit(X_train, y_train, epochs=5, validation_data=(X_val, y_val))
```

### Multi-Input Multi-Output Model

```python
# Multi-input multi-output model
input_1 = keras.Input(shape=(784,), name='input_1')
input_2 = keras.Input(shape=(100,), name='input_2')

# Shared layers
shared_dense = layers.Dense(64, activation='relu')
shared_output = shared_dense(input_1)

# Branch 1
branch_1 = layers.Dense(32, activation='relu')(shared_output)
output_1 = layers.Dense(10, activation='softmax', name='output_1')(branch_1)

# Branch 2
branch_2 = layers.Dense(32, activation='relu')(input_2)
output_2 = layers.Dense(1, activation='sigmoid', name='output_2')(branch_2)

# Create model
multi_io_model = keras.Model(
    inputs=[input_1, input_2],
    outputs=[output_1, output_2]
)

# Compile with different losses for each output
multi_io_model.compile(
    optimizer='adam',
    loss={
        'output_1': 'sparse_categorical_crossentropy',
        'output_2': 'binary_crossentropy'
    },
    metrics={
        'output_1': 'accuracy',
        'output_2': 'accuracy'
    }
)

# Generate sample data for multi-input
X1_train = np.random.random((1000, 784))
X2_train = np.random.random((1000, 100))
y1_train = np.random.randint(0, 10, (1000,))
y2_train = np.random.randint(0, 2, (1000,))

# Train multi-input model
multi_io_model.fit(
    [X1_train, X2_train],
    [y1_train, y2_train],
    epochs=5,
    validation_split=0.2
)
```

## Performance Profiling

### Model Profiling

```python
# Model profiling with TensorBoard
import datetime

# Create log directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

# Train with profiling
model.fit(
    X_train, y_train,
    epochs=5,
    validation_data=(X_val, y_val),
    callbacks=[tensorboard_callback]
)

print(f"TensorBoard logs saved to {log_dir}")
print("Run: tensorboard --logdir logs/fit")
```

### Memory Profiling

```python
# Memory profiling
import psutil
import time

def profile_memory(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Function: {func.__name__}")
        print(f"Memory before: {memory_before:.2f} MB")
        print(f"Memory after: {memory_after:.2f} MB")
        print(f"Memory used: {memory_after - memory_before:.2f} MB")
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        
        return result
    return wrapper

@profile_memory
def train_model():
    model.fit(X_train, y_train, epochs=5, verbose=0)

train_model()
```

## Production Best Practices

### Model Versioning

```python
# Model versioning with metadata
import json
import hashlib

class ModelVersioning:
    def __init__(self, model_dir='./models'):
        self.model_dir = model_dir
    
    def save_model_with_metadata(self, model, version, metadata):
        # Create version directory
        version_dir = f"{self.model_dir}/v{version}"
        
        # Save model
        model.save(f"{version_dir}/model")
        
        # Save metadata
        metadata['version'] = version
        metadata['timestamp'] = datetime.datetime.now().isoformat()
        metadata['model_hash'] = self._calculate_model_hash(model)
        
        with open(f"{version_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {version_dir}")
    
    def _calculate_model_hash(self, model):
        # Calculate hash of model weights
        weights_hash = hashlib.md5()
        for layer in model.layers:
            for weight in layer.get_weights():
                weights_hash.update(weight.tobytes())
        return weights_hash.hexdigest()
    
    def load_model_with_metadata(self, version):
        version_dir = f"{self.model_dir}/v{version}"
        
        # Load model
        model = keras.models.load_model(f"{version_dir}/model")
        
        # Load metadata
        with open(f"{version_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        return model, metadata

# Example usage
versioning = ModelVersioning()

metadata = {
    'description': 'MNIST digit classification model',
    'architecture': 'Sequential Dense',
    'training_data': 'MNIST dataset',
    'accuracy': 0.95
}

versioning.save_model_with_metadata(model, 1, metadata)
loaded_model, loaded_metadata = versioning.load_model_with_metadata(1)

print(f"Loaded model metadata: {loaded_metadata}")
```

### Model Monitoring

```python
# Model monitoring and logging
import logging
from datetime import datetime

class ModelMonitor:
    def __init__(self, log_file='model_monitor.log'):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def log_prediction(self, input_data, prediction, confidence, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'input_shape': input_data.shape,
            'prediction': prediction.tolist(),
            'confidence': float(confidence),
            'input_hash': self._hash_input(input_data)
        }
        
        self.logger.info(f"Prediction: {log_entry}")
        return log_entry
    
    def log_model_performance(self, metrics, epoch=None):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'metrics': metrics
        }
        
        self.logger.info(f"Performance: {log_entry}")
        return log_entry
    
    def _hash_input(self, input_data):
        return hashlib.md5(input_data.tobytes()).hexdigest()

# Example usage
monitor = ModelMonitor()

# Log predictions
sample_input = np.random.random((1, 784))
prediction = model.predict(sample_input)
confidence = np.max(prediction)

monitor.log_prediction(sample_input, prediction, confidence)

# Log performance
metrics = {'accuracy': 0.95, 'loss': 0.1}
monitor.log_model_performance(metrics, epoch=1)
```

## Summary

- Custom training loops provide full control over the training process
- Model optimization techniques improve efficiency and reduce size
- TensorFlow Serving enables production deployment
- Model compression reduces computational requirements
- Advanced architectures support complex use cases
- Performance profiling helps identify bottlenecks
- Production best practices ensure reliable deployment

## Next Steps

- Explore TensorFlow Extended (TFX) for ML pipelines
- Learn about distributed training strategies
- Practice with real-world deployment scenarios
- Implement advanced monitoring and alerting systems 