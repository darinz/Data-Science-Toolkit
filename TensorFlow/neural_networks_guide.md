# Neural Networks with TensorFlow and Keras

A comprehensive guide to building, training, and deploying neural networks using TensorFlow and Keras.

## Table of Contents

1. [Introduction to Keras](#introduction-to-keras)
2. [Building Neural Networks](#building-neural-networks)
3. [Model Compilation](#model-compilation)
4. [Training Neural Networks](#training-neural-networks)
5. [Evaluation and Prediction](#evaluation-and-prediction)
6. [Custom Layers](#custom-layers)
7. [Model Saving and Loading](#model-saving-and-loading)
8. [Advanced Training Techniques](#advanced-training-techniques)

## Introduction to Keras

Keras is a high-level neural network API that runs on top of TensorFlow. It provides a user-friendly interface for building deep learning models.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Check Keras version
print(f"Keras version: {keras.__version__}")

# Verify TensorFlow backend
print(f"Backend: {keras.backend.backend()}")
```

## Building Neural Networks

### Sequential Model

The Sequential model is the simplest way to build a neural network in Keras.

```python
# Simple sequential model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Alternative way to build sequential model
model = keras.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(784,)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))

# Model summary
model.summary()
```

### Functional API

The Functional API provides more flexibility for complex architectures.

```python
# Functional API example
inputs = keras.Input(shape=(784,))
x = layers.Dense(128, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
```

### Layer Types

```python
# Dense (Fully Connected) Layer
dense_layer = layers.Dense(units=64, activation='relu')

# Convolutional Layer
conv_layer = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')

# Recurrent Layer
lstm_layer = layers.LSTM(units=50, return_sequences=True)

# Pooling Layer
pool_layer = layers.MaxPooling2D(pool_size=(2, 2))

# Normalization Layer
norm_layer = layers.BatchNormalization()

# Regularization Layer
dropout_layer = layers.Dropout(rate=0.5)
```

## Model Compilation

Before training, you need to compile the model with an optimizer, loss function, and metrics.

```python
# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Alternative compilation with custom parameters
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)
```

### Optimizers

```python
# Different optimizers
optimizers = {
    'sgd': keras.optimizers.SGD(learning_rate=0.01),
    'adam': keras.optimizers.Adam(learning_rate=0.001),
    'rmsprop': keras.optimizers.RMSprop(learning_rate=0.001),
    'adagrad': keras.optimizers.Adagrad(learning_rate=0.01)
}

# Custom optimizer with momentum
custom_sgd = keras.optimizers.SGD(
    learning_rate=0.01,
    momentum=0.9,
    nesterov=True
)
```

### Loss Functions

```python
# Common loss functions
loss_functions = {
    'binary_crossentropy': keras.losses.BinaryCrossentropy(),
    'categorical_crossentropy': keras.losses.CategoricalCrossentropy(),
    'sparse_categorical_crossentropy': keras.losses.SparseCategoricalCrossentropy(),
    'mse': keras.losses.MeanSquaredError(),
    'mae': keras.losses.MeanAbsoluteError()
}

# Custom loss function
def custom_loss(y_true, y_pred):
    return keras.backend.mean(keras.backend.square(y_true - y_pred))
```

## Training Neural Networks

### Basic Training

```python
# Generate sample data
np.random.seed(42)
X_train = np.random.random((1000, 784))
y_train = np.random.randint(0, 10, (1000,))
X_val = np.random.random((200, 784))
y_val = np.random.randint(0, 10, (200,))

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_val, y_val),
    verbose=1
)
```

### Training with Callbacks

```python
# Define callbacks
callbacks = [
    # Early stopping
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    ),
    
    # Learning rate reduction
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    ),
    
    # Model checkpointing
    keras.callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),
    
    # TensorBoard logging
    keras.callbacks.TensorBoard(log_dir='./logs')
]

# Train with callbacks
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)
```

### Training Visualization

```python
# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# Plot the training history
plot_training_history(history)
```

## Evaluation and Prediction

```python
# Generate test data
X_test = np.random.random((100, 784))
y_test = np.random.randint(0, 10, (100,))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Make predictions
predictions = model.predict(X_test)
print(f"Predictions shape: {predictions.shape}")

# Get predicted classes
predicted_classes = np.argmax(predictions, axis=1)
print(f"Predicted classes: {predicted_classes[:10]}")

# Compare with true labels
print(f"True labels: {y_test[:10]}")
print(f"Accuracy: {np.mean(predicted_classes == y_test):.4f}")
```

## Custom Layers

You can create custom layers by subclassing `keras.layers.Layer`.

```python
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

# Use custom layer
custom_model = keras.Sequential([
    CustomDenseLayer(64, activation='relu', input_shape=(784,)),
    CustomDenseLayer(32, activation='relu'),
    CustomDenseLayer(10, activation='softmax')
])

custom_model.summary()
```

## Model Saving and Loading

```python
# Save the entire model
model.save('my_model.h5')

# Save only the weights
model.save_weights('my_model_weights.h5')

# Load the entire model
loaded_model = keras.models.load_model('my_model.h5')

# Load only the weights
model.load_weights('my_model_weights.h5')

# Save model architecture as JSON
model_json = model.to_json()
with open('model_architecture.json', 'w') as f:
    f.write(model_json)

# Load model architecture from JSON
with open('model_architecture.json', 'r') as f:
    loaded_model_json = f.read()
loaded_model = keras.models.model_from_json(loaded_model_json)
```

## Advanced Training Techniques

### Transfer Learning

```python
# Load pre-trained model
base_model = keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model layers
base_model.trainable = False

# Add custom classification head
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

### Custom Training Loop

```python
# Custom training loop
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_object(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(y, predictions)

# Training loop
for epoch in range(10):
    for x_batch, y_batch in train_dataset:
        train_step(x_batch, y_batch)
    
    print(f'Epoch {epoch + 1}, Loss: {train_loss.result():.4f}, Accuracy: {train_accuracy.result():.4f}')
    
    # Reset metrics
    train_loss.reset_states()
    train_accuracy.reset_states()
```

### Data Augmentation

```python
# Data augmentation layers
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# Apply to model
model = keras.Sequential([
    data_augmentation,
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

## Summary

- Keras provides a high-level API for building neural networks
- Sequential and Functional APIs offer different levels of flexibility
- Proper model compilation with optimizer, loss, and metrics is essential
- Callbacks enable advanced training features like early stopping and checkpointing
- Custom layers and training loops provide maximum flexibility
- Transfer learning and data augmentation are powerful techniques for improving performance

## Next Steps

- Explore convolutional neural networks for image processing
- Learn about recurrent neural networks for sequential data
- Practice with real-world datasets and transfer learning
- Implement advanced architectures like transformers and GANs 