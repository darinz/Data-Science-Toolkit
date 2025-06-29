# Convolutional Neural Networks (CNNs) with TensorFlow

A comprehensive guide to building and training convolutional neural networks for computer vision tasks using TensorFlow and Keras.

## Table of Contents

1. [Introduction to CNNs](#introduction-to-cnns)
2. [CNN Architecture Components](#cnn-architecture-components)
3. [Building CNNs with Keras](#building-cnns-with-keras)
4. [Image Preprocessing](#image-preprocessing)
5. [Training CNNs](#training-cnns)
6. [Transfer Learning](#transfer-learning)
7. [Data Augmentation](#data-augmentation)
8. [Model Evaluation](#model-evaluation)
9. [Advanced CNN Architectures](#advanced-cnn-architectures)

## Introduction to CNNs

Convolutional Neural Networks are specialized neural networks designed for processing grid-like data, particularly images. They use convolutional layers to automatically learn spatial hierarchies of features.

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Verify GPU availability for CNN training
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

## CNN Architecture Components

### Convolutional Layers

Convolutional layers apply filters to input images to detect features like edges, textures, and patterns.

```python
# Basic convolutional layer
conv_layer = layers.Conv2D(
    filters=32,           # Number of filters
    kernel_size=(3, 3),   # Filter size
    activation='relu',    # Activation function
    input_shape=(28, 28, 1)  # Input shape (height, width, channels)
)

# Convolutional layer with padding
conv_with_padding = layers.Conv2D(
    filters=64,
    kernel_size=(5, 5),
    padding='same',       # 'same' or 'valid'
    activation='relu'
)

# Convolutional layer with stride
conv_with_stride = layers.Conv2D(
    filters=128,
    kernel_size=(3, 3),
    strides=(2, 2),       # Reduce spatial dimensions
    activation='relu'
)
```

### Pooling Layers

Pooling layers reduce spatial dimensions and provide translation invariance.

```python
# Max pooling
max_pool = layers.MaxPooling2D(
    pool_size=(2, 2),     # Pool size
    strides=(2, 2)        # Stride (optional)
)

# Average pooling
avg_pool = layers.AveragePooling2D(
    pool_size=(2, 2)
)

# Global pooling (reduces to 1D)
global_avg_pool = layers.GlobalAveragePooling2D()
global_max_pool = layers.GlobalMaxPooling2D()
```

### Normalization and Regularization

```python
# Batch normalization
batch_norm = layers.BatchNormalization()

# Dropout for regularization
dropout = layers.Dropout(rate=0.5)

# Spatial dropout (for CNNs)
spatial_dropout = layers.SpatialDropout2D(rate=0.25)
```

## Building CNNs with Keras

### Simple CNN Architecture

```python
# Simple CNN for MNIST digit classification
model = keras.Sequential([
    # Input layer
    layers.Input(shape=(28, 28, 1)),
    
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Third convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    
    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.summary()
```

### Advanced CNN Architecture

```python
# More sophisticated CNN architecture
def create_advanced_cnn(input_shape, num_classes):
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First block
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second block
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third block
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Classification head
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create model
advanced_model = create_advanced_cnn((32, 32, 3), 10)
advanced_model.summary()
```

## Image Preprocessing

### Data Loading and Preprocessing

```python
# Load and preprocess MNIST dataset
def load_and_preprocess_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for CNN (add channel dimension)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

# Load CIFAR-10 dataset
def load_and_preprocess_cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

# Load datasets
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = load_and_preprocess_mnist()
(x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = load_and_preprocess_cifar10()

print(f"MNIST shapes: {x_train_mnist.shape}, {y_train_mnist.shape}")
print(f"CIFAR-10 shapes: {x_train_cifar.shape}, {y_train_cifar.shape}")
```

### Custom Data Generator

```python
# Custom data generator for image preprocessing
class ImageDataGenerator:
    def __init__(self, rotation_range=0, width_shift_range=0, height_shift_range=0,
                 zoom_range=0, horizontal_flip=False, vertical_flip=False):
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
    
    def augment_image(self, image):
        # Random rotation
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            image = tf.image.rot90(image, k=int(angle // 90))
        
        # Random shifts
        if self.width_shift_range > 0 or self.height_shift_range > 0:
            height, width = image.shape[:2]
            shift_x = int(np.random.uniform(-self.width_shift_range, self.width_shift_range) * width)
            shift_y = int(np.random.uniform(-self.height_shift_range, self.height_shift_range) * height)
            image = tf.roll(image, shift=[shift_y, shift_x], axis=[0, 1])
        
        # Random zoom
        if self.zoom_range > 0:
            scale = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range)
            new_height = int(image.shape[0] * scale)
            new_width = int(image.shape[1] * scale)
            image = tf.image.resize(image, [new_height, new_width])
            image = tf.image.resize_with_crop_or_pad(image, image.shape[0], image.shape[1])
        
        # Random flips
        if self.horizontal_flip and np.random.random() > 0.5:
            image = tf.image.flip_left_right(image)
        
        if self.vertical_flip and np.random.random() > 0.5:
            image = tf.image.flip_up_down(image)
        
        return image

# Create data generator
data_gen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)
```

## Training CNNs

### Basic Training

```python
# Compile and train simple CNN
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    x_train_mnist, y_train_mnist,
    batch_size=32,
    epochs=10,
    validation_split=0.2,
    verbose=1
)
```

### Advanced Training with Callbacks

```python
# Define callbacks
callbacks = [
    # Early stopping
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    
    # Learning rate scheduling
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7
    ),
    
    # Model checkpointing
    keras.callbacks.ModelCheckpoint(
        'best_cnn_model.h5',
        monitor='val_accuracy',
        save_best_only=True
    ),
    
    # TensorBoard logging
    keras.callbacks.TensorBoard(log_dir='./logs/cnn')
]

# Train advanced model
advanced_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = advanced_model.fit(
    x_train_cifar, y_train_cifar,
    batch_size=64,
    epochs=50,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)
```

## Transfer Learning

### Using Pre-trained Models

```python
# Load pre-trained VGG16 model
base_model = keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
base_model.trainable = False

# Create transfer learning model
transfer_model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

transfer_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

transfer_model.summary()
```

### Fine-tuning Pre-trained Models

```python
# Fine-tuning approach
def create_fine_tuned_model(num_classes):
    # Load pre-trained model
    base_model = keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze early layers
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    # Create model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create fine-tuned model
fine_tuned_model = create_fine_tuned_model(10)
fine_tuned_model.summary()
```

## Data Augmentation

### Built-in Data Augmentation

```python
# Data augmentation layers
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
    layers.RandomTranslation(0.1, 0.1),
])

# Model with data augmentation
augmented_model = keras.Sequential([
    data_augmentation,
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### Custom Data Augmentation

```python
# Custom augmentation function
def augment_image(image, label):
    # Random brightness
    image = tf.image.random_brightness(image, 0.2)
    
    # Random contrast
    image = tf.image.random_contrast(image, 0.8, 1.2)
    
    # Random saturation
    image = tf.image.random_saturation(image, 0.8, 1.2)
    
    # Random hue
    image = tf.image.random_hue(image, 0.1)
    
    # Ensure pixel values are in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label

# Create dataset with augmentation
def create_augmented_dataset(images, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# Create augmented datasets
train_dataset = create_augmented_dataset(x_train_cifar, y_train_cifar)
val_dataset = tf.data.Dataset.from_tensor_slices((x_test_cifar, y_test_cifar)).batch(32)
```

## Model Evaluation

### Performance Metrics

```python
# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test_mnist, y_test_mnist, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Make predictions
predictions = model.predict(x_test_mnist)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test_mnist, axis=1)

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification report
print(classification_report(true_classes, predicted_classes))
```

### Visualization Functions

```python
# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Visualize feature maps
def visualize_feature_maps(model, image, layer_name):
    # Create a model that outputs the specified layer
    layer_output = model.get_layer(layer_name).output
    feature_model = keras.Model(inputs=model.input, outputs=layer_output)
    
    # Get feature maps
    feature_maps = feature_model.predict(np.expand_dims(image, axis=0))
    
    # Plot first 16 feature maps
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i in range(16):
        row, col = i // 4, i % 4
        axes[row, col].imshow(feature_maps[0, :, :, i], cmap='viridis')
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Feature Map {i+1}')
    
    plt.tight_layout()
    plt.show()

# Plot training history
plot_training_history(history)
```

## Advanced CNN Architectures

### Residual Networks (ResNet)

```python
# Residual block
def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x
    
    # First convolution
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Second convolution
    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Add shortcut connection
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    
    return x

# Simple ResNet
def create_simple_resnet(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, stride=2)
    x = residual_block(x, 128)
    x = residual_block(x, 256, stride=2)
    x = residual_block(x, 256)
    
    # Global pooling and classification
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, x)

# Create ResNet model
resnet_model = create_simple_resnet((32, 32, 3), 10)
resnet_model.summary()
```

### DenseNet-like Architecture

```python
# Dense block
def dense_block(x, blocks, growth_rate):
    for i in range(blocks):
        # Bottleneck layer
        bottleneck = layers.BatchNormalization()(x)
        bottleneck = layers.ReLU()(bottleneck)
        bottleneck = layers.Conv2D(4 * growth_rate, 1, padding='same')(bottleneck)
        
        # Dense layer
        dense = layers.BatchNormalization()(bottleneck)
        dense = layers.ReLU()(dense)
        dense = layers.Conv2D(growth_rate, 3, padding='same')(dense)
        
        # Concatenate
        x = layers.Concatenate()([x, dense])
    
    return x

# Transition block
def transition_block(x, reduction):
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    in_channels = x.shape[-1]
    out_channels = int(in_channels * reduction)
    
    x = layers.Conv2D(out_channels, 1, padding='same')(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    
    return x

# DenseNet-like model
def create_densenet_like(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Dense blocks
    x = dense_block(x, 6, 32)
    x = transition_block(x, 0.5)
    x = dense_block(x, 12, 32)
    x = transition_block(x, 0.5)
    x = dense_block(x, 24, 32)
    x = transition_block(x, 0.5)
    x = dense_block(x, 16, 32)
    
    # Global pooling and classification
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    return keras.Model(inputs, x)

# Create DenseNet-like model
densenet_model = create_densenet_like((32, 32, 3), 10)
densenet_model.summary()
```

## Summary

- CNNs are specialized for processing grid-like data, particularly images
- Convolutional layers learn spatial hierarchies of features automatically
- Pooling layers reduce spatial dimensions and provide translation invariance
- Transfer learning leverages pre-trained models for new tasks
- Data augmentation improves model generalization
- Advanced architectures like ResNet and DenseNet provide better performance

## Next Steps

- Explore object detection with CNNs
- Learn about semantic segmentation
- Practice with real-world image datasets
- Implement attention mechanisms in CNNs 