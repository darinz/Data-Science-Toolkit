# Neural Networks Guide

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Level](https://img.shields.io/badge/Level-Advanced-red.svg)](https://github.com/yourusername/Toolkit)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)
[![Keras](https://img.shields.io/badge/Keras-2.8%2B-red.svg)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-orange.svg)](https://numpy.org/)
[![Topics](https://img.shields.io/badge/Topics-Neural%20Networks%2C%20Deep%20Learning%2C%20AI-orange.svg)](https://github.com/yourusername/Toolkit)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)](https://github.com/yourusername/Toolkit)

# Neural Networks Fundamentals

A comprehensive guide to building and training neural networks in Python for deep learning applications.

## Table of Contents
1. [Introduction to Neural Networks](#introduction-to-neural-networks)
2. [Building Neural Networks with TensorFlow/Keras](#building-neural-networks-with-tensorflowkeras)
3. [Building Neural Networks with PyTorch](#building-neural-networks-with-pytorch)
4. [Training Neural Networks](#training-neural-networks)
5. [Optimization Techniques](#optimization-techniques)
6. [Regularization Methods](#regularization-methods)
7. [Model Evaluation](#model-evaluation)
8. [Transfer Learning](#transfer-learning)
9. [Best Practices](#best-practices)

## Introduction to Neural Networks

### What are Neural Networks?

Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information and learn patterns from data.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
```

### Basic Neural Network Components

```python
def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid function"""
    return x * (1 - x)

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU function"""
    return np.where(x > 0, 1, 0)

def softmax(x):
    """Softmax activation function"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Plot activation functions
x = np.linspace(-5, 5, 100)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid Function')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, relu(x))
plt.title('ReLU Function')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, np.tanh(x))
plt.title('Tanh Function')
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Building Neural Networks with TensorFlow/Keras

### Basic Neural Network

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set random seed
tf.random.set_seed(42)

def create_basic_neural_network(input_dim, num_classes=1, task='regression'):
    """
    Create a basic neural network
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        task: 'classification' or 'regression'
    
    Returns:
        Compiled neural network model
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax' if task == 'classification' else 'linear')
    ])
    
    # Compile model
    if task == 'classification':
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
    
    return model

# Create sample data
X_class, y_class = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
X_reg, y_reg = make_regression(n_samples=1000, n_features=20, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train classification model
class_model = create_basic_neural_network(X_train.shape[1], num_classes=3, task='classification')
print(class_model.summary())

# Train model
history = class_model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
```

### Advanced Neural Network with Regularization

```python
def create_advanced_neural_network(input_dim, num_classes=1, task='regression'):
    """
    Create an advanced neural network with regularization
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        task: 'classification' or 'regression'
    
    Returns:
        Compiled neural network model
    """
    model = Sequential([
        # Input layer
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden layers
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        
        # Output layer
        Dense(num_classes, activation='softmax' if task == 'classification' else 'linear')
    ])
    
    # Compile model
    if task == 'classification':
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    else:
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
    
    return model

# Create advanced model
advanced_model = create_advanced_neural_network(X_train.shape[1], num_classes=3, task='classification')

# Define callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
]

# Train advanced model
history_advanced = advanced_model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)
```

### Model Training Visualization

```python
def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Training history from model.fit()
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    if 'accuracy' in history.history:
        axes[1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# Plot training history
plot_training_history(history_advanced)
```

## Building Neural Networks with PyTorch

### Basic PyTorch Neural Network

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seed
torch.manual_seed(42)

class BasicNeuralNetwork(nn.Module):
    """
    Basic neural network using PyTorch
    """
    def __init__(self, input_dim, hidden_dims, num_classes=1, task='regression'):
        super(BasicNeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        if task == 'classification':
            layers.append(nn.Linear(prev_dim, num_classes))
            layers.append(nn.Softmax(dim=1))
        else:
            layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Create PyTorch model
input_dim = X_train.shape[1]
hidden_dims = [64, 32, 16]
num_classes = 3

pytorch_model = BasicNeuralNetwork(input_dim, hidden_dims, num_classes, task='classification')

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

# Create data loader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)
```

### Training PyTorch Model

```python
def train_pytorch_model(model, train_loader, criterion, optimizer, num_epochs=50):
    """
    Train PyTorch model
    
    Args:
        model: PyTorch model
        train_loader: Data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
    
    Returns:
        Training history
    """
    model.train()
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    
    return history

# Train PyTorch model
pytorch_history = train_pytorch_model(pytorch_model, train_loader, criterion, optimizer)
```

## Training Neural Networks

### Learning Rate Scheduling

```python
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import ExponentialDecay

def create_learning_rate_schedule(initial_lr=0.001, decay_steps=1000, decay_rate=0.9):
    """
    Create learning rate schedule
    
    Args:
        initial_lr: Initial learning rate
        decay_steps: Number of steps for decay
        decay_rate: Decay rate
    
    Returns:
        Learning rate schedule
    """
    lr_schedule = ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate
    )
    return lr_schedule

# Create model with learning rate schedule
lr_schedule = create_learning_rate_schedule()
optimizer_with_schedule = Adam(learning_rate=lr_schedule)

model_with_schedule = create_basic_neural_network(X_train.shape[1], num_classes=3, task='classification')
model_with_schedule.compile(
    optimizer=optimizer_with_schedule,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train with learning rate schedule
history_schedule = model_with_schedule.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
```

### Batch Size Optimization

```python
def find_optimal_batch_size(X_train, y_train, model_fn, batch_sizes=[16, 32, 64, 128]):
    """
    Find optimal batch size for training
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_fn: Function to create model
        batch_sizes: List of batch sizes to test
    
    Returns:
        Dictionary with results for each batch size
    """
    results = {}
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        # Create model
        model = model_fn(X_train.shape[1], num_classes=3, task='classification')
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0
        )
        
        # Store results
        results[batch_size] = {
            'final_accuracy': history.history['val_accuracy'][-1],
            'final_loss': history.history['val_loss'][-1],
            'training_time': len(history.history['loss'])
        }
    
    return results

# Test different batch sizes
batch_size_results = find_optimal_batch_size(
    X_train_scaled, y_train, create_basic_neural_network
)

# Plot results
batch_sizes = list(batch_size_results.keys())
accuracies = [batch_size_results[bs]['final_accuracy'] for bs in batch_sizes]

plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, accuracies, 'bo-')
plt.xlabel('Batch Size')
plt.ylabel('Validation Accuracy')
plt.title('Batch Size vs Validation Accuracy')
plt.grid(True)
plt.show()
```

## Optimization Techniques

### Gradient Descent Variants

```python
def compare_optimizers(X_train, y_train, optimizers):
    """
    Compare different optimizers
    
    Args:
        X_train: Training features
        y_train: Training labels
        optimizers: Dictionary of optimizers to compare
    
    Returns:
        Dictionary with training histories
    """
    results = {}
    
    for name, optimizer in optimizers.items():
        print(f"Testing optimizer: {name}")
        
        # Create model
        model = create_basic_neural_network(X_train.shape[1], num_classes=3, task='classification')
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        results[name] = history.history
    
    return results

# Define different optimizers
optimizers = {
    'SGD': SGD(learning_rate=0.01),
    'Adam': Adam(learning_rate=0.001),
    'RMSprop': keras.optimizers.RMSprop(learning_rate=0.001),
    'Adagrad': keras.optimizers.Adagrad(learning_rate=0.01)
}

# Compare optimizers
optimizer_results = compare_optimizers(X_train_scaled, y_train, optimizers)

# Plot comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
for name, history in optimizer_results.items():
    plt.plot(history['val_loss'], label=name)
plt.title('Validation Loss by Optimizer')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for name, history in optimizer_results.items():
    plt.plot(history['val_accuracy'], label=name)
plt.title('Validation Accuracy by Optimizer')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Regularization Methods

### Dropout Regularization

```python
def create_model_with_dropout(input_dim, num_classes, dropout_rates=[0.1, 0.3, 0.5]):
    """
    Create models with different dropout rates
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        dropout_rates: List of dropout rates to test
    
    Returns:
        Dictionary of models
    """
    models = {}
    
    for rate in dropout_rates:
        model = Sequential([
            Dense(64, activation='relu', input_shape=(input_dim,)),
            Dropout(rate),
            Dense(32, activation='relu'),
            Dropout(rate),
            Dense(16, activation='relu'),
            Dropout(rate),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        models[f'dropout_{rate}'] = model
    
    return models

# Create models with different dropout rates
dropout_models = create_model_with_dropout(X_train.shape[1], 3)

# Train and compare
dropout_results = {}

for name, model in dropout_models.items():
    print(f"Training {name}")
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )
    
    dropout_results[name] = history.history

# Plot dropout comparison
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
for name, history in dropout_results.items():
    plt.plot(history['val_loss'], label=name)
plt.title('Validation Loss by Dropout Rate')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for name, history in dropout_results.items():
    plt.plot(history['val_accuracy'], label=name)
plt.title('Validation Accuracy by Dropout Rate')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## Model Evaluation

### Model Performance Analysis

```python
def evaluate_model_performance(model, X_test, y_test, model_name="Model"):
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1) if len(y_pred.shape) > 1 else y_pred
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_classes)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Classification report
    report = classification_report(y_test, y_pred_classes)
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'predicted_classes': y_pred_classes
    }

# Evaluate models
basic_eval = evaluate_model_performance(
    class_model, X_test_scaled, y_test, "Basic Neural Network"
)

advanced_eval = evaluate_model_performance(
    advanced_model, X_test_scaled, y_test, "Advanced Neural Network"
)
```

## Best Practices

### Model Architecture Design

```python
def design_optimal_architecture(input_dim, num_classes, max_layers=5):
    """
    Design optimal neural network architecture
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        max_layers: Maximum number of hidden layers
    
    Returns:
        Optimal model
    """
    # Calculate optimal layer sizes
    layer_sizes = []
    current_size = input_dim
    
    for i in range(max_layers):
        # Reduce size by factor of 2, but not below 8
        new_size = max(current_size // 2, 8)
        if new_size >= num_classes:
            layer_sizes.append(new_size)
            current_size = new_size
        else:
            break
    
    # Create model
    layers = []
    prev_size = input_dim
    
    for size in layer_sizes:
        layers.append(Dense(size, activation='relu', input_shape=(prev_size,) if len(layers) == 0 else None))
        layers.append(BatchNormalization())
        layers.append(Dropout(0.3))
        prev_size = size
    
    layers.append(Dense(num_classes, activation='softmax'))
    
    model = Sequential(layers)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create optimal architecture
optimal_model = design_optimal_architecture(X_train.shape[1], 3)
print(optimal_model.summary())
```

### Hyperparameter Optimization

```python
def optimize_hyperparameters(X_train, y_train, param_grid):
    """
    Optimize hyperparameters using grid search
    
    Args:
        X_train: Training features
        y_train: Training labels
        param_grid: Dictionary of parameter grids
    
    Returns:
        Best model and parameters
    """
    best_score = 0
    best_params = None
    best_model = None
    
    for hidden_layers in param_grid['hidden_layers']:
        for dropout_rate in param_grid['dropout_rate']:
            for learning_rate in param_grid['learning_rate']:
                print(f"Testing: layers={hidden_layers}, dropout={dropout_rate}, lr={learning_rate}")
                
                # Create model
                model = Sequential()
                prev_size = X_train.shape[1]
                
                for layer_size in hidden_layers:
                    model.add(Dense(layer_size, activation='relu', input_shape=(prev_size,) if len(model.layers) == 0 else None))
                    model.add(BatchNormalization())
                    model.add(Dropout(dropout_rate))
                    prev_size = layer_size
                
                model.add(Dense(3, activation='softmax'))
                
                model.compile(
                    optimizer=Adam(learning_rate=learning_rate),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0
                )
                
                # Evaluate
                val_accuracy = max(history.history['val_accuracy'])
                
                if val_accuracy > best_score:
                    best_score = val_accuracy
                    best_params = {
                        'hidden_layers': hidden_layers,
                        'dropout_rate': dropout_rate,
                        'learning_rate': learning_rate
                    }
                    best_model = model
    
    return best_model, best_params, best_score

# Define parameter grid
param_grid = {
    'hidden_layers': [[64, 32], [128, 64, 32], [64, 32, 16]],
    'dropout_rate': [0.2, 0.3, 0.4],
    'learning_rate': [0.001, 0.01]
}

# Optimize hyperparameters
best_model, best_params, best_score = optimize_hyperparameters(
    X_train_scaled, y_train, param_grid
)

print(f"Best parameters: {best_params}")
print(f"Best validation accuracy: {best_score:.4f}")
```

## Exercises

1. **Basic Neural Network**: Build a neural network to classify the Iris dataset.
2. **Architecture Design**: Experiment with different architectures and compare their performance.
3. **Regularization**: Implement and compare different regularization techniques.
4. **Optimization**: Test different optimizers and learning rate schedules.
5. **Transfer Learning**: Use a pre-trained model for a new classification task.

## Next Steps

After mastering neural networks, explore:
- [Computer Vision](computer_vision_guide.md)
- [Natural Language Processing](nlp_guide.md)
- [Reinforcement Learning](reinforcement_learning_guide.md)
- [Advanced Deep Learning Techniques](../PyTorch/advanced_pytorch_techniques_guide.md) 