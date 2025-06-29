# PyTorch Neural Networks: A Comprehensive Guide

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Table of Contents

1. [Introduction to Neural Networks](#introduction-to-neural-networks)
2. [Building Neural Networks](#building-neural-networks)
3. [Training Neural Networks](#training-neural-networks)
4. [Loss Functions](#loss-functions)
5. [Optimizers](#optimizers)
6. [Regularization Techniques](#regularization-techniques)
7. [Model Evaluation](#model-evaluation)
8. [Advanced Architectures](#advanced-architectures)
9. [Best Practices](#best-practices)

## Introduction to Neural Networks

Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process input data to produce output predictions.

### Key Components:

- **Input Layer**: Receives the input data
- **Hidden Layers**: Process the data through weighted connections
- **Output Layer**: Produces the final predictions
- **Weights**: Parameters that are learned during training
- **Biases**: Additional parameters that help with model flexibility
- **Activation Functions**: Non-linear functions that introduce complexity

### Basic Neural Network Structure:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x

# Create model
model = SimpleNN(input_size=10, hidden_size=20, output_size=1)
print(model)
```

## Building Neural Networks

### 1. Sequential Models

```python
# Using nn.Sequential
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 10),
    nn.Softmax(dim=1)
)

print(model)
```

### 2. Custom Module Classes

```python
class CustomNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(CustomNN, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Create model
model = CustomNN(input_size=784, hidden_sizes=[256, 128, 64], output_size=10)
print(model)
```

### 3. Layer Types

```python
class LayerTypesExample(nn.Module):
    def __init__(self):
        super(LayerTypesExample, self).__init__()
        
        # Linear (Fully Connected) Layer
        self.linear = nn.Linear(10, 5)
        
        # Convolutional Layer
        self.conv1d = nn.Conv1d(1, 16, kernel_size=3)
        self.conv2d = nn.Conv2d(3, 16, kernel_size=3)
        
        # Recurrent Layer
        self.lstm = nn.LSTM(10, 20, batch_first=True)
        self.gru = nn.GRU(10, 20, batch_first=True)
        
        # Normalization Layers
        self.batch_norm = nn.BatchNorm1d(5)
        self.layer_norm = nn.LayerNorm(5)
        
        # Pooling Layers
        self.max_pool1d = nn.MaxPool1d(2)
        self.avg_pool2d = nn.AvgPool2d(2)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Example forward pass
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x
```

### 4. Activation Functions

```python
class ActivationFunctions(nn.Module):
    def __init__(self):
        super(ActivationFunctions, self).__init__()
        
        # Common activation functions
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.gelu = nn.GELU()
        self.swish = lambda x: x * torch.sigmoid(x)  # Custom Swish
    
    def forward(self, x):
        # Apply different activations
        relu_out = self.relu(x)
        sigmoid_out = self.sigmoid(x)
        tanh_out = self.tanh(x)
        
        return relu_out, sigmoid_out, tanh_out

# Test activations
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
model = ActivationFunctions()
relu_out, sigmoid_out, tanh_out = model(x)

print(f"Input: {x}")
print(f"ReLU: {relu_out}")
print(f"Sigmoid: {sigmoid_out}")
print(f"Tanh: {tanh_out}")
```

## Training Neural Networks

### 1. Basic Training Loop

```python
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Average Loss: {epoch_loss:.4f}')

# Example usage
model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Create dummy data
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = torch.utils.data.TensorDataset(X, y)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Train model
train_model(model, train_loader, criterion, optimizer)
```

### 2. Training with Validation

```python
def train_with_validation(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    return train_losses, val_losses
```

### 3. Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience

def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, patience=7):
    early_stopping = EarlyStopping(patience=patience)
    
    for epoch in range(100):  # Max epochs
        # Training
        model.train()
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}')
        
        # Check early stopping
        if early_stopping(val_loss/len(val_loader)):
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
```

## Loss Functions

### 1. Regression Losses

```python
# Mean Squared Error (MSE)
mse_loss = nn.MSELoss()

# Mean Absolute Error (MAE)
mae_loss = nn.L1Loss()

# Huber Loss (combines MSE and MAE)
huber_loss = nn.SmoothL1Loss()

# Example usage
predictions = torch.tensor([1.0, 2.0, 3.0])
targets = torch.tensor([1.1, 1.9, 3.1])

print(f"MSE Loss: {mse_loss(predictions, targets):.4f}")
print(f"MAE Loss: {mae_loss(predictions, targets):.4f}")
print(f"Huber Loss: {huber_loss(predictions, targets):.4f}")
```

### 2. Classification Losses

```python
# Cross Entropy Loss (combines LogSoftmax and NLLLoss)
ce_loss = nn.CrossEntropyLoss()

# Binary Cross Entropy Loss
bce_loss = nn.BCELoss()

# Binary Cross Entropy with Logits
bce_logits_loss = nn.BCEWithLogitsLoss()

# Example usage
# For multi-class classification
logits = torch.tensor([[1.0, 2.0, 0.5], [2.0, 1.0, 0.1]])
targets = torch.tensor([1, 0])  # Class indices

ce_loss_value = ce_loss(logits, targets)
print(f"Cross Entropy Loss: {ce_loss_value:.4f}")

# For binary classification
predictions = torch.sigmoid(torch.tensor([0.5, 0.8, 0.2]))
binary_targets = torch.tensor([1.0, 1.0, 0.0])

bce_loss_value = bce_loss(predictions, binary_targets)
print(f"Binary Cross Entropy Loss: {bce_loss_value:.4f}")
```

### 3. Custom Loss Functions

```python
class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        mae_loss = self.mae(predictions, targets)
        
        # Combine MSE and MAE with weight alpha
        combined_loss = self.alpha * mse_loss + (1 - self.alpha) * mae_loss
        return combined_loss

# Example usage
custom_loss = CustomLoss(alpha=0.7)
predictions = torch.tensor([1.0, 2.0, 3.0])
targets = torch.tensor([1.1, 1.9, 3.1])

loss_value = custom_loss(predictions, targets)
print(f"Custom Loss: {loss_value:.4f}")
```

## Optimizers

### 1. Stochastic Gradient Descent (SGD)

```python
# Basic SGD
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# SGD with weight decay (L2 regularization)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# SGD with Nesterov momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
```

### 2. Adam Optimizer

```python
# Basic Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Adam with custom parameters
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4
)
```

### 3. Learning Rate Scheduling

```python
# Step LR scheduler
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Exponential LR scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Cosine annealing scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# Reduce LR on plateau
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=10, verbose=True
)

# Example training loop with scheduler
for epoch in range(100):
    # Training code here...
    
    # Update learning rate
    scheduler.step()  # or scheduler.step(val_loss) for ReduceLROnPlateau
    
    print(f'Epoch {epoch+1}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
```

## Regularization Techniques

### 1. Dropout

```python
class DropoutExample(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(DropoutExample, self).__init__()
        self.layer1 = nn.Linear(784, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.layer3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.dropout1(torch.relu(self.layer1(x)))
        x = self.dropout2(torch.relu(self.layer2(x)))
        x = self.layer3(x)
        return x
```

### 2. Batch Normalization

```python
class BatchNormExample(nn.Module):
    def __init__(self):
        super(BatchNormExample, self).__init__()
        self.layer1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.layer3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.bn1(torch.relu(self.layer1(x)))
        x = self.bn2(torch.relu(self.layer2(x)))
        x = self.layer3(x)
        return x
```

### 3. Weight Decay (L2 Regularization)

```python
# L2 regularization through optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Manual L2 regularization
def l2_regularization_loss(model, lambda_reg=1e-4):
    l2_loss = 0.0
    for param in model.parameters():
        l2_loss += torch.norm(param, p=2)
    return lambda_reg * l2_loss

# Add to training loop
loss = criterion(output, target) + l2_regularization_loss(model)
```

## Model Evaluation

### 1. Accuracy Metrics

```python
def calculate_accuracy(predictions, targets):
    """Calculate accuracy for classification"""
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total

def calculate_metrics(predictions, targets, threshold=0.5):
    """Calculate precision, recall, F1-score for binary classification"""
    predicted = (predictions > threshold).float()
    
    tp = (predicted * targets).sum().item()
    fp = (predicted * (1 - targets)).sum().item()
    fn = ((1 - predicted) * targets).sum().item()
    tn = ((1 - predicted) * (1 - targets)).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1
```

### 2. Model Evaluation Loop

```python
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    avg_loss = test_loss / len(test_loader)
    
    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy
```

### 3. Confusion Matrix

```python
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(model, test_loader, num_classes=10):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return cm
```

## Advanced Architectures

### 1. Residual Networks (ResNet)

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.layer4 = self.make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
```

### 2. Attention Mechanism

```python
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        energy = torch.einsum("nqhd,nkhd->nqhk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nqhk,nvhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out
```

## Best Practices

### 1. Model Initialization

```python
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)

# Apply initialization
model = YourModel()
model.apply(init_weights)
```

### 2. Data Loading

```python
# Efficient data loading
train_dataset = YourDataset(data, targets)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,  # Parallel data loading
    pin_memory=True  # Faster data transfer to GPU
)
```

### 3. Model Checkpointing

```python
def save_checkpoint(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss
```

### 4. Gradient Clipping

```python
# Clip gradients to prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

## Summary

This comprehensive guide covers PyTorch neural networks:

1. **Building Networks**: Creating models with Sequential and custom modules
2. **Training**: Complete training loops with validation and early stopping
3. **Loss Functions**: Various loss functions for different tasks
4. **Optimizers**: Different optimization algorithms and learning rate scheduling
5. **Regularization**: Techniques to prevent overfitting
6. **Evaluation**: Metrics and evaluation procedures
7. **Advanced Architectures**: ResNet and attention mechanisms
8. **Best Practices**: Initialization, data loading, and model management

Mastering these concepts will enable you to build and train effective neural networks for various machine learning tasks.

## Next Steps

- Explore the [Image Classification Tutorial](../Image-Classifier/) for real-world applications
- Practice with different datasets and architectures
- Experiment with hyperparameter tuning and model optimization

## References

- [PyTorch Neural Networks Documentation](https://pytorch.org/docs/stable/nn.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch](https://pytorch.org/deep-learning-with-pytorch) 