# PyTorch Image Classification: A Comprehensive Guide

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Table of Contents

1. [Introduction to Image Classification](#introduction-to-image-classification)
2. [Data Preparation](#data-preparation)
3. [Model Architectures](#model-architectures)
4. [Training Pipeline](#training-pipeline)
5. [Data Augmentation](#data-augmentation)
6. [Transfer Learning](#transfer-learning)
7. [Model Evaluation](#model-evaluation)
8. [Advanced Techniques](#advanced-techniques)
9. [Deployment](#deployment)
10. [Best Practices](#best-practices)

## Introduction to Image Classification

Image classification is a fundamental computer vision task where the goal is to assign a label or class to an input image. This guide covers building, training, and deploying image classification models using PyTorch.

### Key Concepts:

- **Convolutional Neural Networks (CNNs)**: Specialized neural networks for image processing
- **Feature Extraction**: Learning hierarchical features from images
- **Classification Head**: Final layers that map features to class predictions
- **Data Augmentation**: Techniques to increase training data diversity
- **Transfer Learning**: Using pre-trained models for new tasks

### Basic Image Classification Pipeline:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

# Create data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Define model
model = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128 * 8 * 8, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

## Data Preparation

### 1. Custom Dataset Class

```python
import os
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Load all image paths and labels
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Usage
dataset = CustomImageDataset(
    root_dir='path/to/dataset',
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
)
```

### 2. Data Transforms

```python
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random

class RandomRotation:
    """Custom random rotation transform"""
    def __init__(self, degrees):
        self.degrees = degrees
    
    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        return F.rotate(img, angle)

class RandomCrop:
    """Custom random crop transform"""
    def __init__(self, size):
        self.size = size
    
    def __call__(self, img):
        width, height = img.size
        crop_size = min(self.size, width, height)
        
        left = random.randint(0, width - crop_size)
        top = random.randint(0, height - crop_size)
        right = left + crop_size
        bottom = top + crop_size
        
        return F.crop(img, top, left, crop_size, crop_size)

# Training transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Validation transforms
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### 3. Data Loading and Splitting

```python
from torch.utils.data import DataLoader, random_split

def create_data_loaders(dataset, train_ratio=0.8, batch_size=32, num_workers=4):
    """Create train and validation data loaders"""
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

# Usage
train_loader, val_loader = create_data_loaders(dataset)
```

## Model Architectures

### 1. Simple CNN

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

### 2. ResNet-like Architecture

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

### 3. EfficientNet-like Architecture

```python
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConv, self).__init__()
        
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_residual = in_channels == out_channels and stride == 1
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride, 
                      padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EfficientNet(nn.Module):
    def __init__(self, num_classes=10):
        super(EfficientNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            
            MBConv(32, 16, 3, 1, 1),
            MBConv(16, 24, 3, 2, 6),
            MBConv(24, 24, 3, 1, 6),
            MBConv(24, 40, 5, 2, 6),
            MBConv(40, 40, 5, 1, 6),
            MBConv(40, 80, 3, 2, 6),
            MBConv(80, 80, 3, 1, 6),
            MBConv(80, 80, 3, 1, 6),
            MBConv(80, 112, 5, 1, 6),
            MBConv(112, 112, 5, 1, 6),
            MBConv(112, 192, 5, 2, 6),
            MBConv(192, 192, 5, 1, 6),
            MBConv(192, 192, 5, 1, 6),
            MBConv(192, 320, 3, 1, 6),
            
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

## Training Pipeline

### 1. Complete Training Loop

```python
import time
from tqdm import tqdm

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs):
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 20)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f'New best model saved! Validation Accuracy: {val_acc:.2f}%')
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')

# Usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

trainer = Trainer(model, criterion, optimizer, scheduler, device)
trainer.train(train_loader, val_loader, num_epochs=50)
```

### 2. Learning Rate Scheduling

```python
def create_scheduler(optimizer, scheduler_type='cosine', num_epochs=100):
    """Create different types of learning rate schedulers"""
    
    if scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    elif scheduler_type == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    elif scheduler_type == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=10, verbose=True
        )
    
    elif scheduler_type == 'one_cycle':
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01, epochs=num_epochs, steps_per_epoch=len(train_loader)
        )
    
    else:
        return None
```

## Data Augmentation

### 1. Advanced Augmentation Techniques

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_advanced_transforms():
    """Advanced data augmentation using Albumentations"""
    
    train_transform = A.Compose([
        A.RandomResizedCrop(224, 224, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform
```

### 2. Mixup and CutMix

```python
def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0):
    """CutMix data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2
```

## Transfer Learning

### 1. Using Pre-trained Models

```python
import torchvision.models as models

def create_transfer_model(model_name, num_classes, freeze_backbone=False):
    """Create a transfer learning model"""
    
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        num_features = model.classifier.in_features
        model.classifier = nn.Linear(num_features, num_classes)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Freeze backbone if requested
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
    
    return model

# Usage
model = create_transfer_model('resnet50', num_classes=10, freeze_backbone=True)
```

### 2. Fine-tuning Strategies

```python
def fine_tune_model(model, num_epochs, freeze_epochs=5):
    """Fine-tuning strategy with gradual unfreezing"""
    
    # Phase 1: Freeze backbone, train only classifier
    for name, param in model.named_parameters():
        if 'fc' not in name and 'classifier' not in name:
            param.requires_grad = False
    
    print("Phase 1: Training classifier only")
    # Train for freeze_epochs
    
    # Phase 2: Unfreeze last few layers
    for name, param in model.named_parameters():
        if 'layer4' in name or 'layer3' in name:
            param.requires_grad = True
    
    print("Phase 2: Training last few layers")
    # Train for a few more epochs
    
    # Phase 3: Unfreeze all layers with lower learning rate
    for param in model.parameters():
        param.requires_grad = True
    
    print("Phase 3: Training all layers")
    # Train for remaining epochs
```

## Model Evaluation

### 1. Comprehensive Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device, class_names):
    """Comprehensive model evaluation"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    report = classification_report(all_targets, all_predictions, 
                                 target_names=class_names, output_dict=True)
    
    # Plot confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return accuracy, report, all_probabilities

# Usage
accuracy, report, probabilities = evaluate_model(model, test_loader, device, class_names)
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Per-class F1-scores: {report['weighted avg']['f1-score']:.4f}")
```

### 2. Error Analysis

```python
def analyze_errors(model, test_loader, device, class_names):
    """Analyze model errors"""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            # Find misclassified samples
            mask = predicted != target
            if mask.any():
                error_indices = mask.nonzero().squeeze()
                for idx in error_indices:
                    errors.append({
                        'image': data[idx].cpu(),
                        'true_label': target[idx].item(),
                        'predicted_label': predicted[idx].item(),
                        'confidence': torch.softmax(output[idx], dim=0).max().item()
                    })
    
    # Display some error examples
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, error in enumerate(errors[:6]):
        row, col = i // 3, i % 3
        img = error['image'].permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        
        axes[row, col].imshow(img)
        axes[row, col].set_title(
            f'True: {class_names[error["true_label"]]}\n'
            f'Pred: {class_names[error["predicted_label"]]}\n'
            f'Conf: {error["confidence"]:.3f}'
        )
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
```

## Advanced Techniques

### 1. Ensemble Methods

```python
class EnsembleModel(nn.Module):
    def __init__(self, models, weights=None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights is not None else [1.0] * len(models)
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Weighted average of outputs
        weighted_output = torch.zeros_like(outputs[0])
        for output, weight in zip(outputs, self.weights):
            weighted_output += weight * output
        
        return weighted_output

# Create ensemble
model1 = create_transfer_model('resnet18', num_classes=10)
model2 = create_transfer_model('resnet50', num_classes=10)
model3 = create_transfer_model('densenet121', num_classes=10)

ensemble = EnsembleModel([model1, model2, model3], weights=[0.3, 0.4, 0.3])
```

### 2. Test Time Augmentation (TTA)

```python
def test_time_augmentation(model, image, num_augments=10):
    """Apply test time augmentation"""
    model.eval()
    predictions = []
    
    # Original image
    with torch.no_grad():
        pred = model(image.unsqueeze(0))
        predictions.append(torch.softmax(pred, dim=1))
    
    # Augmented versions
    for _ in range(num_augments - 1):
        # Apply random augmentation
        augmented = apply_random_augmentation(image)
        with torch.no_grad():
            pred = model(augmented.unsqueeze(0))
            predictions.append(torch.softmax(pred, dim=1))
    
    # Average predictions
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred

def apply_random_augmentation(image):
    """Apply random augmentation to image"""
    # Random horizontal flip
    if random.random() > 0.5:
        image = torch.flip(image, dims=[2])
    
    # Random rotation
    angle = random.uniform(-10, 10)
    # Apply rotation (simplified)
    
    # Random brightness/contrast
    brightness = random.uniform(0.9, 1.1)
    contrast = random.uniform(0.9, 1.1)
    image = image * brightness
    image = (image - image.mean()) * contrast + image.mean()
    
    return image
```

## Deployment

### 1. Model Export

```python
def export_model(model, save_path, input_shape=(1, 3, 224, 224)):
    """Export model for deployment"""
    model.eval()
    
    # Export to TorchScript
    dummy_input = torch.randn(input_shape)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(f"{save_path}.pt")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        f"{save_path}.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {save_path}.pt and {save_path}.onnx")

# Usage
export_model(model, "image_classifier")
```

### 2. Inference Pipeline

```python
class ImageClassifier:
    def __init__(self, model_path, class_names, device='cpu'):
        self.device = device
        self.class_names = class_names
        
        # Load model
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_path):
        """Predict class for a single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'class': self.class_names[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy()
        }
    
    def predict_batch(self, image_paths):
        """Predict classes for multiple images"""
        results = []
        for image_path in image_paths:
            results.append(self.predict(image_path))
        return results

# Usage
classifier = ImageClassifier("best_model.pth", class_names, device)
result = classifier.predict("path/to/image.jpg")
print(f"Predicted: {result['class']} (confidence: {result['confidence']:.3f})")
```

## Best Practices

### 1. Model Architecture Selection

```python
def select_model_architecture(dataset_size, num_classes, input_size, constraints):
    """Select appropriate model architecture based on constraints"""
    
    if constraints.get('speed') == 'fast':
        if dataset_size < 10000:
            return 'simple_cnn'
        else:
            return 'resnet18'
    
    elif constraints.get('accuracy') == 'high':
        if dataset_size > 50000:
            return 'resnet50'
        else:
            return 'efficientnet_b0'
    
    elif constraints.get('memory') == 'low':
        return 'mobilenet_v2'
    
    else:
        return 'resnet18'  # Default choice
```

### 2. Hyperparameter Optimization

```python
import optuna

def objective(trial):
    """Objective function for hyperparameter optimization"""
    
    # Hyperparameters to optimize
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Create model with hyperparameters
    model = create_model_with_hyperparams(dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train and evaluate
    train_loader, val_loader = create_data_loaders(batch_size)
    trainer = Trainer(model, criterion, optimizer, None, device)
    trainer.train(train_loader, val_loader, num_epochs=10)
    
    return trainer.val_accuracies[-1]  # Return best validation accuracy

# Run optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best trial: {study.best_trial.value}")
print(f"Best hyperparameters: {study.best_trial.params}")
```

### 3. Monitoring and Logging

```python
import wandb
from torch.utils.tensorboard import SummaryWriter

def setup_logging(project_name, run_name):
    """Setup logging with Weights & Biases and TensorBoard"""
    
    # Weights & Biases
    wandb.init(project=project_name, name=run_name)
    
    # TensorBoard
    writer = SummaryWriter(f'runs/{run_name}')
    
    return writer

def log_metrics(writer, epoch, train_loss, train_acc, val_loss, val_acc, lr):
    """Log training metrics"""
    
    # Log to TensorBoard
    writer.add_scalar('Loss/Train', train_loss, epoch)
    writer.add_scalar('Loss/Validation', val_loss, epoch)
    writer.add_scalar('Accuracy/Train', train_acc, epoch)
    writer.add_scalar('Accuracy/Validation', val_acc, epoch)
    writer.add_scalar('Learning_Rate', lr, epoch)
    
    # Log to Weights & Biases
    wandb.log({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'learning_rate': lr,
        'epoch': epoch
    })

# Usage in training loop
writer = setup_logging("image-classification", "resnet50-experiment")
log_metrics(writer, epoch, train_loss, train_acc, val_loss, val_acc, lr)
```

## Summary

This comprehensive guide covers PyTorch image classification:

1. **Data Preparation**: Custom datasets, transforms, and data loading
2. **Model Architectures**: Simple CNNs, ResNet, and EfficientNet
3. **Training Pipeline**: Complete training loops with validation
4. **Data Augmentation**: Advanced techniques including Mixup and CutMix
5. **Transfer Learning**: Using pre-trained models effectively
6. **Model Evaluation**: Comprehensive evaluation and error analysis
7. **Advanced Techniques**: Ensembles and test time augmentation
8. **Deployment**: Model export and inference pipelines
9. **Best Practices**: Architecture selection and hyperparameter optimization

Mastering these concepts will enable you to build effective image classification systems for various applications.

## Next Steps

- Experiment with different datasets (CIFAR-10, ImageNet, custom datasets)
- Try advanced architectures (Vision Transformers, ConvNeXt)
- Explore multi-label classification and object detection
- Implement real-time inference systems

## References

- [PyTorch Vision Documentation](https://pytorch.org/vision/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning for Computer Vision](https://cs231n.github.io/)
- [ImageNet Dataset](https://image-net.org/) 