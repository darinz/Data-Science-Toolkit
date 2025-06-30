# Computer Vision Guide

A comprehensive guide to computer vision techniques and applications using Python for AI/ML projects.

## Table of Contents
1. [Introduction to Computer Vision](#introduction-to-computer-vision)
2. [Image Processing Fundamentals](#image-processing-fundamentals)
3. [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
4. [Image Classification](#image-classification)
5. [Object Detection](#object-detection)
6. [Image Segmentation](#image-segmentation)
7. [Feature Extraction](#feature-extraction)
8. [Image Augmentation](#image-augmentation)
9. [Best Practices](#best-practices)

## Introduction to Computer Vision

### What is Computer Vision?

Computer vision is a field of artificial intelligence that enables computers to interpret and understand visual information from the world, including images and videos.

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
```

### Basic Image Operations

```python
def load_and_display_image(image_path):
    """
    Load and display an image using different methods
    
    Args:
        image_path: Path to the image file
    """
    # Using OpenCV
    img_cv = cv2.imread(image_path)
    img_cv_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    
    # Using PIL
    img_pil = Image.open(image_path)
    img_pil_array = np.array(img_pil)
    
    # Using matplotlib
    img_matplotlib = plt.imread(image_path)
    
    # Display images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_cv_rgb)
    axes[0].set_title('OpenCV')
    axes[0].axis('off')
    
    axes[1].imshow(img_pil_array)
    axes[1].set_title('PIL')
    axes[1].axis('off')
    
    axes[2].imshow(img_matplotlib)
    axes[2].set_title('Matplotlib')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return img_cv, img_pil, img_matplotlib

def get_image_info(image):
    """
    Get basic information about an image
    
    Args:
        image: Image array
    
    Returns:
        Dictionary with image information
    """
    info = {
        'shape': image.shape,
        'dtype': image.dtype,
        'min_value': image.min(),
        'max_value': image.max(),
        'mean_value': image.mean(),
        'size': image.size
    }
    
    if len(image.shape) == 3:
        info['channels'] = image.shape[2]
        info['height'] = image.shape[0]
        info['width'] = image.shape[1]
    else:
        info['channels'] = 1
        info['height'] = image.shape[0]
        info['width'] = image.shape[1]
    
    return info

# Example usage (you would need an actual image file)
# img_cv, img_pil, img_matplotlib = load_and_display_image('path/to/image.jpg')
# info = get_image_info(img_cv)
# print(info)
```

## Image Processing Fundamentals

### Basic Image Transformations

```python
def basic_image_transformations(image):
    """
    Apply basic image transformations
    
    Args:
        image: Input image array
    
    Returns:
        Dictionary of transformed images
    """
    transformations = {}
    
    # Resize
    height, width = image.shape[:2]
    resized = cv2.resize(image, (width//2, height//2))
    transformations['resized'] = resized
    
    # Rotate
    center = (width//2, height//2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    transformations['rotated'] = rotated
    
    # Flip horizontally
    flipped_h = cv2.flip(image, 1)
    transformations['flipped_horizontal'] = flipped_h
    
    # Flip vertically
    flipped_v = cv2.flip(image, 0)
    transformations['flipped_vertical'] = flipped_v
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        transformations['grayscale'] = gray
    
    return transformations

def display_transformations(original, transformations):
    """
    Display original image and its transformations
    
    Args:
        original: Original image
        transformations: Dictionary of transformed images
    """
    n_transforms = len(transformations)
    fig, axes = plt.subplots(2, (n_transforms + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    # Display original
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Display transformations
    for i, (name, img) in enumerate(transformations.items(), 1):
        if len(img.shape) == 3:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            axes[i].imshow(img, cmap='gray')
        axes[i].set_title(name)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_transforms + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
```

### Image Filtering

```python
def apply_image_filters(image):
    """
    Apply various image filters
    
    Args:
        image: Input image array
    
    Returns:
        Dictionary of filtered images
    """
    filters = {}
    
    # Gaussian blur
    gaussian = cv2.GaussianBlur(image, (15, 15), 0)
    filters['gaussian_blur'] = gaussian
    
    # Median blur
    median = cv2.medianBlur(image, 15)
    filters['median_blur'] = median
    
    # Bilateral filter
    bilateral = cv2.bilateralFilter(image, 15, 75, 75)
    filters['bilateral_filter'] = bilateral
    
    # Convert to grayscale for edge detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Sobel edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    filters['sobel_edges'] = sobel
    
    # Canny edge detection
    canny = cv2.Canny(gray, 50, 150)
    filters['canny_edges'] = canny
    
    # Laplacian edge detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    filters['laplacian_edges'] = laplacian
    
    return filters

def display_filters(original, filters):
    """
    Display original image and filtered versions
    
    Args:
        original: Original image
        filters: Dictionary of filtered images
    """
    n_filters = len(filters)
    fig, axes = plt.subplots(2, (n_filters + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    # Display original
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Display filters
    for i, (name, img) in enumerate(filters.items(), 1):
        if len(img.shape) == 3:
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            axes[i].imshow(img, cmap='gray')
        axes[i].set_title(name)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_filters + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
```

### Color Space Transformations

```python
def color_space_transformations(image):
    """
    Convert image between different color spaces
    
    Args:
        image: Input BGR image
    
    Returns:
        Dictionary of images in different color spaces
    """
    color_spaces = {}
    
    # RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color_spaces['RGB'] = rgb
    
    # HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_spaces['HSV'] = hsv
    
    # LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    color_spaces['LAB'] = lab
    
    # YUV
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    color_spaces['YUV'] = yuv
    
    # Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    color_spaces['Grayscale'] = gray
    
    return color_spaces

def display_color_spaces(original, color_spaces):
    """
    Display image in different color spaces
    
    Args:
        original: Original BGR image
        color_spaces: Dictionary of images in different color spaces
    """
    n_spaces = len(color_spaces)
    fig, axes = plt.subplots(2, (n_spaces + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    # Display original
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original (BGR)')
    axes[0].axis('off')
    
    # Display color spaces
    for i, (name, img) in enumerate(color_spaces.items(), 1):
        if len(img.shape) == 3:
            axes[i].imshow(img)
        else:
            axes[i].imshow(img, cmap='gray')
        axes[i].set_title(name)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_spaces + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
```

## Convolutional Neural Networks (CNNs)

### Building a Basic CNN

```python
def create_basic_cnn(input_shape, num_classes):
    """
    Create a basic CNN architecture
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        Compiled CNN model
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_advanced_cnn(input_shape, num_classes):
    """
    Create an advanced CNN with batch normalization and regularization
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
    
    Returns:
        Compiled CNN model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

### CNN Visualization

```python
def visualize_cnn_layers(model, input_image):
    """
    Visualize intermediate layers of a CNN
    
    Args:
        model: Trained CNN model
        input_image: Input image for visualization
    """
    # Get layer outputs
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # Get activations
    activations = activation_model.predict(input_image)
    
    # Visualize activations
    for i, activation in enumerate(activations):
        n_features = activation.shape[-1]
        size = activation.shape[1]
        
        # Display grid of activations
        n_cols = 8
        n_rows = n_features // n_cols + (1 if n_features % n_cols != 0 else 0)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2*n_rows))
        axes = axes.flatten()
        
        for j in range(n_features):
            if j < n_features:
                axes[j].imshow(activation[0, :, :, j], cmap='viridis')
                axes[j].set_title(f'Feature {j+1}')
                axes[j].axis('off')
            else:
                axes[j].axis('off')
        
        plt.suptitle(f'Layer {i+1} Activations')
        plt.tight_layout()
        plt.show()

def plot_model_architecture(model):
    """
    Plot CNN model architecture
    
    Args:
        model: CNN model
    """
    keras.utils.plot_model(
        model,
        to_file='cnn_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB'
    )
    print("Model architecture saved as 'cnn_architecture.png'")
```

## Image Classification

### Data Preparation

```python
def prepare_image_data(data_dir, img_size=(224, 224), batch_size=32):
    """
    Prepare image data for training
    
    Args:
        data_dir: Directory containing image data
        img_size: Target image size
        batch_size: Batch size for training
    
    Returns:
        Training and validation data generators
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='training'
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation'
    )
    
    return train_generator, val_generator

def create_classification_model(input_shape, num_classes, model_type='basic'):
    """
    Create image classification model
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of classes
        model_type: Type of model ('basic', 'advanced', 'transfer')
    
    Returns:
        Compiled model
    """
    if model_type == 'basic':
        return create_basic_cnn(input_shape, num_classes)
    elif model_type == 'advanced':
        return create_advanced_cnn(input_shape, num_classes)
    elif model_type == 'transfer':
        return create_transfer_learning_model(input_shape, num_classes)
    else:
        raise ValueError("model_type must be 'basic', 'advanced', or 'transfer'")

def train_classification_model(model, train_generator, val_generator, epochs=50):
    """
    Train image classification model
    
    Args:
        model: Model to train
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs: Number of training epochs
    
    Returns:
        Training history
    """
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    return history
```

### Transfer Learning for Image Classification

```python
def create_transfer_learning_model(input_shape, num_classes, base_model_name='vgg16'):
    """
    Create transfer learning model using pre-trained networks
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of classes
        base_model_name: Name of pre-trained model
    
    Returns:
        Transfer learning model
    """
    # Load pre-trained model
    if base_model_name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'mobilenet':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("base_model_name must be 'vgg16', 'resnet50', or 'mobilenet'")
    
    # Freeze base model
    base_model.trainable = False
    
    # Create new model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def fine_tune_transfer_model(model, train_generator, val_generator, epochs_fine_tune=10):
    """
    Fine-tune transfer learning model
    
    Args:
        model: Transfer learning model
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs_fine_tune: Number of fine-tuning epochs
    
    Returns:
        Fine-tuned model and history
    """
    # Unfreeze some layers
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze early layers, unfreeze later layers
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune
    fine_tune_history = model.fit(
        train_generator,
        epochs=epochs_fine_tune,
        validation_data=val_generator,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ],
        verbose=1
    )
    
    return model, fine_tune_history
```

## Object Detection

### Basic Object Detection with OpenCV

```python
def detect_objects_opencv(image, cascade_path):
    """
    Detect objects using OpenCV Haar cascades
    
    Args:
        image: Input image
        cascade_path: Path to cascade file
    
    Returns:
        Image with detected objects
    """
    # Load cascade classifier
    cascade = cv2.CascadeClassifier(cascade_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect objects
    objects = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangles around detected objects
    result = image.copy()
    for (x, y, w, h) in objects:
        cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return result, objects

def detect_faces(image):
    """
    Detect faces in an image
    
    Args:
        image: Input image
    
    Returns:
        Image with detected faces
    """
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangles around faces
    result = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return result, faces
```

### YOLO Object Detection

```python
def load_yolo_model():
    """
    Load YOLO model for object detection
    
    Returns:
        YOLO model and classes
    """
    # Load YOLO model
    net = cv2.dnn.readNet(
        "yolov3.weights",
        "yolov3.cfg"
    )
    
    # Load classes
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Get output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, classes, output_layers

def detect_objects_yolo(image, net, classes, output_layers, confidence_threshold=0.5):
    """
    Detect objects using YOLO
    
    Args:
        image: Input image
        net: YOLO network
        classes: Class names
        output_layers: Output layer names
        confidence_threshold: Confidence threshold
    
    Returns:
        Image with detected objects
    """
    height, width, channels = image.shape
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    # Set input
    net.setInput(blob)
    
    # Forward pass
    outs = net.forward(output_layers)
    
    # Information to display on screen
    class_ids = []
    confidences = []
    boxes = []
    
    # Showing information on the screen
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    
    # Draw boxes
    result = image.copy()
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(result, f"{label} {confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result, boxes, class_ids, confidences
```

## Image Segmentation

### Basic Image Segmentation

```python
def basic_image_segmentation(image):
    """
    Perform basic image segmentation
    
    Args:
        image: Input image
    
    Returns:
        Segmented image
    """
    # Convert to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to float
    rgb_float = rgb.astype(np.float32) / 255.0
    
    # Reshape for clustering
    pixels = rgb_float.reshape(-1, 3)
    
    # K-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(pixels)
    
    # Reshape back
    segmented = labels.reshape(rgb.shape[:2])
    
    return segmented

def watershed_segmentation(image):
    """
    Perform watershed segmentation
    
    Args:
        image: Input image
    
    Returns:
        Segmented image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    
    # Finding unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(image, markers)
    
    return markers
```

## Feature Extraction

### Traditional Feature Extraction

```python
def extract_sift_features(image):
    """
    Extract SIFT features from image
    
    Args:
        image: Input image
    
    Returns:
        Keypoints and descriptors
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def extract_orb_features(image):
    """
    Extract ORB features from image
    
    Args:
        image: Input image
    
    Returns:
        Keypoints and descriptors
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create ORB detector
    orb = cv2.ORB_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def match_features(descriptors1, descriptors2, method='sift'):
    """
    Match features between two images
    
    Args:
        descriptors1: Descriptors from first image
        descriptors2: Descriptors from second image
        method: Feature matching method
    
    Returns:
        Matches between descriptors
    """
    if method == 'sift':
        # FLANN matcher for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        # BF matcher for ORB
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = matcher.match(descriptors1, descriptors2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    return matches
```

### Deep Learning Feature Extraction

```python
def extract_deep_features(model, image, layer_name='global_average_pooling2d'):
    """
    Extract deep features using pre-trained model
    
    Args:
        model: Pre-trained model
        image: Input image
        layer_name: Name of layer to extract features from
    
    Returns:
        Deep features
    """
    # Create feature extraction model
    feature_model = keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    
    # Preprocess image
    img_array = np.expand_dims(image, axis=0)
    img_array = img_array / 255.0
    
    # Extract features
    features = feature_model.predict(img_array)
    
    return features.flatten()

def create_feature_extractor(base_model_name='vgg16'):
    """
    Create feature extractor from pre-trained model
    
    Args:
        base_model_name: Name of pre-trained model
    
    Returns:
        Feature extraction model
    """
    if base_model_name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    elif base_model_name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    elif base_model_name == 'mobilenet':
        base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
    else:
        raise ValueError("base_model_name must be 'vgg16', 'resnet50', or 'mobilenet'")
    
    return base_model
```

## Image Augmentation

### Advanced Image Augmentation

```python
def create_advanced_augmentation():
    """
    Create advanced image augmentation pipeline
    
    Returns:
        ImageDataGenerator with advanced augmentations
    """
    datagen = ImageDataGenerator(
        # Basic augmentations
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        
        # Advanced augmentations
        brightness_range=[0.8, 1.2],
        channel_shift_range=50.0,
        
        # Preprocessing
        rescale=1./255,
        preprocessing_function=None,
        
        # Fill mode
        fill_mode='nearest',
        cval=0.0
    )
    
    return datagen

def apply_custom_augmentations(image):
    """
    Apply custom image augmentations
    
    Args:
        image: Input image
    
    Returns:
        List of augmented images
    """
    augmented_images = []
    
    # Original image
    augmented_images.append(image)
    
    # Rotation
    for angle in [90, 180, 270]:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        augmented_images.append(rotated)
    
    # Brightness adjustment
    bright = cv2.convertScaleAbs(image, alpha=1.5, beta=30)
    dark = cv2.convertScaleAbs(image, alpha=0.7, beta=-30)
    augmented_images.extend([bright, dark])
    
    # Noise addition
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    augmented_images.append(noisy)
    
    # Blur
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    augmented_images.append(blurred)
    
    return augmented_images

def display_augmentations(original, augmented_images):
    """
    Display original and augmented images
    
    Args:
        original: Original image
        augmented_images: List of augmented images
    """
    n_images = len(augmented_images) + 1
    fig, axes = plt.subplots(2, (n_images + 1) // 2, figsize=(15, 8))
    axes = axes.flatten()
    
    # Display original
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Display augmented images
    titles = ['Rotation 90°', 'Rotation 180°', 'Rotation 270°', 
              'Bright', 'Dark', 'Noisy', 'Blurred']
    
    for i, (img, title) in enumerate(zip(augmented_images, titles), 1):
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(title)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
```

## Best Practices

### Model Evaluation and Visualization

```python
def evaluate_vision_model(model, test_generator, class_names):
    """
    Evaluate vision model performance
    
    Args:
        model: Trained model
        test_generator: Test data generator
        class_names: List of class names
    
    Returns:
        Evaluation results
    """
    # Make predictions
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Calculate metrics
    from sklearn.metrics import classification_report, confusion_matrix
    accuracy = np.mean(predicted_classes == true_classes)
    report = classification_report(true_classes, predicted_classes, target_names=class_names)
    cm = confusion_matrix(true_classes, predicted_classes)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes,
        'confusion_matrix': cm,
        'classification_report': report
    }

def visualize_predictions(model, test_generator, num_samples=8):
    """
    Visualize model predictions
    
    Args:
        model: Trained model
        test_generator: Test data generator
        num_samples: Number of samples to visualize
    """
    # Get batch of images
    images, labels = next(test_generator)
    
    # Make predictions
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)
    
    # Display results
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.flatten()
    
    for i in range(min(num_samples, len(images))):
        axes[i].imshow(images[i])
        axes[i].set_title(f'True: {true_classes[i]}\nPred: {predicted_classes[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
```

### Performance Optimization

```python
def optimize_vision_pipeline():
    """
    Optimize computer vision pipeline for performance
    
    Returns:
        Optimized pipeline components
    """
    # Use mixed precision training
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)
    
    # Optimize data loading
    AUTOTUNE = tf.data.AUTOTUNE
    
    # Use prefetching
    def optimize_dataset(dataset):
        return dataset.prefetch(AUTOTUNE)
    
    # Use caching for repeated data
    def cache_dataset(dataset):
        return dataset.cache()
    
    return {
        'mixed_precision': policy,
        'autotune': AUTOTUNE,
        'optimize_dataset': optimize_dataset,
        'cache_dataset': cache_dataset
    }

def create_efficient_model(input_shape, num_classes):
    """
    Create efficient vision model
    
    Args:
        input_shape: Input image shape
        num_classes: Number of classes
    
    Returns:
        Efficient model
    """
    # Use MobileNetV2 for efficiency
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape,
        alpha=0.75  # Reduced width multiplier
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Create efficient model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Use efficient optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

## Exercises

1. **Image Classification**: Build a CNN to classify images from a custom dataset.
2. **Object Detection**: Implement face detection using OpenCV Haar cascades.
3. **Image Segmentation**: Perform semantic segmentation on images using deep learning.
4. **Feature Extraction**: Extract and match SIFT features between two images.
5. **Transfer Learning**: Use a pre-trained model for a new image classification task.

## Next Steps

After mastering computer vision, explore:
- [Natural Language Processing](nlp_guide.md)
- [Reinforcement Learning](reinforcement_learning_guide.md)
- [Advanced Deep Learning Techniques](../PyTorch/advanced_pytorch_techniques_guide.md)
- [Model Deployment](model_deployment_guide.md) 