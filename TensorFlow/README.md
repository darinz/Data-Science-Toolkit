# TensorFlow Tutorials

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.15+-red.svg)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Conda](https://img.shields.io/badge/Conda-Environment-green.svg)](https://docs.conda.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

A comprehensive collection of TensorFlow tutorials covering deep learning fundamentals, neural networks, and practical applications.

## What's Included

### Core TensorFlow Concepts
- **TensorFlow Basics** - Understanding tensors, operations, and computational graphs
- **Neural Networks** - Building and training neural networks from scratch
- **Deep Learning Models** - CNN, RNN, LSTM, and Transformer implementations
- **Model Training** - Loss functions, optimizers, and training loops
- **Data Pipelines** - Efficient data loading and preprocessing with tf.data

### Practical Applications
- **Image Classification** - Convolutional Neural Networks for computer vision
- **Natural Language Processing** - Text classification and generation
- **Time Series Forecasting** - LSTM and GRU models for sequential data
- **Transfer Learning** - Using pre-trained models for custom tasks

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Basic understanding of Python and machine learning concepts
- GPU support (optional but recommended for deep learning)

### Installation

**Using Conda (Recommended):**
```bash
cd TensorFlow
conda env create -f environment.yml
conda activate tf-tutorials
```

**Using pip:**
```bash
cd TensorFlow
pip install -r requirements.txt
```

### Verify Installation
```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

## File Structure

```
TensorFlow/
├── tensorflow_basics_guide.md         # TensorFlow fundamentals and basics
├── neural_networks_guide.md           # Building and training neural networks
├── cnn_guide.md                       # Convolutional Neural Networks
├── rnn_guide.md                       # Recurrent Neural Networks
├── data_pipelines_guide.md            # Data loading and preprocessing
├── advanced_guide.md                  # Advanced TensorFlow techniques
├── environment.yml                    # Conda environment configuration
├── requirements.txt                   # pip dependencies
└── README.md                          # This file
```

## Tutorial Structure

### 1. TensorFlow Basics (`tensorflow_basics_guide.md`)
- Understanding tensors and operations
- Computational graphs and eager execution
- Basic mathematical operations
- Variable and constant tensors

### 2. Neural Networks (`neural_networks_guide.md`)
- Building neural networks with Keras
- Dense layers and activation functions
- Model compilation and training
- Evaluation and prediction

### 3. Convolutional Neural Networks (`cnn_guide.md`)
- CNN architecture fundamentals
- Image preprocessing and augmentation
- Training CNNs for image classification
- Transfer learning with pre-trained models

### 4. Recurrent Neural Networks (`rnn_guide.md`)
- RNN, LSTM, and GRU architectures
- Sequential data processing
- Time series forecasting
- Text generation and classification

### 5. Data Pipelines (`data_pipelines_guide.md`)
- Efficient data loading with tf.data
- Data preprocessing and augmentation
- Performance optimization
- Custom data generators

### 6. Advanced TensorFlow (`advanced_guide.md`)
- Custom training loops
- Model subclassing
- Advanced optimization techniques
- Model deployment strategies

## Running the Tutorials

### Reading the Guides
These are comprehensive markdown guides that you can read directly or follow along with your own code:

```bash
# Open guides in your preferred markdown viewer
# Or use a text editor/IDE to read them
```

### Interactive Learning
To follow along with the examples in the guides:

```bash
# Start Jupyter Lab for interactive coding
jupyter lab

# Or use Google Colab for cloud-based development
# Upload code examples to https://colab.research.google.com/
```

### Sequential Learning
For comprehensive learning, follow the guides in order:

1. Start with `tensorflow_basics_guide.md`
2. Progress to `neural_networks_guide.md`
3. Explore `cnn_guide.md` for computer vision
4. Study `rnn_guide.md` for sequential data
5. Learn `data_pipelines_guide.md` for efficient data handling
6. Master `advanced_guide.md` for advanced techniques

## Environment Management

### Conda Commands

```bash
# Create new environment
conda env create -f environment.yml

# Activate environment
conda activate tf-tutorials

# Update environment (when dependencies change)
conda env update -f environment.yml --prune

# Remove environment
conda remove --name tf-tutorials --all

# List all environments
conda env list

# Deactivate current environment
conda deactivate
```

### pip Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Check installed packages
pip list

# Export current environment
pip freeze > requirements.txt

# Install from requirements
pip install -r requirements.txt
```

## Learning Path

### Beginner Path
1. **Start with TensorFlow Basics** - Understand core concepts
2. **Build Neural Networks** - Learn Keras API and model building
3. **Practice with Simple Models** - Train basic classification models
4. **Understand Data Pipelines** - Learn efficient data handling

### Intermediate Path
1. **Explore CNNs** - Master computer vision applications
2. **Dive into RNNs** - Handle sequential data
3. **Optimize Data Pipelines** - Improve training efficiency
4. **Experiment with Transfer Learning** - Use pre-trained models

### Advanced Path
1. **Custom Model Architectures** - Build complex neural networks
2. **Advanced Training Techniques** - Custom training loops
3. **Model Deployment** - Deploy models to production
4. **Research Applications** - Apply to cutting-edge problems

## Key Concepts Covered

### TensorFlow Fundamentals
- **Tensors** - Multi-dimensional arrays and operations
- **Computational Graphs** - Symbolic computation and optimization
- **Eager Execution** - Immediate evaluation and debugging
- **Automatic Differentiation** - Gradient computation for training

### Neural Network Architecture
- **Layers** - Building blocks of neural networks
- **Activation Functions** - Non-linear transformations
- **Loss Functions** - Measuring model performance
- **Optimizers** - Updating model parameters

### Deep Learning Models
- **Convolutional Networks** - Image processing and computer vision
- **Recurrent Networks** - Sequential data and time series
- **Attention Mechanisms** - Focus on relevant information
- **Transformer Models** - State-of-the-art NLP architectures

### Training and Optimization
- **Backpropagation** - Computing gradients efficiently
- **Regularization** - Preventing overfitting
- **Hyperparameter Tuning** - Optimizing model performance
- **Model Checkpointing** - Saving and loading models

## Common Use Cases

### Basic Neural Network
```python
import tensorflow as tf
from tensorflow import keras

# Create a simple neural network
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
```

### Convolutional Neural Network
```python
import tensorflow as tf
from tensorflow import keras

# Create a CNN
model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile and train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

### Recurrent Neural Network
```python
import tensorflow as tf
from tensorflow import keras

# Create an LSTM model
model = keras.Sequential([
    keras.layers.LSTM(64, return_sequences=True, input_shape=(None, 1)),
    keras.layers.LSTM(64),
    keras.layers.Dense(1)
])

# Compile for regression
model.compile(optimizer='adam', loss='mse')
```

## Integration with Other Libraries

### NumPy Integration
```python
import tensorflow as tf
import numpy as np

# Convert between NumPy and TensorFlow
numpy_array = np.random.randn(100, 10)
tensor = tf.convert_to_tensor(numpy_array)

# Use TensorFlow operations
result = tf.reduce_mean(tensor, axis=0)
numpy_result = result.numpy()
```

### pandas Integration
```python
import tensorflow as tf
import pandas as pd

# Load data with pandas
df = pd.read_csv('data.csv')
features = df.drop('target', axis=1).values
target = df['target'].values

# Convert to TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((features, target))
dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
```

### Matplotlib Integration
```python
import tensorflow as tf
import matplotlib.pyplot as plt

# Plot training history
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

## Additional Resources

### Official Documentation
- [TensorFlow Official Documentation](https://www.tensorflow.org/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Hub](https://tfhub.dev/) - Pre-trained models

### Learning Resources
- [TensorFlow GitHub Repository](https://github.com/tensorflow/tensorflow)
- [TensorFlow Community](https://www.tensorflow.org/community)
- [Google Colab TensorFlow Guide](https://www.tensorflow.org/tutorials/quickstart/beginner)

### Recommended Books
- "Hands-On Machine Learning" by Aurélien Géron
- "Deep Learning with Python" by François Chollet
- "TensorFlow in Action" by Thushan Ganegedara

## Contributing

Found an error or have a suggestion? Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

**Happy Deep Learning!** 