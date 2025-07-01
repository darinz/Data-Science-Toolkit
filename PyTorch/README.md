# PyTorch Deep Learning Tutorials

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Conda](https://img.shields.io/badge/Conda-Environment-44A833.svg?logo=anaconda)](https://docs.conda.io/)

A comprehensive collection of PyTorch deep learning tutorials designed to take you from beginner to advanced concepts in machine learning and neural networks.

## Tutorial Series

This repository contains a structured learning path through PyTorch fundamentals:

### 1. [Tensor Operations](./01_Tensor/) 
[![Tensor](https://img.shields.io/badge/Tensor-Operations-FF6B6B.svg)](./01_Tensor/)
Learn the fundamentals of PyTorch tensors - the building blocks of deep learning.

### 2. [Autograd & Automatic Differentiation](./02_Autograd/)
[![Autograd](https://img.shields.io/badge/Autograd-Differentiation-4ECDC4.svg)](./02_Autograd/)
Master PyTorch's automatic differentiation engine for training neural networks.

### 3. [Neural Networks](./03_Neural-Networks/)
[![Neural Networks](https://img.shields.io/badge/Neural-Networks-45B7D1.svg)](./03_Neural-Networks/)
Build and train neural networks using PyTorch's `torch.nn` module.

### 4. [Image Classification](./04_Image-Classifier/)
[![Image Classification](https://img.shields.io/badge/Image-Classification-96CEB4.svg)](./04_Image-Classifier/)
Create a complete image classification system using CIFAR-10 dataset.

## Quick Start

### Prerequisites
- Python 3.10 or higher
- Basic understanding of Python programming
- Familiarity with machine learning concepts (recommended)

### Installation

#### Option 1: Using Conda (Recommended)
```bash
# Clone the repository
git clone https://github.com/darinz/PyTorch.git
cd PyTorch

# Create environment for any tutorial
cd 01_Tensor  # or 02_Autograd, 03_Neural-Networks, 04_Image-Classifier
conda env create -f environment.yml
conda activate dl
```

#### Option 2: Using Google Colab
All tutorials are compatible with Google Colab. Simply upload the `.ipynb` files to your Colab workspace.

## Learning Path

We recommend following the tutorials in this order:

1. **Tensor** → **Autograd** → **Neural Networks** → **Image Classification**

Each tutorial builds upon the previous one, ensuring a solid foundation in PyTorch concepts.

## Environment Management

### Common Commands
```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate dl

# Update environment
conda env update -f environment.yml --prune

# Remove environment
conda remove --name dl --all

# List environments
conda env list
```

## Tutorial Details

| Tutorial | Description | Difficulty | Duration |
|----------|-------------|------------|----------|
| [Tensor](./01_Tensor/) | PyTorch tensor operations and manipulation | Beginner | 1-2 hours |
| [Autograd](./02_Autograd/) | Automatic differentiation and gradients | Beginner | 2-3 hours |
| [Neural Networks](./03_Neural-Networks/) | Building and training neural networks | Intermediate | 3-4 hours |
| [Image Classification](./04_Image-Classifier/) | Complete image classification project | Intermediate | 4-5 hours |

## What You'll Learn

- **Tensor Operations**: Manipulate multi-dimensional arrays efficiently
- **Automatic Differentiation**: Understand how gradients are computed automatically
- **Neural Network Architecture**: Design and implement various network architectures
- **Model Training**: Train models with proper loss functions and optimizers
- **Image Classification**: Build a complete computer vision pipeline
- **Best Practices**: Learn PyTorch conventions and optimization techniques

## Additional Resources

- [PyTorch Official Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Google Colab PyTorch Guide](https://pytorch.org/tutorials/beginner/colab)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the excellent framework
- CIFAR-10 dataset creators
- The open-source community for continuous improvements

---

**Happy Learning!**
