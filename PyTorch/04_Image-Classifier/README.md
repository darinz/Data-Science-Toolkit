# Deep Learning with PyTorch Image Classifier Tutorial

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Conda](https://img.shields.io/badge/Conda-Environment-44A833.svg?logo=anaconda)](https://docs.conda.io/)
[![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-orange.svg)](https://pytorch.org/tutorials/)
[![Duration](https://img.shields.io/badge/Duration-4--5%20hours-yellow.svg)](https://pytorch.org/tutorials/)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-lightgrey.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)

In this tutorial, we'll be building and training an image classifier using the CIFAR10 dataset, which includes classes such as 'airplane,' 'automobile,' 'bird,' 'cat,' 'deer,' 'dog,' 'frog,' 'horse,' 'ship,' and 'truck.' The images in CIFAR-10 have dimensions of 3x32x32, indicating they are 3-channel color images with a resolution of 32x32 pixels.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Build a complete image classification pipeline
- Work with the CIFAR-10 dataset
- Implement convolutional neural networks (CNNs)
- Train and evaluate image classification models
- Handle data augmentation and preprocessing
- Visualize training progress and results
- Deploy and use trained models

## Prerequisites

Before starting this tutorial it is recommended that you have installed [PyTorch](https://pytorch.org/) or use [Google Colab](https://colab.research.google.com/?utm_source=scs-index), and have a basic understanding of [Python programming language](https://www.python.org/doc/) and [PyTorch Tensors](https://github.com/darinz/DL-PT-Tensor):

### Google Colab

For tips on running tutorial notebooks in Google Colab, see [Colab Pytorch Tutorial](https://pytorch.org/tutorials/beginner/colab)

### Conda Environment Setup

Use the first command to create new independent environment for the project. Or use the other two commands to remove or update the Conda environment.

```shell
# To create a conda environment.
conda env create -f environment.yml

# To remove a conda environment.
conda remove --name dl --all

# To update a conda environment when some new libraries are added.
conda env update -f environment.yml --prune
```
Then, install [PyTorch](https://pytorch.org/).

## What You'll Learn

- **Computer Vision**: Understand the basics of image processing and classification
- **Convolutional Neural Networks**: Build and train CNNs for image recognition
- **Data Loading**: Efficiently load and preprocess image datasets
- **Data Augmentation**: Apply transformations to improve model generalization
- **Model Training**: Implement complete training loops with validation
- **Model Evaluation**: Assess model performance using various metrics
- **Model Deployment**: Save and load trained models for inference

## Related Tutorials

- [Tensor Tutorial](../Tensor/) - Start here
- [Autograd Tutorial](../Autograd/)
- [Neural Networks Tutorial](../Neural-Networks/) - Previous in the series

## Additional Resources

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Computer Vision Tutorials](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [PyTorch Vision Documentation](https://pytorch.org/vision/stable/index.html)

## Citation

```bibtex

```

---

**Ready to build your first image classifier? Open the `image_classifier.ipynb` notebook!**
