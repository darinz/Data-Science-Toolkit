# Deep Learning with PyTorch Neural Networks Tutorial

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Conda](https://img.shields.io/badge/Conda-Environment-44A833.svg?logo=anaconda)](https://docs.conda.io/)
[![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-orange.svg)](https://pytorch.org/tutorials/)
[![Duration](https://img.shields.io/badge/Duration-3--4%20hours-yellow.svg)](https://pytorch.org/tutorials/)

In this tutorial, we'll learn about ``torch.nn``, which is used to build Neural Networks for Deep Learning in PyTorch. It's essential to note that ``nn`` relies on ``autograd`` for model definition and differentiation. Review the ``torch.autograd`` [tutoral](https://github.com/darinz/DL-PT-Autograd) prior to starting this one.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Build neural networks using PyTorch's `torch.nn` module
- Define custom layers and activation functions
- Train neural networks with proper loss functions and optimizers
- Handle data loading and preprocessing
- Implement forward and backward passes
- Monitor training progress and metrics

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

- **Neural Network Architecture**: Design and implement various network architectures
- **Layer Types**: Work with different types of layers (Linear, Conv2d, etc.)
- **Activation Functions**: Use and understand different activation functions
- **Loss Functions**: Choose appropriate loss functions for different tasks
- **Optimizers**: Implement various optimization algorithms
- **Training Loops**: Build complete training pipelines

## Related Tutorials

- [Tensor Tutorial](../Tensor/) - Start here
- [Autograd Tutorial](../Autograd/) - Previous in the series
- [Image Classification Tutorial](../Image-Classifier/) - Next in the series

## Additional Resources

- [PyTorch nn Module Documentation](https://pytorch.org/docs/stable/nn.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## Citation

```bibtex

```

---

**Ready to build your first neural network? Open the `pt_neural_networks.ipynb` notebook!**
