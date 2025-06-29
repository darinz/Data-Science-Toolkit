# Deep Learning with PyTorch Autograd Tutorial

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Conda](https://img.shields.io/badge/Conda-Environment-44A833.svg?logo=anaconda)](https://docs.conda.io/)
[![Difficulty](https://img.shields.io/badge/Difficulty-Beginner-green.svg)](https://pytorch.org/tutorials/)
[![Duration](https://img.shields.io/badge/Duration-2--3%20hours-yellow.svg)](https://pytorch.org/tutorials/)

In this tutorial, we'll learn about ``torch.autograd``, which serves as PyTorch's automatic differentiation engine, driving the training of neural networks. This tutorial aims to provide you with a conceptual grasp of how autograd contributes to the training process of a neural network.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Understand how automatic differentiation works in PyTorch
- Compute gradients automatically using autograd
- Control gradient computation with `requires_grad`
- Work with gradient accumulation and clearing
- Debug gradient computation issues

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

- **Automatic Differentiation**: Understand how PyTorch computes gradients automatically
- **Gradient Computation**: Learn to compute gradients for complex functions
- **Gradient Control**: Control when and how gradients are computed
- **Backward Pass**: Understand the backward propagation mechanism
- **Gradient Accumulation**: Handle gradient accumulation in training loops

## Related Tutorials

- [Tensor Tutorial](../Tensor/) - Previous in the series
- [Neural Networks Tutorial](../Neural-Networks/) - Next in the series
- [Image Classification Tutorial](../Image-Classifier/)

## Further Readings

-  [In-place operations & Multithreaded Autograd](https://pytorch.org/docs/stable/notes/autograd.html)
-  [Example implementation of reverse-mode autodiff](https://colab.research.google.com/drive/1VpeE6UvEPRz9HmsHh1KS0XxXjYu533EC)
-  [Video: PyTorch Autograd Explained - In-depth Tutorial](https://www.youtube.com/watch?v=MswxJw-8PvE)

## Citation

```bibtex

```

---

**Ready to dive into automatic differentiation? Open the `pt_autograd.ipynb` notebook!**
