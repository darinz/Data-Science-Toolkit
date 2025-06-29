# Deep Learning with PyTorch Tensor Tutorial

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![Conda](https://img.shields.io/badge/Conda-Environment-44A833.svg?logo=anaconda)](https://docs.conda.io/)
[![Difficulty](https://img.shields.io/badge/Difficulty-Beginner-green.svg)](https://pytorch.org/tutorials/)
[![Duration](https://img.shields.io/badge/Duration-1--2%20hours-yellow.svg)](https://pytorch.org/tutorials/)

In this tutorial, you will learn the following:
- Utilizing PyTorch tensors to encode inputs, outputs, and parameters for deep learning models.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Create and manipulate PyTorch tensors
- Perform basic tensor operations
- Understand tensor shapes and dimensions
- Convert between tensors and other data types
- Use tensors for mathematical operations

## Prerequisites

Before starting this tutorial it is recommended that you have installed [PyTorch](https://pytorch.org/) or use [Google Colab](https://colab.research.google.com/?utm_source=scs-index), and have a basic understanding of [Python programming language](https://www.python.org/doc/):

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

- **Tensor Creation**: Learn different ways to create tensors
- **Tensor Operations**: Perform mathematical operations on tensors
- **Shape Manipulation**: Reshape, transpose, and manipulate tensor dimensions
- **Data Types**: Work with different tensor data types
- **Device Management**: Move tensors between CPU and GPU

## Related Tutorials

- [Autograd Tutorial](../Autograd/) - Next in the series
- [Neural Networks Tutorial](../Neural-Networks/)
- [Image Classification Tutorial](../Image-Classifier/)

## References

- [PyTorch Tensor Documentation](https://pytorch.org/docs/stable/tensors.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

## Citation

```bibtex

```

---

**Ready to start? Open the `pt_tensor.ipynb` notebook and begin your PyTorch journey!**
