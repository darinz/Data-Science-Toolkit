# NumPy Random Generation: A Comprehensive Guide

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](../LICENSE)

## Table of Contents

1. [Introduction to Random Generation](#introduction-to-random-generation)
2. [Random Number Generator Basics](#random-number-generator-basics)
3. [Uniform Distribution](#uniform-distribution)
4. [Normal (Gaussian) Distribution](#normal-gaussian-distribution)
5. [Other Probability Distributions](#other-probability-distributions)
6. [Random Array Generation](#random-array-generation)
7. [Statistical Sampling](#statistical-sampling)
8. [Seeding and Reproducibility](#seeding-and-reproducibility)
9. [Monte Carlo Simulations](#monte-carlo-simulations)
10. [Applications in Data Science](#applications-in-data-science)
11. [Best Practices](#best-practices)
12. [Common Pitfalls](#common-pitfalls)

## Introduction to Random Generation

Random number generation is fundamental to many applications including:
- Statistical sampling and analysis
- Monte Carlo simulations
- Machine learning and data augmentation
- Cryptography and security
- Game development and modeling
- Scientific computing and research

NumPy provides a comprehensive random number generation system through the `numpy.random` module, offering various probability distributions and sampling methods.

## Random Number Generator Basics

### Setting Random Seed

Setting a random seed ensures reproducible results:

```python
import numpy as np
np.random.seed(42)  # Set seed for reproducibility
print(np.random.random(5))
```

### Random State Objects

For isolated random number generation:

```python
rng1 = np.random.RandomState(42)
rng2 = np.random.RandomState(123)
print(rng1.random(3))
print(rng2.random(3))
```

### Random Number Quality

```python
large_sample = np.random.random(10000)
print(f"Mean: {np.mean(large_sample):.4f}")
print(f"Std: {np.std(large_sample):.4f}")
```

## Uniform Distribution

The uniform distribution is the most basic probability distribution, where all values in a range are equally likely.

```python
# Uniform [0, 1)
print(np.random.random(5))
# Uniform [low, high)
print(np.random.uniform(10, 20, 5))
# Uniform integers [low, high)
print(np.random.randint(1, 101, 5))
```

## Normal (Gaussian) Distribution

The normal distribution is one of the most important probability distributions in statistics and natural sciences.

```python
# Standard normal (mean=0, std=1)
print(np.random.randn(5))
# Custom normal distribution
print(np.random.normal(100, 15, 5))
```

## Other Probability Distributions

NumPy supports many other distributions:

```python
# Exponential
print(np.random.exponential(scale=1.0, size=5))
# Binomial
print(np.random.binomial(n=10, p=0.5, size=5))
# Poisson
print(np.random.poisson(lam=3.0, size=5))
# Beta
print(np.random.beta(a=0.5, b=0.5, size=5))
# Gamma
print(np.random.gamma(shape=2.0, scale=2.0, size=5))
```

## Random Array Generation

Generate random arrays of any shape:

```python
# 2D array of uniform random numbers
print(np.random.random((3, 4)))
# 3D array of normal random numbers
print(np.random.normal(0, 1, (2, 3, 4)))
```

## Statistical Sampling

Sampling from arrays:

```python
arr = np.arange(10)
# Random permutation
print(np.random.permutation(arr))
# Random choice
print(np.random.choice(arr, size=5, replace=False))
# Shuffle in-place
np.random.shuffle(arr)
print(arr)
```

## Seeding and Reproducibility

Always set a seed for reproducibility in experiments:

```python
np.random.seed(123)
print(np.random.random(3))
```

## Monte Carlo Simulations

Monte Carlo methods use random sampling to estimate results:

```python
# Estimate Pi using Monte Carlo
n = 100000
x = np.random.uniform(-1, 1, n)
y = np.random.uniform(-1, 1, n)
inside = (x**2 + y**2) <= 1
pi_estimate = 4 * np.sum(inside) / n
print(f"Estimated Pi: {pi_estimate}")
```

## Applications in Data Science

- Data augmentation (random crops, flips, noise)
- Bootstrapping and resampling
- Synthetic data generation
- Randomized algorithms
- Model initialization

## Best Practices

- Always set a seed for reproducibility in research
- Use vectorized random generation for performance
- Prefer `numpy.random.Generator` (NumPy 1.17+) for new code
- Use appropriate distributions for your application

## Common Pitfalls

- Forgetting to set a seed (results not reproducible)
- Using global random state in parallel code (use separate generators)
- Misunderstanding distribution parameters (e.g., scale vs. std)
- Sampling with replacement when you mean without (or vice versa)

---

This guide covers the essential random generation techniques in NumPy. Practice these concepts to master random sampling and simulation for data science and scientific computing. 