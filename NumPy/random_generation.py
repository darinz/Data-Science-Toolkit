#!/usr/bin/env python3
"""
NumPy Random Generation: Probability Distributions and Statistical Sampling

Welcome to the NumPy random generation tutorial! This tutorial covers 
comprehensive random number generation, probability distributions, and 
statistical sampling techniques using NumPy.

This script covers:
- Random number generation basics
- Probability distributions (uniform, normal, exponential, etc.)
- Statistical sampling methods
- Random array generation
- Seeding and reproducibility
- Monte Carlo simulations
- Applications in data science and machine learning

Prerequisites:
- Python 3.8 or higher
- Basic understanding of NumPy (covered in numpy_basics.py)
- Basic probability and statistics concepts
- NumPy installed (pip install numpy)
"""

import numpy as np
import sys
import matplotlib.pyplot as plt

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_subsection_header(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")

def main():
    """Main function to run all tutorial sections."""
    
    print("NumPy Random Generation Tutorial")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print("Random generation tutorial started successfully!")

    # Section 1: Introduction to Random Generation
    print_section_header("1. Introduction to Random Generation")
    
    print("""
Random number generation is fundamental to many applications including:
- Statistical sampling and analysis
- Monte Carlo simulations
- Machine learning and data augmentation
- Cryptography and security
- Game development and modeling
- Scientific computing and research

NumPy provides a comprehensive random number generation system through 
numpy.random module, offering various probability distributions and 
sampling methods.

Key Features:
✅ Multiple probability distributions
✅ Efficient vectorized operations
✅ Reproducible results with seeding
✅ High-quality random number generators
✅ Statistical sampling methods
✅ Integration with NumPy arrays
""")

    # Section 2: Random Number Generator Basics
    print_section_header("2. Random Number Generator Basics")
    
    print("""
Understanding the basics of random number generation and how to control 
reproducibility is essential for scientific computing.
""")

    print_subsection_header("Setting Random Seed")
    
    print("Setting a random seed ensures reproducible results:")
    print("```python")
    print("np.random.seed(42)  # Set seed for reproducibility")
    print("```")
    
    # Set seed for reproducible results
    np.random.seed(42)
    
    # Generate some random numbers
    random_numbers = np.random.random(5)
    print(f"Random numbers with seed 42: {random_numbers}")
    
    # Reset seed and generate again
    np.random.seed(42)
    random_numbers_again = np.random.random(5)
    print(f"Random numbers with same seed: {random_numbers_again}")
    print(f"Results are identical: {np.array_equal(random_numbers, random_numbers_again)}")

    print_subsection_header("Random State Objects")
    
    print("Using RandomState objects for isolated random number generation:")
    
    # Create separate random state objects
    rng1 = np.random.RandomState(42)
    rng2 = np.random.RandomState(123)
    
    print(f"RNG1 (seed 42): {rng1.random(3)}")
    print(f"RNG2 (seed 123): {rng2.random(3)}")
    print(f"RNG1 again: {rng1.random(3)}")  # Different from first call

    print_subsection_header("Random Number Quality")
    
    print("Checking the quality of random number generation:")
    
    # Generate a large number of random values
    large_sample = np.random.random(10000)
    
    print(f"Sample statistics:")
    print(f"  - Mean: {np.mean(large_sample):.4f} (expected: ~0.5)")
    print(f"  - Standard deviation: {np.std(large_sample):.4f} (expected: ~0.289)")
    print(f"  - Minimum: {np.min(large_sample):.4f}")
    print(f"  - Maximum: {np.max(large_sample):.4f}")
    print(f"  - Range: {np.max(large_sample) - np.min(large_sample):.4f}")

    # Section 3: Uniform Distribution
    print_section_header("3. Uniform Distribution")
    
    print("""
The uniform distribution is the most basic probability distribution, 
where all values in a range are equally likely.
""")

    print_subsection_header("Uniform Random Numbers")
    
    print("Generating uniform random numbers:")
    print("```python")
    print("np.random.random(size)  # Uniform [0, 1)")
    print("np.random.uniform(low, high, size)  # Uniform [low, high)")
    print("np.random.randint(low, high, size)  # Uniform integers [low, high)")
    print("```")
    
    # Uniform random numbers between 0 and 1
    uniform_01 = np.random.random(5)
    print(f"Uniform [0, 1): {uniform_01}")
    
    # Uniform random numbers in custom range
    uniform_custom = np.random.uniform(10, 20, 5)
    print(f"Uniform [10, 20): {uniform_custom}")
    
    # Uniform random integers
    uniform_ints = np.random.randint(1, 101, 5)
    print(f"Uniform integers [1, 101): {uniform_ints}")

    print_subsection_header("Uniform Distribution Properties")
    
    # Generate large sample for analysis
    uniform_sample = np.random.uniform(0, 10, 10000)
    
    print(f"Uniform distribution sample (0 to 10):")
    print(f"  - Mean: {np.mean(uniform_sample):.3f} (expected: 5.0)")
    print(f"  - Variance: {np.var(uniform_sample):.3f} (expected: 8.33)")
    print(f"  - Standard deviation: {np.std(uniform_sample):.3f} (expected: 2.89)")
    
    # Check distribution shape
    print(f"  - 25th percentile: {np.percentile(uniform_sample, 25):.3f}")
    print(f"  - 50th percentile (median): {np.percentile(uniform_sample, 50):.3f}")
    print(f"  - 75th percentile: {np.percentile(uniform_sample, 75):.3f}")

    # Section 4: Normal (Gaussian) Distribution
    print_section_header("4. Normal (Gaussian) Distribution")
    
    print("""
The normal distribution is one of the most important probability 
distributions in statistics and natural sciences.
""")

    print_subsection_header("Normal Distribution Generation")
    
    print("Generating normal random numbers:")
    print("```python")
    print("np.random.normal(mean, std, size)  # Normal distribution")
    print("np.random.randn(size)  # Standard normal (mean=0, std=1)")
    print("```")
    
    # Standard normal distribution
    standard_normal = np.random.randn(5)
    print(f"Standard normal (mean=0, std=1): {standard_normal}")
    
    # Custom normal distribution
    custom_normal = np.random.normal(100, 15, 5)
    print(f"Normal (mean=100, std=15): {custom_normal}")

    print_subsection_header("Normal Distribution Properties")
    
    # Generate large sample
    normal_sample = np.random.normal(50, 10, 10000)
    
    print(f"Normal distribution sample (mean=50, std=10):")
    print(f"  - Mean: {np.mean(normal_sample):.3f} (expected: 50.0)")
    print(f"  - Standard deviation: {np.std(normal_sample):.3f} (expected: 10.0)")
    print(f"  - Variance: {np.var(normal_sample):.3f} (expected: 100.0)")
    
    # Check empirical rule (68-95-99.7 rule)
    within_1std = np.sum(np.abs(normal_sample - 50) <= 10)
    within_2std = np.sum(np.abs(normal_sample - 50) <= 20)
    within_3std = np.sum(np.abs(normal_sample - 50) <= 30)
    
    print(f"  - Within 1 std: {within_1std/10000*100:.1f}% (expected: 68.3%)")
    print(f"  - Within 2 std: {within_2std/10000*100:.1f}% (expected: 95.4%)")
    print(f"  - Within 3 std: {within_3std/10000*100:.1f}% (expected: 99.7%)")

    print_subsection_header("Multivariate Normal Distribution")
    
    print("Generating multivariate normal distributions:")
    
    # Define mean and covariance
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    
    # Generate multivariate normal samples
    multivariate_normal = np.random.multivariate_normal(mean, cov, 1000)
    
    print(f"Multivariate normal sample shape: {multivariate_normal.shape}")
    print(f"Sample mean: {np.mean(multivariate_normal, axis=0)}")
    print(f"Sample covariance:\n{np.cov(multivariate_normal.T)}")

    # Section 5: Other Common Distributions
    print_section_header("5. Other Common Distributions")
    
    print("""
NumPy provides many other probability distributions for various 
applications in statistics and modeling.
""")

    print_subsection_header("Exponential Distribution")
    
    print("Exponential distribution for modeling time between events:")
    
    # Generate exponential random numbers
    exponential_sample = np.random.exponential(scale=2.0, size=1000)
    
    print(f"Exponential distribution (scale=2.0):")
    print(f"  - Mean: {np.mean(exponential_sample):.3f} (expected: 2.0)")
    print(f"  - Standard deviation: {np.std(exponential_sample):.3f} (expected: 2.0)")
    print(f"  - Median: {np.median(exponential_sample):.3f}")

    print_subsection_header("Poisson Distribution")
    
    print("Poisson distribution for counting rare events:")
    
    # Generate Poisson random numbers
    poisson_sample = np.random.poisson(lam=5.0, size=1000)
    
    print(f"Poisson distribution (λ=5.0):")
    print(f"  - Mean: {np.mean(poisson_sample):.3f} (expected: 5.0)")
    print(f"  - Variance: {np.var(poisson_sample):.3f} (expected: 5.0)")
    print(f"  - Unique values: {np.unique(poisson_sample)}")

    print_subsection_header("Binomial Distribution")
    
    print("Binomial distribution for modeling success/failure experiments:")
    
    # Generate binomial random numbers
    binomial_sample = np.random.binomial(n=10, p=0.3, size=1000)
    
    print(f"Binomial distribution (n=10, p=0.3):")
    print(f"  - Mean: {np.mean(binomial_sample):.3f} (expected: 3.0)")
    print(f"  - Variance: {np.var(binomial_sample):.3f} (expected: 2.1)")
    print(f"  - Unique values: {np.unique(binomial_sample)}")

    print_subsection_header("Gamma Distribution")
    
    print("Gamma distribution for modeling waiting times and reliability:")
    
    # Generate gamma random numbers
    gamma_sample = np.random.gamma(shape=2.0, scale=1.0, size=1000)
    
    print(f"Gamma distribution (shape=2.0, scale=1.0):")
    print(f"  - Mean: {np.mean(gamma_sample):.3f} (expected: 2.0)")
    print(f"  - Variance: {np.var(gamma_sample):.3f} (expected: 2.0)")

    print_subsection_header("Beta Distribution")
    
    print("Beta distribution for modeling probabilities and proportions:")
    
    # Generate beta random numbers
    beta_sample = np.random.beta(a=2.0, b=5.0, size=1000)
    
    print(f"Beta distribution (α=2.0, β=5.0):")
    print(f"  - Mean: {np.mean(beta_sample):.3f} (expected: 0.286)")
    print(f"  - Range: [{np.min(beta_sample):.3f}, {np.max(beta_sample):.3f}]")

    # Section 6: Random Array Generation
    print_section_header("6. Random Array Generation")
    
    print("""
NumPy provides convenient functions for generating random arrays 
with specific shapes and distributions.
""")

    print_subsection_header("Random Array Shapes")
    
    print("Generating random arrays with different shapes:")
    
    # 1D array
    arr_1d = np.random.rand(5)
    print(f"1D array: {arr_1d}")
    
    # 2D array
    arr_2d = np.random.rand(3, 4)
    print(f"2D array:\n{arr_2d}")
    
    # 3D array
    arr_3d = np.random.rand(2, 3, 4)
    print(f"3D array shape: {arr_3d.shape}")
    print(f"3D array (first slice):\n{arr_3d[0]}")

    print_subsection_header("Random Array with Different Distributions")
    
    print("Generating arrays with different distributions:")
    
    # Normal distribution array
    normal_array = np.random.normal(0, 1, (3, 3))
    print(f"Normal array:\n{normal_array}")
    
    # Integer array
    int_array = np.random.randint(1, 100, (3, 3))
    print(f"Integer array:\n{int_array}")
    
    # Choice from array
    choices = ['A', 'B', 'C', 'D']
    choice_array = np.random.choice(choices, (3, 3))
    print(f"Choice array:\n{choice_array}")

    print_subsection_header("Random Permutations")
    
    print("Generating random permutations:")
    
    # Permute array
    original = np.arange(10)
    permuted = np.random.permutation(original)
    print(f"Original: {original}")
    print(f"Permuted: {permuted}")
    
    # Shuffle in place
    arr_to_shuffle = np.arange(5)
    np.random.shuffle(arr_to_shuffle)
    print(f"Shuffled: {arr_to_shuffle}")

    # Section 7: Statistical Sampling
    print_section_header("7. Statistical Sampling")
    
    print("""
Statistical sampling methods are essential for data analysis, 
survey design, and machine learning.
""")

    print_subsection_header("Simple Random Sampling")
    
    print("Simple random sampling without replacement:")
    
    # Create population
    population = np.arange(1000)
    print(f"Population size: {len(population)}")
    
    # Simple random sample
    sample_size = 100
    sample = np.random.choice(population, size=sample_size, replace=False)
    print(f"Sample size: {len(sample)}")
    print(f"Sample mean: {np.mean(sample):.2f}")
    print(f"Population mean: {np.mean(population):.2f}")

    print_subsection_header("Stratified Sampling")
    
    print("Stratified sampling example:")
    
    # Create stratified population
    strata_1 = np.random.normal(50, 10, 600)  # 60% of population
    strata_2 = np.random.normal(80, 15, 400)  # 40% of population
    population_stratified = np.concatenate([strata_1, strata_2])
    
    print(f"Stratified population size: {len(population_stratified)}")
    print(f"Population mean: {np.mean(population_stratified):.2f}")
    
    # Stratified sample
    sample_strata_1 = np.random.choice(strata_1, size=60, replace=False)
    sample_strata_2 = np.random.choice(strata_2, size=40, replace=False)
    stratified_sample = np.concatenate([sample_strata_1, sample_strata_2])
    
    print(f"Stratified sample size: {len(stratified_sample)}")
    print(f"Stratified sample mean: {np.mean(stratified_sample):.2f}")

    print_subsection_header("Bootstrap Sampling")
    
    print("Bootstrap sampling for statistical inference:")
    
    # Original sample
    original_sample = np.random.normal(100, 15, 50)
    print(f"Original sample mean: {np.mean(original_sample):.2f}")
    
    # Bootstrap samples
    n_bootstrap = 1000
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(original_sample, size=len(original_sample), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    print(f"Bootstrap results:")
    print(f"  - Mean of bootstrap means: {np.mean(bootstrap_means):.2f}")
    print(f"  - Standard error: {np.std(bootstrap_means):.2f}")
    print(f"  - 95% confidence interval: [{np.percentile(bootstrap_means, 2.5):.2f}, {np.percentile(bootstrap_means, 97.5):.2f}]")

    # Section 8: Monte Carlo Simulations
    print_section_header("8. Monte Carlo Simulations")
    
    print("""
Monte Carlo simulations use random sampling to solve mathematical 
problems and estimate complex quantities.
""")

    print_subsection_header("Estimating π using Monte Carlo")
    
    print("Estimating π using Monte Carlo method:")
    
    # Generate random points in unit square
    n_points = 100000
    x = np.random.uniform(-1, 1, n_points)
    y = np.random.uniform(-1, 1, n_points)
    
    # Calculate distance from origin
    distances = np.sqrt(x**2 + y**2)
    
    # Count points inside unit circle
    inside_circle = np.sum(distances <= 1)
    
    # Estimate π
    pi_estimate = 4 * inside_circle / n_points
    print(f"Number of points: {n_points}")
    print(f"Points inside circle: {inside_circle}")
    print(f"Estimated π: {pi_estimate:.6f}")
    print(f"Actual π: {np.pi:.6f}")
    print(f"Error: {abs(pi_estimate - np.pi):.6f}")

    print_subsection_header("Monte Carlo Integration")
    
    print("Monte Carlo integration example:")
    
    # Define function to integrate: f(x) = x^2
    def f(x):
        return x**2
    
    # Monte Carlo integration
    n_samples = 100000
    x_samples = np.random.uniform(0, 2, n_samples)
    y_samples = f(x_samples)
    
    # Estimate integral
    integral_estimate = 2 * np.mean(y_samples)  # interval width * mean height
    actual_integral = 8/3  # ∫(0 to 2) x^2 dx = 8/3
    
    print(f"Monte Carlo integral estimate: {integral_estimate:.6f}")
    print(f"Actual integral: {actual_integral:.6f}")
    print(f"Error: {abs(integral_estimate - actual_integral):.6f}")

    print_subsection_header("Random Walk Simulation")
    
    print("Simulating a random walk:")
    
    # Parameters
    n_steps = 1000
    n_walks = 100
    
    # Simulate random walks
    walks = np.zeros((n_walks, n_steps))
    
    for i in range(n_walks):
        steps = np.random.choice([-1, 1], size=n_steps)
        walks[i] = np.cumsum(steps)
    
    # Analyze results
    final_positions = walks[:, -1]
    print(f"Random walk simulation:")
    print(f"  - Number of walks: {n_walks}")
    print(f"  - Steps per walk: {n_steps}")
    print(f"  - Mean final position: {np.mean(final_positions):.2f}")
    print(f"  - Standard deviation of final positions: {np.std(final_positions):.2f}")

    # Section 9: Applications in Data Science
    print_section_header("9. Applications in Data Science")
    
    print("""
Random generation is widely used in data science for data augmentation, 
model validation, and uncertainty quantification.
""")

    print_subsection_header("Data Augmentation")
    
    print("Data augmentation for machine learning:")
    
    # Original data
    original_data = np.random.normal(0, 1, 100)
    print(f"Original data size: {len(original_data)}")
    print(f"Original data mean: {np.mean(original_data):.3f}")
    
    # Add noise for augmentation
    noise_level = 0.1
    augmented_data = original_data + np.random.normal(0, noise_level, len(original_data))
    
    print(f"Augmented data size: {len(augmented_data)}")
    print(f"Augmented data mean: {np.mean(augmented_data):.3f}")
    print(f"Noise level: {noise_level}")

    print_subsection_header("Cross-Validation Sampling")
    
    print("Generating cross-validation folds:")
    
    # Create dataset
    dataset_size = 1000
    indices = np.arange(dataset_size)
    
    # Generate random folds
    n_folds = 5
    np.random.shuffle(indices)
    fold_size = dataset_size // n_folds
    
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < n_folds - 1 else dataset_size
        
        test_indices = indices[start_idx:end_idx]
        train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
        
        print(f"Fold {i+1}: Train size = {len(train_indices)}, Test size = {len(test_indices)}")

    print_subsection_header("Confidence Intervals")
    
    print("Generating confidence intervals using bootstrap:")
    
    # Sample data
    data = np.random.normal(100, 15, 50)
    
    # Bootstrap confidence interval for mean
    n_bootstrap = 1000
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Calculate confidence intervals
    ci_90 = np.percentile(bootstrap_means, [5, 95])
    ci_95 = np.percentile(bootstrap_means, [2.5, 97.5])
    ci_99 = np.percentile(bootstrap_means, [0.5, 99.5])
    
    print(f"Sample mean: {np.mean(data):.2f}")
    print(f"90% confidence interval: [{ci_90[0]:.2f}, {ci_90[1]:.2f}]")
    print(f"95% confidence interval: [{ci_95[0]:.2f}, {ci_95[1]:.2f}]")
    print(f"99% confidence interval: [{ci_99[0]:.2f}, {ci_99[1]:.2f}]")

    # Section 10: Best Practices and Tips
    print_section_header("10. Best Practices and Tips")
    
    print("""
Follow these best practices for reliable and efficient random generation:
""")

    print_subsection_header("Reproducibility")
    
    print("""
1. Always set seeds for reproducible results:
   ```python
   np.random.seed(42)  # For reproducible results
   ```

2. Use separate RandomState objects for different components:
   ```python
   rng1 = np.random.RandomState(42)
   rng2 = np.random.RandomState(123)
   ```

3. Document your random generation approach:
   ```python
   # Set seed for reproducibility
   np.random.seed(42)
   
   # Generate training data
   X_train = np.random.normal(0, 1, (1000, 10))
   y_train = np.random.binomial(1, 0.5, 1000)
   ```
""")

    print_subsection_header("Performance Optimization")
    
    print("""
1. Generate arrays in one call when possible:
   ```python
   # Good - single call
   large_array = np.random.normal(0, 1, (10000, 100))
   
   # Avoid - multiple calls
   large_array = np.zeros((10000, 100))
   for i in range(10000):
       large_array[i] = np.random.normal(0, 1, 100)
   ```

2. Use appropriate distributions:
   ```python
   # For counts: Poisson
   counts = np.random.poisson(5, 1000)
   
   # For waiting times: Exponential
   wait_times = np.random.exponential(2, 1000)
   
   # For proportions: Beta
   proportions = np.random.beta(2, 5, 1000)
   ```
""")

    print_subsection_header("Statistical Considerations")
    
    print("""
1. Check distribution properties:
   ```python
   sample = np.random.normal(100, 15, 1000)
   print(f"Mean: {np.mean(sample):.2f}")
   print(f"Std: {np.std(sample):.2f}")
   print(f"Skewness: {scipy.stats.skew(sample):.2f}")
   ```

2. Use sufficient sample sizes:
   ```python
   # For reliable estimates, use large samples
   n_samples = 10000  # Large sample for stable estimates
   ```

3. Validate your random generation:
   ```python
   # Test uniform distribution
   uniform_sample = np.random.random(10000)
   print(f"Mean should be ~0.5: {np.mean(uniform_sample):.3f}")
   print(f"Std should be ~0.289: {np.std(uniform_sample):.3f}")
   ```
""")

    # Section 11: Summary and Next Steps
    print_section_header("11. Summary and Next Steps")
    
    print("""
Congratulations! You've completed the NumPy random generation tutorial. Here's what you've learned:

Key Concepts Covered:
✅ Random Number Generation: Basics and reproducibility
✅ Probability Distributions: Uniform, normal, exponential, and more
✅ Random Array Generation: Creating arrays with specific distributions
✅ Statistical Sampling: Simple, stratified, and bootstrap sampling
✅ Monte Carlo Simulations: Estimating complex quantities
✅ Data Science Applications: Augmentation, validation, and inference
✅ Best Practices: Reproducibility and performance optimization

Next Steps:

1. Practice Distribution Generation: Work with different probability distributions
2. Master Sampling Techniques: Implement various sampling strategies
3. Build Monte Carlo Simulations: Create simulations for complex problems
4. Apply to Real Data: Use random generation with actual datasets
5. Explore Advanced Topics: Learn about MCMC and other advanced methods
6. Study Related Libraries: Explore scipy.stats and other statistical libraries

Additional Resources:
- NumPy Random: https://numpy.org/doc/stable/reference/random/
- SciPy Statistics: https://docs.scipy.org/doc/scipy/reference/stats.html
- Random Number Generation: https://numpy.org/doc/stable/reference/random/legacy.html
- Monte Carlo Methods: Various textbooks and online courses

Practice Exercises:
1. Generate samples from different probability distributions
2. Implement various sampling strategies
3. Build Monte Carlo simulations for estimation problems
4. Create data augmentation pipelines
5. Perform bootstrap analysis on real data
6. Validate random number generation quality

Happy Random Generation! 🎲
""")

if __name__ == "__main__":
    # Run the tutorial
    main()
    
    print("\n" + "="*60)
    print(" Tutorial completed successfully!")
    print(" Master NumPy random generation!")
    print("="*60) 