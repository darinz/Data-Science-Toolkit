# Artificial Intelligence, Machine Learning, and Data Science (AIMLDS) Toolkit

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green.svg)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-blue.svg)](https://matplotlib.org/)
[![Conda](https://img.shields.io/badge/Conda-Environment-green.svg)](https://docs.conda.io/)

A comprehensive collection of toolkits for essential AI/ML and Data Science libraries in Python. This toolkit is designed to help you quickly get started with scientific computing, data analysis, machine learning, deep learning, and data visualization.

## What's Included

### Core Data Science Libraries
- **NumPy Tutorials** - Scientific computing fundamentals and array operations (Python scripts)
- **pandas Tutorials** - Data manipulation, analysis, and DataFrame operations
- **Matplotlib Tutorials** - Data visualization and plotting fundamentals

### Machine Learning & Deep Learning
- **scikit-learn Tutorials** - Machine learning algorithms and workflows
- **TensorFlow Tutorials** - Deep learning and neural networks
- **PyTorch Tutorials** - Alternative deep learning framework

### Development Environment
- **Jupyter Tutorials** - Interactive computing and notebook workflows (Python scripts)

## Quick Start

### Prerequisites
- Python 3.10 or higher
- Basic understanding of Python programming
- Conda or pip package manager

### Installation Options

#### Option 1: Google Colab (Recommended for Beginners)
- Open the tutorial notebooks directly in [Google Colab](https://colab.research.google.com/)
- No local setup required
- Free GPU access available

#### Option 2: Local Environment Setup

**Using Conda (Recommended):**
```bash
# Clone the repository
git clone https://github.com/darinz/AIMLDS-Toolkit.git
cd AIMLDS-Toolkit

# Create environment for specific module
cd NumPy
conda env create -f environment.yml
conda activate numpy-tutorials

# Repeat for other modules as needed
cd ../pandas
conda env create -f environment.yml
conda activate pandas-tutorials
```

**Using pip:**
```bash
# Install dependencies for specific module
cd NumPy
pip install -r requirements.txt

# Repeat for other modules as needed
cd ../pandas
pip install -r requirements.txt
```

## Tutorial Structure

### NumPy Module
- **Files:** `numpy_basics.py`, `advanced_indexing.py`, `linear_algebra.py`, `random_generation.py`, `array_manipulation.py`
- **Dependencies:** NumPy, pandas, JupyterLab
- **Focus:** Scientific computing fundamentals
- **Note:** All NumPy tutorials are available as Python scripts for easy execution

### pandas Module  
- **File:** `pandas_df_ultraquick_tutorial.ipynb`
- **Dependencies:** NumPy, pandas, JupyterLab
- **Focus:** Data analysis and manipulation

### TensorFlow Module
- **Files:** `tf_basics.ipynb`, `neural_networks.ipynb`, `cnn_tutorial.ipynb`, `rnn_tutorial.ipynb`, `data_pipelines.ipynb`
- **Dependencies:** TensorFlow, NumPy, pandas, Matplotlib, JupyterLab
- **Focus:** Deep learning and neural networks

### scikit-learn Module
- **Files:** `ml_basics.ipynb`, `supervised_learning.ipynb`, `unsupervised_learning.ipynb`, `feature_engineering.ipynb`, `model_selection.ipynb`, `real_world_applications.ipynb`
- **Dependencies:** scikit-learn, NumPy, pandas, Matplotlib, JupyterLab
- **Focus:** Machine learning algorithms and workflows

### Jupyter Module
- **Files:** `jupyter_basics.py`, `jupyterlab_interface.py`, `magic_commands.py`, `interactive_widgets.py`, `best_practices.py`, `advanced_features.py`, `deployment.py`
- **Dependencies:** Jupyter, JupyterLab, NumPy, pandas, Matplotlib
- **Focus:** Interactive computing and notebook workflows
- **Note:** All Jupyter tutorials are available as Python scripts for easy execution

### Matplotlib Module
- **Files:** `matplotlib_basics.py`, `plot_types.ipynb`, `customization.ipynb`, `subplots_layout.ipynb`, `statistical_plots.ipynb`, `3d_plotting.ipynb`, `advanced_features.ipynb`, `publication_quality.ipynb`
- **Dependencies:** Matplotlib, NumPy, pandas, JupyterLab
- **Focus:** Data visualization and plotting
- **Note:** Basics tutorial is available as both Python script and Jupyter notebook

### PyTorch Module
- **Files:** Various tutorials in subdirectories (Tensor, Neural-Networks, Autograd, Image-Classifier)
- **Dependencies:** PyTorch, NumPy, pandas, Matplotlib, JupyterLab
- **Focus:** Deep learning with PyTorch

## Learning Path

### Beginner Path
1. **Start with NumPy** - Build your foundation in scientific computing
2. **Move to pandas** - Learn data manipulation and analysis
3. **Explore Matplotlib** - Master data visualization
4. **Learn Jupyter** - Understand interactive computing workflows

### Machine Learning Path
1. **Begin with scikit-learn** - Learn machine learning fundamentals
2. **Practice with real datasets** - Apply ML algorithms to practical problems
3. **Master feature engineering** - Prepare data for modeling
4. **Optimize model performance** - Tune hyperparameters and evaluate models

### Deep Learning Path
1. **Choose your framework** - Start with either TensorFlow or PyTorch
2. **Learn neural network basics** - Understand layers, activation functions, and training
3. **Explore specialized architectures** - CNNs for computer vision, RNNs for sequential data
4. **Build end-to-end projects** - Combine all skills in practical applications

## Environment Management

### Conda Commands
```bash
# Create new environment
conda env create -f environment.yml

# Activate environment
conda activate [module-name]-tutorials

# Update environment
conda env update -f environment.yml --prune

# Remove environment
conda remove --name [module-name]-tutorials --all

# List environments
conda env list
```

### Jupyter Commands
```bash
# Start Jupyter Lab
jupyter lab

# Start Jupyter Notebook
jupyter notebook

# Start with specific port
jupyter lab --port=8888

# Start with no browser
jupyter lab --no-browser
```

### Running NumPy Tutorials
```bash
# Navigate to NumPy directory
cd NumPy

# Run individual tutorials
python numpy_basics.py
python advanced_indexing.py
python linear_algebra.py
python random_generation.py
python array_manipulation.py

# Or run all tutorials in sequence
for script in *.py; do
    echo "Running $script..."
    python "$script"
    echo "Completed $script"
    echo "----------------------------------------"
done
```

### Running Jupyter Tutorials
```bash
# Navigate to Jupyter directory
cd Jupyter

# Run individual tutorials
python jupyter_basics.py
python jupyterlab_interface.py
python magic_commands.py
python interactive_widgets.py
python best_practices.py
python advanced_features.py
python deployment.py

# Or run all tutorials in sequence
for script in *.py; do
    echo "Running $script..."
    python "$script"
    echo "Completed $script"
    echo "----------------------------------------"
done
```

## Key Features

- **Comprehensive Coverage** - All major data science and ML libraries
- **Interactive Learning** - Hands-on tutorials with practical examples
- **Real-world Applications** - Industry-standard workflows and best practices
- **Modular Design** - Learn each library independently or as part of a complete curriculum
- **Production Ready** - Skills that translate directly to professional work
- **Flexible Formats** - Both Jupyter notebooks and Python scripts available

## Datasets and Examples

Each module includes:
- **Built-in datasets** - scikit-learn, TensorFlow, and other library datasets
- **Synthetic data** - Generated examples for learning concepts
- **Real-world examples** - Practical applications and use cases
- **Custom datasets** - Examples you can create and modify

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Contribution Guidelines
1. Follow the existing code style and structure
2. Add comprehensive documentation and comments
3. Include practical examples and use cases
4. Test notebooks in different environments
5. Update environment files when adding new dependencies

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [NumPy](https://numpy.org/) - The fundamental package for scientific computing
- [pandas](https://pandas.pydata.org/) - Data analysis and manipulation library
- [TensorFlow](https://www.tensorflow.org/) - Deep learning framework
- [scikit-learn](https://scikit-learn.org/) - Machine learning library
- [Matplotlib](https://matplotlib.org/) - Plotting library
- [Jupyter](https://jupyter.org/) - Interactive computing platform
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Google Colab](https://colab.research.google.com/) - Free cloud-based Jupyter environment

## Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the individual module README files for specific guidance
- Refer to the official documentation for each library
- Join our community discussions

## Roadmap

### Planned Additions
- **Seaborn Tutorials** - Advanced statistical visualization
- **Plotly Tutorials** - Interactive plotting and dashboards
- **Streamlit Tutorials** - Web application development
- **FastAPI Tutorials** - API development for ML models
- **MLOps Tutorials** - Model deployment and monitoring

### Community Requests
- Additional deep learning architectures
- More real-world datasets and examples
- Advanced optimization techniques
- Cloud deployment tutorials
- Industry-specific applications

---

**Happy Learning! 🚀**

*This toolkit is designed to grow with the community. Your feedback and contributions help make it better for everyone.*
