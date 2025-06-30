# Artificial Intelligence, Machine Learning, and Data Science Toolkit

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue.svg)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/Pandas-2.0+-blue.svg)](https://pandas.pydata.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green.svg)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-blue.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.12+-orange.svg)](https://seaborn.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.0+-purple.svg)](https://plotly.com/python/)
[![Conda](https://img.shields.io/badge/Conda-Environment-green.svg)](https://docs.conda.io/)

A comprehensive collection of toolkits for essential AI/ML and Data Science libraries in Python. This toolkit is designed to help you quickly get started with scientific computing, data analysis, machine learning, deep learning, and data visualization.

## What's Included

### Core Data Science Libraries
- **[Python Tutorials](./Python/)** - Python fundamentals, data science, and AI/ML applications (Markdown guides)
- **[NumPy Tutorials](./NumPy/)** - Scientific computing fundamentals and array operations (Markdown guides + Jupyter notebooks)
- **[pandas Tutorials](./pandas/)** - Data manipulation, analysis, and DataFrame operations (Markdown guides + Jupyter notebooks)
- **[Matplotlib Tutorials](./Matplotlib/)** - Data visualization and plotting fundamentals (Markdown guides)
- **[Seaborn Tutorials](./Seaborn/)** - Statistical data visualization (Markdown guides)
- **[Plotly Tutorials](./Plotly/)** - Interactive data visualization and dashboards (Markdown guides)

### Machine Learning & Deep Learning
- **[scikit-learn Tutorials](./scikit-learn/)** - Machine learning algorithms and workflows (Markdown guides)
- **[TensorFlow Tutorials](./TensorFlow/)** - Deep learning and neural networks (Markdown guides)
- **[PyTorch Tutorials](./PyTorch/)** - Alternative deep learning framework (Markdown guides + Jupyter notebooks in subdirectories)

### Development Environment
- **[Jupyter Tutorials](./Jupyter/)** - Interactive computing and notebook workflows (Markdown guides)

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

### Python Module
- **Files:** `python_basics_guide.md`, `data_manipulation_guide.md`, `neural_networks_guide.md`, `nlp_guide.md`, `supervised_learning_guide.md`, `computer_vision_guide.md`
- **Dependencies:** Python, NumPy, pandas, scikit-learn, TensorFlow, PyTorch, JupyterLab
- **Focus:** Python fundamentals, data science, and AI/ML applications

### NumPy Module
- **Files:** `numpy_basics_guide.md`, `advanced_indexing_guide.md`, `linear_algebra_guide.md`, `random_generation_guide.md`, `array_manipulation_guide.md`, `numpy_ultraquick_tutorial.ipynb`, `random_generation.py`
- **Dependencies:** NumPy, pandas, JupyterLab
- **Focus:** Scientific computing fundamentals
- **Note:** Comprehensive markdown guides with one Python script and one Jupyter notebook

### pandas Module  
- **Files:** `pandas_basics_guide.md`, `data_analysis_guide.md`, `data_visualization_guide.md`, `time_series_guide.md`, `pandas_df_ultraquick_tutorial.ipynb`
- **Dependencies:** NumPy, pandas, JupyterLab
- **Focus:** Data analysis and manipulation

### Matplotlib Module
- **Files:** `matplotlib_basics_guide.md`, `plot_types_guide.md`, `customization_guide.md`, `subplots_layout_guide.md`, `statistical_plots_guide.md`, `3d_plotting_guide.md`, `advanced_features_guide.md`, `publication_quality_guide.md`
- **Dependencies:** Matplotlib, NumPy, pandas, JupyterLab
- **Focus:** Data visualization and plotting

### Seaborn Module
- **Files:** `seaborn_basics_guide.md`, `statistical_plots_guide.md`, `categorical_plots_guide.md`, `distribution_analysis_guide.md`, `correlation_analysis_guide.md`, `multi_plot_grids_guide.md`, `advanced_features_guide.md`
- **Dependencies:** Seaborn, Matplotlib, NumPy, pandas, JupyterLab
- **Focus:** Statistical data visualization

### Plotly Module
- **Files:** `plotly_basics_guide.md`, `plot_types_guide.md`
- **Dependencies:** Plotly, NumPy, pandas, JupyterLab
- **Focus:** Interactive data visualization and dashboards

### TensorFlow Module
- **Files:** `tensorflow_basics_guide.md`, `neural_networks_guide.md`, `cnn_guide.md`, `rnn_guide.md`, `data_pipelines_guide.md`, `advanced_guide.md`
- **Dependencies:** TensorFlow, NumPy, pandas, Matplotlib, JupyterLab
- **Focus:** Deep learning and neural networks

### scikit-learn Module
- **Files:** `ml_basics_guide.md`, `supervised_learning_guide.md`, `unsupervised_learning_guide.md`, `feature_engineering_guide.md`, `model_selection_guide.md`, `real_world_applications_guide.md`
- **Dependencies:** scikit-learn, NumPy, pandas, Matplotlib, JupyterLab
- **Focus:** Machine learning algorithms and workflows

### Jupyter Module
- **Files:** `jupyter_basics_guide.md`, `jupyterlab_interface_guide.md`, `magic_commands_guide.md`, `interactive_widgets_guide.md`, `best_practices_guide.md`, `advanced_features_guide.md`, `deployment_guide.md`
- **Dependencies:** Jupyter, JupyterLab, NumPy, pandas, Matplotlib
- **Focus:** Interactive computing and notebook workflows

### PyTorch Module
- **Main Files:** `advanced_pytorch_techniques_guide.md`
- **Tensor Subdirectory:** `comprehensive_tensor_guide.md`, `tensor_operations_guide.md`, `pt_tensor.ipynb`
- **Neural-Networks Subdirectory:** `neural_networks_comprehensive_guide.md`, `pt_neural_networks.ipynb`
- **Autograd Subdirectory:** `autograd_comprehensive_guide.md`, `pt_autograd.ipynb`
- **Image-Classifier Subdirectory:** `image_classification_comprehensive_guide.md`, `image_classifier.ipynb`
- **Dependencies:** PyTorch, NumPy, pandas, Matplotlib, JupyterLab
- **Focus:** Deep learning with PyTorch

## Learning Path

### Beginner Path
1. **Start with Python** - Build your foundation in Python programming for data science
2. **Move to NumPy** - Learn scientific computing fundamentals
3. **Advance to pandas** - Master data manipulation and analysis
4. **Explore Matplotlib** - Learn data visualization
5. **Learn Jupyter** - Understand interactive computing workflows

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

### Visualization Path
1. **Start with Matplotlib** - Learn fundamental plotting concepts
2. **Explore Seaborn** - Master statistical visualization
3. **Advance to Plotly** - Create interactive visualizations and dashboards
4. **Combine techniques** - Use multiple libraries for comprehensive visualization

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

### Running Python Script Tutorials
```bash
# Navigate to NumPy directory
cd NumPy

# Run the available Python script
python random_generation.py
```

### Running Jupyter Tutorials
```bash
# Navigate to module directory
cd NumPy

# Start Jupyter Lab
jupyter lab

# Open the notebook file
# numpy_ultraquick_tutorial.ipynb
```

## Module-Specific Guides

Each module contains detailed README files with:
- **Installation instructions** for that specific module
- **Tutorial descriptions** and learning objectives
- **File structure** and dependencies
- **Running instructions** for tutorials
- **Best practices** and tips
- **Additional resources** and references

## Contributing

We welcome contributions! Please feel free to:
- Report bugs or issues
- Suggest new features or improvements
- Submit pull requests with enhancements
- Improve documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- All the open-source library maintainers and contributors
- The Python data science community
- Educators and researchers who inspire these tutorials

---

**Happy Learning!**
