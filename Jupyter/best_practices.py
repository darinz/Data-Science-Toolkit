#!/usr/bin/env python3
"""
Jupyter Best Practices: Writing Professional Notebooks

Welcome to the Jupyter Best Practices tutorial! This guide covers essential 
practices for creating professional, maintainable, and effective Jupyter notebooks 
that can be shared, reproduced, and used in production environments.

This script covers:
- Notebook organization and structure
- Code quality and style
- Documentation and markdown
- Version control with Git
- Performance optimization
- Collaboration and sharing

Prerequisites:
- Python 3.8 or higher
- Basic understanding of Jupyter (covered in jupyter_basics.py)
- Familiarity with Git and version control
"""

import os
import sys
import json
import time
from datetime import datetime

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
    
    print("Jupyter Best Practices Tutorial")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print("Best practices tutorial started successfully!")

    # Section 1: Introduction to Best Practices
    print_section_header("1. Introduction to Best Practices")
    
    print("""
Jupyter notebooks are powerful tools for data science, but without proper 
practices, they can become difficult to maintain, share, and reproduce. 
This tutorial covers essential best practices for creating professional notebooks.

Key Principles:
- Reproducibility: Notebooks should produce consistent results
- Readability: Code and documentation should be clear and well-organized
- Maintainability: Notebooks should be easy to update and modify
- Collaboration: Notebooks should be easy to share and work on with others
- Performance: Notebooks should be efficient and well-optimized

Benefits of Following Best Practices:
✅ Consistent and reliable results
✅ Easy to understand and maintain
✅ Smooth collaboration with team members
✅ Professional presentation and sharing
✅ Better performance and efficiency
✅ Easier debugging and troubleshooting
""")

    # Section 2: Notebook Organization and Structure
    print_section_header("2. Notebook Organization and Structure")
    
    print("""
A well-organized notebook follows a logical structure that guides readers 
through your analysis or project. Here's a recommended structure:

1. Header Section:
   - Title and description
   - Author and date
   - Version information
   - Dependencies and requirements

2. Setup Section:
   - Import statements
   - Configuration settings
   - Data loading and preprocessing
   - Environment setup

3. Analysis Section:
   - Exploratory data analysis
   - Data visualization
   - Statistical analysis
   - Model development

4. Results Section:
   - Key findings and insights
   - Model performance metrics
   - Visualizations and charts
   - Conclusions and recommendations

5. Appendix Section:
   - Additional analysis
   - Code documentation
   - References and citations
   - Future work and improvements
""")

    # Demonstrate notebook structure
    print_subsection_header("Notebook Structure Example")
    
    print("""
```markdown
# Data Analysis Project: Customer Segmentation

**Author:** Data Scientist  
**Date:** 2024-01-15  
**Version:** 1.0  
**Description:** Analysis of customer data to identify segments

## 1. Setup and Imports

```python
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configure plotting
plt.style.use('seaborn')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)
```

## 2. Data Loading and Preprocessing

```python
# Load data
df = pd.read_csv('customer_data.csv')

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
df.head()
```

## 3. Exploratory Data Analysis

```python
# Basic statistics
df.describe()

# Check for missing values
df.isnull().sum()

# Visualize distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# ... plotting code ...
```

## 4. Model Development

```python
# Prepare data for clustering
features = ['age', 'income', 'spending_score']
X = df[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform clustering
kmeans = KMeans(n_clusters=4, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)
```

## 5. Results and Conclusions

```python
# Analyze clusters
cluster_summary = df.groupby('cluster')[features].mean()
print("Cluster Centers:")
print(cluster_summary)

# Visualize results
plt.figure(figsize=(10, 6))
# ... visualization code ...
```

## 6. Appendix

### Additional Analysis
- Cluster validation metrics
- Feature importance analysis

### References
- K-means clustering algorithm
- Customer segmentation best practices
```
""")

    # Section 3: Code Quality and Style
    print_section_header("3. Code Quality and Style")
    
    print("""
Writing clean, readable code is essential for maintainable notebooks. 
Follow these guidelines for better code quality:

1. Code Style:
   - Follow PEP 8 style guidelines
   - Use meaningful variable names
   - Keep functions and classes small
   - Add appropriate comments
   - Use consistent formatting

2. Code Organization:
   - Separate concerns into different cells
   - Group related operations together
   - Use functions for reusable code
   - Avoid long, complex cells
   - Keep cells focused on single tasks

3. Error Handling:
   - Use try-except blocks appropriately
   - Provide meaningful error messages
   - Handle edge cases gracefully
   - Validate inputs and outputs
   - Log important events and errors

4. Performance:
   - Use efficient data structures
   - Avoid unnecessary computations
   - Profile code when needed
   - Use appropriate libraries
   - Optimize memory usage
""")

    # Demonstrate code quality examples
    print_subsection_header("Code Quality Examples")
    
    print("""
Good Code Example:
```python
# Load and preprocess customer data
def load_customer_data(file_path):
    \"\"\"Load customer data from CSV file and perform basic preprocessing.\"\"\"
    try:
        df = pd.read_csv(file_path)
        
        # Validate required columns
        required_columns = ['customer_id', 'age', 'income', 'spending_score']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean data
        df = df.dropna(subset=required_columns)
        df = df[df['age'] > 0]  # Remove invalid ages
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Use the function
customer_data = load_customer_data('customers.csv')
if customer_data is not None:
    print(f"Loaded {len(customer_data)} customer records")
```

Poor Code Example:
```python
# Bad: No error handling, unclear variable names
df = pd.read_csv('data.csv')
df = df.dropna()
df = df[df['a'] > 0]
print(len(df))
```
""")

    # Section 4: Documentation and Markdown
    print_section_header("4. Documentation and Markdown")
    
    print("""
Good documentation is crucial for making notebooks understandable and 
reproducible. Use markdown cells effectively to explain your work.

1. Markdown Best Practices:
   - Use clear, descriptive headers
   - Write concise but informative descriptions
   - Include mathematical equations when needed
   - Use code blocks for examples
   - Add links to references and resources

2. Documentation Content:
   - Explain the purpose and goals
   - Describe data sources and preprocessing
   - Document assumptions and limitations
   - Explain methodology and algorithms
   - Interpret results and conclusions

3. Code Comments:
   - Explain complex logic
   - Document function parameters and returns
   - Add inline comments for clarity
   - Use docstrings for functions
   - Keep comments up to date
""")

    # Demonstrate documentation examples
    print_subsection_header("Documentation Examples")
    
    print("""
Good Documentation Example:
```markdown
# Customer Segmentation Analysis

## Overview
This notebook analyzes customer data to identify distinct customer segments 
using K-means clustering. The analysis helps understand customer behavior 
patterns and supports targeted marketing strategies.

## Data Description
- **Source:** Customer transaction database
- **Period:** January 2023 - December 2023
- **Variables:** Age, Income, Spending Score, Purchase Frequency
- **Sample Size:** 10,000 customers

## Methodology
1. **Data Preprocessing:** Clean missing values, remove outliers
2. **Feature Engineering:** Create derived variables
3. **Clustering:** Apply K-means algorithm with optimal k
4. **Validation:** Assess cluster quality using silhouette score

## Key Findings
- Identified 4 distinct customer segments
- High-income customers show highest spending variability
- Age is not strongly correlated with spending patterns
- Recommendations for segment-specific marketing strategies

## Limitations
- Analysis based on historical data only
- Assumes customer behavior remains stable
- Limited to available variables in dataset
```

Code Documentation Example:
```python
def perform_customer_segmentation(data, n_clusters=4, random_state=42):
    \"\"\"
    Perform customer segmentation using K-means clustering.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Customer data with features for clustering
    n_clusters : int, default=4
        Number of clusters to create
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    tuple
        (cluster_labels, cluster_centers, silhouette_score)
    
    Notes:
    ------
    Features are automatically scaled using StandardScaler.
    Optimal number of clusters can be determined using elbow method.
    \"\"\"
    # Scale features for clustering
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(data_scaled)
    
    # Calculate silhouette score
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    
    return cluster_labels, kmeans.cluster_centers_, silhouette_avg
```
""")

    # Section 5: Version Control with Git
    print_section_header("5. Version Control with Git")
    
    print("""
Version control is essential for tracking changes, collaborating with others, 
and maintaining a history of your work. Git is the most popular version 
control system for Jupyter notebooks.

1. Git Setup for Jupyter:
   - Initialize Git repository
   - Configure .gitignore for Jupyter
   - Set up remote repository
   - Configure user information

2. Best Practices:
   - Commit frequently with meaningful messages
   - Use branches for different features
   - Review changes before committing
   - Keep commits focused and atomic
   - Document major changes

3. .gitignore Configuration:
   - Exclude output files and caches
   - Ignore sensitive data
   - Exclude temporary files
   - Include only necessary files
""")

    # Demonstrate Git setup
    print_subsection_header("Git Setup Example")
    
    print("""
.gitignore for Jupyter Projects:
```gitignore
# Jupyter Notebook
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# IPython
profile_default/
ipython_config.py

# JupyterLab
.jupyter/lab/

# Data files (adjust as needed)
*.csv
*.xlsx
*.json
data/
datasets/

# Output files
outputs/
results/
figures/
*.png
*.jpg
*.pdf

# Environment files
.env
.venv
env/
venv/
ENV/

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
```

Git Workflow Example:
```bash
# Initialize repository
git init
git add .gitignore
git commit -m "Initial commit: Add .gitignore"

# Add notebook files
git add *.ipynb
git commit -m "Add customer segmentation notebook"

# Create feature branch
git checkout -b feature/data-preprocessing
# ... make changes ...
git add .
git commit -m "Add data preprocessing functions"

# Merge back to main
git checkout main
git merge feature/data-preprocessing
```
""")

    # Section 6: Performance Optimization
    print_section_header("6. Performance Optimization")
    
    print("""
Optimizing notebook performance is important for working with large datasets 
and complex computations. Here are key strategies:

1. Data Loading and Storage:
   - Use appropriate file formats (Parquet, HDF5)
   - Load only necessary columns
   - Use data types efficiently
   - Consider chunked processing
   - Cache intermediate results

2. Computation Optimization:
   - Vectorize operations when possible
   - Use efficient libraries (NumPy, pandas)
   - Avoid loops in favor of vectorized operations
   - Profile code to identify bottlenecks
   - Use parallel processing when appropriate

3. Memory Management:
   - Monitor memory usage
   - Delete unnecessary variables
   - Use generators for large datasets
   - Restart kernel when needed
   - Use memory-efficient data structures
""")

    # Demonstrate performance optimization
    print_subsection_header("Performance Optimization Examples")
    
    print("""
Efficient Data Loading:
```python
# Good: Load only necessary columns
columns_needed = ['customer_id', 'age', 'income', 'spending_score']
df = pd.read_csv('large_dataset.csv', usecols=columns_needed)

# Good: Use appropriate data types
df['customer_id'] = df['customer_id'].astype('category')
df['age'] = df['age'].astype('int8')
df['income'] = df['income'].astype('float32')

# Good: Use efficient file formats
df.to_parquet('data.parquet')  # Faster than CSV
df = pd.read_parquet('data.parquet')
```

Vectorized Operations:
```python
# Good: Vectorized operation
df['income_category'] = pd.cut(df['income'], bins=5, labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'])

# Bad: Loop-based operation
income_categories = []
for income in df['income']:
    if income < 30000:
        income_categories.append('Low')
    elif income < 50000:
        income_categories.append('Medium')
    else:
        income_categories.append('High')
df['income_category'] = income_categories

# Good: Use NumPy for numerical operations
import numpy as np
result = np.sqrt(np.sum(df[['age', 'income']]**2, axis=1))
```

Memory Management:
```python
# Monitor memory usage
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"Memory usage: {get_memory_usage():.2f} MB")

# Clean up memory
import gc
del large_dataframe
gc.collect()

# Use generators for large datasets
def process_data_in_chunks(file_path, chunk_size=10000):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        yield process_chunk(chunk)
```
""")

    # Section 7: Collaboration and Sharing
    print_section_header("7. Collaboration and Sharing")
    
    print("""
Effective collaboration requires clear communication, consistent practices, 
and appropriate tools for sharing and reviewing notebooks.

1. Collaboration Best Practices:
   - Use consistent naming conventions
   - Document assumptions and decisions
   - Provide clear instructions for setup
   - Use version control effectively
   - Review and test each other's work

2. Sharing Options:
   - GitHub/GitLab for version control
   - JupyterHub for shared environments
   - Google Colab for cloud-based collaboration
   - nbviewer for static notebook viewing
   - Binder for interactive environments

3. Review Process:
   - Code review guidelines
   - Documentation review
   - Results validation
   - Performance assessment
   - Security considerations
""")

    # Demonstrate collaboration setup
    print_subsection_header("Collaboration Setup Example")
    
    print("""
README.md for Collaborative Projects:
```markdown
# Customer Segmentation Project

## Overview
This project analyzes customer data to identify segments for targeted marketing.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/team/customer-segmentation.git
   cd customer-segmentation
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start Jupyter:
   ```bash
   jupyter lab
   ```

## Project Structure
```
customer-segmentation/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_customer_segmentation.ipynb
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   └── utils.py
├── requirements.txt
└── README.md
```

## Contributing Guidelines
1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and test thoroughly
3. Commit with descriptive messages
4. Create pull request with detailed description
5. Request review from team members

## Code Review Checklist
- [ ] Code follows style guidelines
- [ ] Documentation is clear and complete
- [ ] Results are reproducible
- [ ] Performance is acceptable
- [ ] No sensitive data is included
```
""")

    # Section 8: Testing and Validation
    print_section_header("8. Testing and Validation")
    
    print("""
Testing and validation ensure that your notebooks produce reliable and 
correct results. Implement testing strategies appropriate for your project.

1. Testing Strategies:
   - Unit tests for functions
   - Integration tests for workflows
   - Data validation tests
   - Performance benchmarks
   - Reproducibility tests

2. Validation Techniques:
   - Cross-validation for models
   - Holdout validation sets
   - Statistical significance tests
   - Sensitivity analysis
   - Robustness checks

3. Quality Assurance:
   - Code review processes
   - Documentation review
   - Results validation
   - Performance monitoring
   - Security audits
""")

    # Demonstrate testing examples
    print_subsection_header("Testing Examples")
    
    print("""
Unit Testing Example:
```python
import unittest
import pandas as pd
import numpy as np

class TestCustomerSegmentation(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        self.test_data = pd.DataFrame({
            'age': [25, 35, 45, 55],
            'income': [30000, 50000, 70000, 90000],
            'spending_score': [50, 60, 70, 80]
        })
    
    def test_data_loading(self):
        """Test data loading function."""
        df = load_customer_data('test_data.csv')
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 4)
        self.assertListEqual(list(df.columns), ['age', 'income', 'spending_score'])
    
    def test_clustering(self):
        """Test clustering function."""
        labels, centers, score = perform_customer_segmentation(self.test_data, n_clusters=2)
        self.assertEqual(len(labels), len(self.test_data))
        self.assertEqual(centers.shape, (2, 3))
        self.assertGreater(score, 0)
    
    def test_data_validation(self):
        """Test data validation."""
        # Test with invalid data
        invalid_data = self.test_data.copy()
        invalid_data.loc[0, 'age'] = -5  # Invalid age
        
        with self.assertRaises(ValueError):
            validate_customer_data(invalid_data)

# Run tests
if __name__ == '__main__':
    unittest.main()
```

Data Validation Example:
```python
def validate_customer_data(df):
    """Validate customer data for clustering."""
    errors = []
    
    # Check required columns
    required_columns = ['age', 'income', 'spending_score']
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")
    
    # Check data types
    if not pd.api.types.is_numeric_dtype(df['age']):
        errors.append("Age must be numeric")
    
    # Check value ranges
    if (df['age'] < 0).any():
        errors.append("Age cannot be negative")
    
    if (df['income'] < 0).any():
        errors.append("Income cannot be negative")
    
    # Check for missing values
    if df[required_columns].isnull().any().any():
        errors.append("Missing values found in required columns")
    
    if errors:
        raise ValueError("Data validation failed: " + "; ".join(errors))
    
    return True
```
""")

    # Section 9: Security and Privacy
    print_section_header("9. Security and Privacy")
    
    print("""
Security and privacy are crucial considerations when working with sensitive 
data in Jupyter notebooks. Implement appropriate safeguards.

1. Data Security:
   - Never commit sensitive data to version control
   - Use environment variables for credentials
   - Encrypt sensitive data when possible
   - Implement access controls
   - Regular security audits

2. Privacy Protection:
   - Anonymize personal data
   - Use data masking techniques
   - Implement data retention policies
   - Follow privacy regulations (GDPR, etc.)
   - Document data handling procedures

3. Best Practices:
   - Use .env files for configuration
   - Implement proper authentication
   - Regular security updates
   - Monitor for security issues
   - Train team on security practices
""")

    # Demonstrate security practices
    print_subsection_header("Security Examples")
    
    print("""
Environment Configuration:
```python
# .env file (not committed to Git)
DATABASE_URL=postgresql://user:password@localhost/db
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here

# Load environment variables
import os
from dotenv import load_dotenv

load_dotenv()

# Use environment variables
database_url = os.getenv('DATABASE_URL')
api_key = os.getenv('API_KEY')
```

Data Anonymization:
```python
import hashlib

def anonymize_customer_data(df):
    """Anonymize customer data for analysis."""
    df_anon = df.copy()
    
    # Hash customer IDs
    df_anon['customer_id'] = df_anon['customer_id'].apply(
        lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:8]
    )
    
    # Remove personally identifiable information
    if 'email' in df_anon.columns:
        df_anon = df_anon.drop('email', axis=1)
    
    if 'phone' in df_anon.columns:
        df_anon = df_anon.drop('phone', axis=1)
    
    # Age binning for privacy
    df_anon['age_group'] = pd.cut(df_anon['age'], 
                                 bins=[0, 25, 35, 45, 55, 100], 
                                 labels=['18-25', '26-35', '36-45', '46-55', '55+'])
    df_anon = df_anon.drop('age', axis=1)
    
    return df_anon
```

Secure Data Loading:
```python
def load_secure_data(file_path, encryption_key=None):
    """Load data with security measures."""
    try:
        if file_path.endswith('.encrypted'):
            # Decrypt file first
            if encryption_key is None:
                raise ValueError("Encryption key required for encrypted files")
            # Decryption logic here
            pass
        
        # Validate file path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Log access (for audit trail)
        log_data_access(file_path, datetime.now())
        
        return df
    
    except Exception as e:
        log_security_event(f"Failed to load {file_path}: {e}")
        raise
```
""")

    # Section 10: Deployment and Production
    print_section_header("10. Deployment and Production")
    
    print("""
Deploying Jupyter notebooks to production requires careful planning and 
appropriate tools. Consider the specific requirements of your use case.

1. Deployment Options:
   - JupyterHub for multi-user environments
   - Papermill for automated execution
   - Voilà for dashboard deployment
   - Binder for interactive demos
   - Cloud platforms (AWS, GCP, Azure)

2. Production Considerations:
   - Security and authentication
   - Resource management
   - Monitoring and logging
   - Backup and recovery
   - Performance optimization

3. Automation:
   - Scheduled notebook execution
   - Automated testing
   - Continuous integration
   - Automated deployment
   - Monitoring and alerting
""")

    # Demonstrate deployment examples
    print_subsection_header("Deployment Examples")
    
    print("""
Papermill for Automated Execution:
```python
# notebook_template.ipynb with parameters cell
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {"tags": ["parameters"]},
   "outputs": [],
   "source": [
    "input_file = 'data.csv'\n",
    "output_file = 'results.csv'\n",
    "n_clusters = 4"
   ]
  }
 ]
}

# Execute with papermill
import papermill as pm

pm.execute_notebook(
    'notebook_template.ipynb',
    'output_notebook.ipynb',
    parameters={
        'input_file': 'customer_data.csv',
        'output_file': 'segmentation_results.csv',
        'n_clusters': 5
    }
)
```

Voilà Dashboard:
```python
# dashboard.ipynb
import ipywidgets as widgets
import plotly.express as px
import pandas as pd

# Load data
df = pd.read_csv('customer_data.csv')

# Create interactive widgets
cluster_slider = widgets.IntSlider(value=4, min=2, max=8, description='Clusters:')
update_button = widgets.Button(description='Update Analysis')

# Create output widget
output = widgets.Output()

def update_analysis(b):
    with output:
        output.clear_output()
        
        # Perform clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=cluster_slider.value)
        df['cluster'] = kmeans.fit_predict(df[['age', 'income']])
        
        # Create plot
        fig = px.scatter(df, x='age', y='income', color='cluster', 
                        title=f'Customer Segments ({cluster_slider.value} clusters)')
        fig.show()

update_button.on_click(update_analysis)

# Display dashboard
display(widgets.VBox([cluster_slider, update_button, output]))
```

# Deploy with Voilà
# voila dashboard.ipynb --port=8866
```
""")

    # Section 11: Monitoring and Maintenance
    print_section_header("11. Monitoring and Maintenance")
    
    print("""
Regular monitoring and maintenance ensure that your notebooks continue to 
work correctly and efficiently over time.

1. Monitoring:
   - Execution time and performance
   - Error rates and failures
   - Resource usage
   - Data quality metrics
   - User feedback and issues

2. Maintenance Tasks:
   - Update dependencies regularly
   - Review and update documentation
   - Optimize performance bottlenecks
   - Fix bugs and issues
   - Archive old notebooks

3. Quality Assurance:
   - Regular code reviews
   - Performance testing
   - Security audits
   - Documentation updates
   - User training and support
""")

    # Section 12: Summary and Next Steps
    print_section_header("12. Summary and Next Steps")
    
    print("""
Congratulations! You've completed the Jupyter Best Practices tutorial. Here's what you've learned:

Key Concepts Covered:
✅ Notebook Organization: Structured and logical notebook layout
✅ Code Quality: Clean, readable, and maintainable code
✅ Documentation: Clear and comprehensive documentation
✅ Version Control: Git integration and collaboration
✅ Performance Optimization: Efficient data processing and computation
✅ Collaboration: Team workflows and sharing practices
✅ Testing and Validation: Ensuring reliable results
✅ Security and Privacy: Protecting sensitive data
✅ Deployment: Production-ready notebook deployment
✅ Monitoring and Maintenance: Ongoing quality assurance

Next Steps:

1. Apply Best Practices: Start using these practices in your notebooks
2. Set Up Version Control: Implement Git workflow for your projects
3. Create Templates: Develop notebook templates for common tasks
4. Establish Guidelines: Create team-specific best practices
5. Automate Processes: Implement automated testing and deployment
6. Monitor and Improve: Continuously monitor and improve your workflows

Additional Resources:
- Jupyter Documentation: https://jupyter.org/
- Git Documentation: https://git-scm.com/doc
- PEP 8 Style Guide: https://www.python.org/dev/peps/pep-0008/
- Data Science Best Practices: https://github.com/jupyter/jupyter/wiki

Practice Exercises:
1. Refactor an existing notebook following best practices
2. Set up version control for a notebook project
3. Create a notebook template for your team
4. Implement automated testing for notebook functions
5. Deploy a notebook as a dashboard or report
6. Conduct a code review of a colleague's notebook

Happy Notebook-ing! 📓✨
""")

if __name__ == "__main__":
    # Run the tutorial
    main()
    
    print("\n" + "="*60)
    print(" Tutorial completed successfully!")
    print(" Apply these best practices to your notebooks!")
    print("="*60) 