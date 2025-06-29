# Jupyter Advanced Features: Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Multi-language Kernels](#multi-language-kernels)
3. [Parallel and Distributed Computing](#parallel-and-distributed-computing)
4. [Database Integration](#database-integration)
5. [API Development and Web Services](#api-development-and-web-services)
6. [Custom Extensions and Plugins](#custom-extensions-and-plugins)
7. [Enterprise Features and Security](#enterprise-features-and-security)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Introduction

Jupyter supports advanced features for power users, including multi-language support, parallel computing, database integration, and extensibility. These features enable robust workflows for data science, research, and enterprise use.

## Multi-language Kernels

Jupyter can run code in many languages via kernels.

### Installing Additional Kernels

```bash
# R kernel
conda install -c r r-irkernel

# Julia kernel
conda install -c conda-forge julia
julia -e 'using Pkg; Pkg.add("IJulia")'

# JavaScript kernel
pip install ijavascript
ijsinstall

# List available kernels
jupyter kernelspec list
```

### Using Multiple Languages in One Notebook

```python
# Python cell
import numpy as np
x = np.linspace(0, 10, 100)

# R cell (if R kernel installed)
%%R
summary(cars)

# Julia cell (if Julia kernel installed)
%%julia
sqrt(2)

# Bash cell
%%bash
echo "Hello from Bash!"
```

## Parallel and Distributed Computing

Jupyter can leverage parallelism for faster computation.

### IPython Parallel

```bash
# Install IPython parallel
pip install ipyparallel

# Start controller and engines
ipcluster start -n 4
```

```python
# Connect to cluster
import ipyparallel as ipp
rc = ipp.Client()
lview = rc.load_balanced_view()

# Parallel map
def square(x):
    return x ** 2

results = lview.map(square, range(10))
print(list(results))
```

### Dask for Distributed Computing

```bash
pip install dask[distributed]
```

```python
from dask.distributed import Client
client = Client()

import dask.array as da
x = da.random.random((10000, 10000), chunks=(1000, 1000))
result = x.mean().compute()
print(result)
```

## Database Integration

Jupyter can connect to SQL and NoSQL databases for data analysis.

### SQL Databases

```bash
pip install sqlalchemy ipython-sql
```

```python
# Load SQL extension
%load_ext sql

# Connect to SQLite
%sql sqlite:///my_database.db

# Query data
%%sql
SELECT * FROM users LIMIT 5;
```

### Pandas Database Integration

```python
import pandas as pd
import sqlite3

conn = sqlite3.connect('my_database.db')
df = pd.read_sql_query('SELECT * FROM users', conn)
print(df.head())
```

### NoSQL Databases (MongoDB Example)

```bash
pip install pymongo
```

```python
from pymongo import MongoClient
client = MongoClient('mongodb://localhost:27017/')
db = client['mydb']
collection = db['users']

for doc in collection.find().limit(5):
    print(doc)
```

## API Development and Web Services

Jupyter can be used to prototype and test APIs.

### Using Flask in a Notebook

```python
from flask import Flask, jsonify
app = Flask(__name__)

@app.route('/api/hello')
def hello():
    return jsonify({'message': 'Hello, world!'})

# Run Flask app (in a script, not directly in notebook)
# app.run(port=5000)
```

### Calling APIs from Jupyter

```python
import requests
response = requests.get('https://api.github.com/users/octocat')
print(response.json())
```

## Custom Extensions and Plugins

Jupyter is highly extensible.

### Installing Extensions

```bash
# Jupyter Notebook extensions
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# Enable an extension
jupyter nbextension enable codefolding/main
```

### Writing a Simple Notebook Extension

```python
# Example: Add a custom button to the toolbar
from IPython.display import display, Javascript

def add_button():
    display(Javascript('''
        Jupyter.toolbar.add_buttons_group([
            {
                'label': 'My Button',
                'icon': 'fa-rocket',
                'callback': function() { alert('Button clicked!'); }
            }
        ]);
    '''))

add_button()
```

## Enterprise Features and Security

### Authentication and Access Control
- Use JupyterHub for multi-user environments: https://jupyterhub.readthedocs.io/
- Integrate with OAuth, LDAP, or custom authentication

### Notebook Security
- Never share notebooks with embedded credentials
- Use environment variables for secrets
- Use signed notebook execution for trusted code

### Auditing and Logging
- Enable logging for notebook actions
- Use JupyterHub's audit features for enterprise deployments

## Best Practices
- Use version control for notebooks and extensions
- Document custom kernels and extensions
- Regularly update and audit installed extensions
- Use virtual environments for isolation
- Monitor resource usage in parallel/distributed jobs

## Troubleshooting

### Common Issues
- **Kernel not found**: Check kernel installation and `jupyter kernelspec list`
- **Parallel jobs not running**: Ensure engines are started and accessible
- **Database connection errors**: Check credentials and network access
- **Extension not loading**: Check compatibility and enable status

### Getting Help
- [Jupyter Documentation](https://jupyter.org/documentation)
- [Jupyter Discourse](https://discourse.jupyter.org/)
- [JupyterHub Docs](https://jupyterhub.readthedocs.io/)

---

**Explore the power of Jupyter's advanced features!** 🚀 