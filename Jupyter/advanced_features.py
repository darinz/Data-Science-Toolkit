#!/usr/bin/env python3
"""
Jupyter Advanced Features: Beyond the Basics

Welcome to the Jupyter Advanced Features tutorial! This guide covers advanced 
capabilities that extend Jupyter's functionality for complex workflows, 
multi-language support, and enterprise applications.

This script covers:
- Multi-language kernels (R, Julia, JavaScript)
- Parallel computing and distributed processing
- Database integration and connections
- API development and web services
- Custom extensions and plugins
- Enterprise features and security

Prerequisites:
- Python 3.8 or higher
- Basic understanding of Jupyter (covered in jupyter_basics.py)
- Familiarity with other programming languages (helpful)
"""

import os
import sys
import json
import subprocess
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
    
    print("Jupyter Advanced Features Tutorial")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print("Advanced features tutorial started successfully!")

    # Section 1: Introduction to Advanced Features
    print_section_header("1. Introduction to Advanced Features")
    
    print("""
Jupyter's advanced features extend its capabilities beyond basic notebook 
functionality, enabling complex workflows, multi-language support, and 
enterprise-grade applications.

Key Advanced Features:
- Multi-language kernel support
- Parallel and distributed computing
- Database and API integration
- Custom extensions and plugins
- Enterprise security and authentication
- High-performance computing
- Real-time collaboration
- Advanced visualization

Benefits:
✅ Multi-language data science workflows
✅ Scalable computing capabilities
✅ Enterprise integration
✅ Custom functionality development
✅ Enhanced security and collaboration
✅ Professional-grade applications
""")

    # Section 2: Multi-Language Kernels
    print_section_header("2. Multi-Language Kernels")
    
    print("""
Jupyter supports multiple programming languages through different kernels. 
This enables you to use the best language for each task in your workflow.

Supported Languages:
1. Python (ipykernel) - Default kernel
2. R (IRkernel) - Statistical computing
3. Julia (IJulia) - High-performance computing
4. JavaScript (ijavascript) - Web development
5. Scala (almond) - Big data processing
6. Java (IJava) - Enterprise applications
7. C++ (xeus-cling) - Systems programming
8. Go (gophernotes) - Systems and web development
""")

    # Demonstrate kernel examples
    print_subsection_header("Kernel Examples")
    
    kernel_examples = [
        ("R Kernel", "Statistical analysis and visualization", """
# R code example
library(ggplot2)
library(dplyr)

# Load and analyze data
data <- read.csv("data.csv")
summary(data)

# Create visualization
ggplot(data, aes(x=age, y=income)) + 
  geom_point() + 
  geom_smooth(method="lm") +
  theme_minimal() +
  labs(title="Age vs Income Relationship")
        """),
        
        ("Julia Kernel", "High-performance numerical computing", """
# Julia code example
using DataFrames
using Plots
using Statistics

# Load data
df = DataFrame(CSV.File("data.csv"))

# Perform computations
mean_income = mean(df.income)
std_income = std(df.income)

# Create plot
scatter(df.age, df.income, 
       xlabel="Age", ylabel="Income",
       title="Age vs Income",
       legend=false)
        """),
        
        ("JavaScript Kernel", "Web development and visualization", """
// JavaScript code example
const data = [
    {age: 25, income: 30000},
    {age: 35, income: 50000},
    {age: 45, income: 70000}
];

// Create visualization with D3.js
const svg = d3.select("body")
    .append("svg")
    .attr("width", 400)
    .attr("height", 300);

svg.selectAll("circle")
    .data(data)
    .enter()
    .append("circle")
    .attr("cx", d => d.age * 4)
    .attr("cy", d => 300 - d.income / 200)
    .attr("r", 5)
    .attr("fill", "steelblue");
        """)
    ]
    
    for language, description, example in kernel_examples:
        print(f"\n{language}:")
        print(f"Description: {description}")
        print(f"Example:")
        print(example)

    # Section 3: Parallel Computing
    print_section_header("3. Parallel Computing")
    
    print("""
Jupyter supports parallel computing through various libraries and extensions, 
enabling you to leverage multiple cores and distributed systems.

Parallel Computing Options:
1. Multiprocessing: Python's built-in parallel processing
2. IPyParallel: Interactive parallel computing
3. Dask: Parallel computing with task scheduling
4. Ray: Distributed computing framework
5. Spark: Big data processing
6. MPI: Message passing interface
""")

    # Demonstrate parallel computing examples
    print_subsection_header("Parallel Computing Examples")
    
    print("""
Multiprocessing Example:
```python
import multiprocessing as mp
import numpy as np
from functools import partial

def process_chunk(data_chunk):
    \"\"\"Process a chunk of data.\"\"\"
    return np.mean(data_chunk)

def parallel_processing(data, n_processes=None):
    \"\"\"Process data in parallel.\"\"\"
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    # Split data into chunks
    chunk_size = len(data) // n_processes
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
    
    # Process in parallel
    with mp.Pool(n_processes) as pool:
        results = pool.map(process_chunk, chunks)
    
    return results

# Usage
data = np.random.rand(1000000)
results = parallel_processing(data)
print(f"Results: {results}")
```

IPyParallel Example:
```python
import ipyparallel as ipp

# Start parallel engines
rc = ipp.Client()
view = rc.load_balanced_view()

# Define function to run in parallel
def square(x):
    return x ** 2

# Apply function to data in parallel
data = list(range(1000))
results = view.map(square, data)

# Collect results
squared_values = results.get()
print(f"First 10 squared values: {squared_values[:10]}")
```

Dask Example:
```python
import dask.dataframe as dd
import pandas as pd

# Create large dataset
df = pd.DataFrame({
    'x': np.random.rand(1000000),
    'y': np.random.rand(1000000)
})

# Convert to Dask DataFrame
ddf = dd.from_pandas(df, npartitions=4)

# Perform parallel operations
result = ddf.groupby('x').y.mean().compute()
print(f"Grouped result shape: {result.shape}")
```
""")

    # Section 4: Database Integration
    print_section_header("4. Database Integration")
    
    print("""
Jupyter can integrate with various databases for data storage, retrieval, 
and analysis. This enables working with large datasets and real-time data.

Supported Databases:
1. SQLite: Lightweight database
2. PostgreSQL: Advanced open-source database
3. MySQL: Popular relational database
4. MongoDB: NoSQL document database
5. Redis: In-memory data store
6. BigQuery: Google's data warehouse
7. Snowflake: Cloud data platform
""")

    # Demonstrate database integration
    print_subsection_header("Database Integration Examples")
    
    print("""
SQLite Integration:
```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('data.db')

# Create table
conn.execute('''
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY,
        name TEXT,
        age INTEGER,
        income REAL
    )
''')

# Insert data
data = [
    (1, 'Alice', 25, 50000),
    (2, 'Bob', 30, 60000),
    (3, 'Charlie', 35, 70000)
]
conn.executemany('INSERT INTO customers VALUES (?, ?, ?, ?)', data)
conn.commit()

# Query data
df = pd.read_sql_query('SELECT * FROM customers WHERE age > 25', conn)
print(df)

conn.close()
```

PostgreSQL Integration:
```python
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

# Connect using SQLAlchemy
engine = create_engine('postgresql://user:password@localhost/dbname')

# Read data
df = pd.read_sql('SELECT * FROM customers', engine)

# Write data
df.to_sql('new_table', engine, if_exists='replace', index=False)

# Complex query
query = '''
    SELECT 
        age_group,
        AVG(income) as avg_income,
        COUNT(*) as count
    FROM customers
    GROUP BY 
        CASE 
            WHEN age < 30 THEN 'Young'
            WHEN age < 50 THEN 'Middle'
            ELSE 'Senior'
        END
'''

result = pd.read_sql(query, engine)
print(result)
```

MongoDB Integration:
```python
from pymongo import MongoClient
import pandas as pd

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['analytics']
collection = db['customers']

# Insert data
data = [
    {'name': 'Alice', 'age': 25, 'income': 50000},
    {'name': 'Bob', 'age': 30, 'income': 60000},
    {'name': 'Charlie', 'age': 35, 'income': 70000}
]
collection.insert_many(data)

# Query data
results = collection.find({'age': {'$gt': 25}})
df = pd.DataFrame(list(results))
print(df)

# Aggregation pipeline
pipeline = [
    {'$group': {'_id': '$age', 'avg_income': {'$avg': '$income'}}},
    {'$sort': {'_id': 1}}
]

agg_results = list(collection.aggregate(pipeline))
print(agg_results)
```
""")

    # Section 5: API Development
    print_section_header("5. API Development")
    
    print("""
Jupyter can be used to develop and test APIs, enabling integration with 
web services and external data sources.

API Development Options:
1. Flask: Lightweight web framework
2. FastAPI: Modern, fast web framework
3. Django: Full-featured web framework
4. Requests: HTTP library for API calls
5. GraphQL: Query language for APIs
6. WebSockets: Real-time communication
""")

    # Demonstrate API development
    print_subsection_header("API Development Examples")
    
    print("""
Flask API Example:
```python
from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load data
df = pd.read_csv('customer_data.csv')

@app.route('/api/customers', methods=['GET'])
def get_customers():
    \"\"\"Get all customers with optional filtering.\"\"\"
    age_min = request.args.get('age_min', type=int)
    age_max = request.args.get('age_max', type=int)
    
    filtered_df = df.copy()
    
    if age_min is not None:
        filtered_df = filtered_df[filtered_df['age'] >= age_min]
    if age_max is not None:
        filtered_df = filtered_df[filtered_df['age'] <= age_max]
    
    return jsonify(filtered_df.to_dict('records'))

@app.route('/api/customers/<int:customer_id>', methods=['GET'])
def get_customer(customer_id):
    \"\"\"Get specific customer by ID.\"\"\"
    customer = df[df['id'] == customer_id]
    if customer.empty:
        return jsonify({'error': 'Customer not found'}), 404
    
    return jsonify(customer.iloc[0].to_dict())

@app.route('/api/analytics/summary', methods=['GET'])
def get_summary():
    \"\"\"Get analytics summary.\"\"\"
    summary = {
        'total_customers': len(df),
        'avg_age': df['age'].mean(),
        'avg_income': df['income'].mean(),
        'age_distribution': df['age'].value_counts().to_dict()
    }
    return jsonify(summary)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

FastAPI Example:
```python
from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
from typing import List, Optional

app = FastAPI(title="Customer Analytics API")

# Pydantic models
class Customer(BaseModel):
    id: int
    name: str
    age: int
    income: float

class CustomerCreate(BaseModel):
    name: str
    age: int
    income: float

# Load data
df = pd.read_csv('customer_data.csv')

@app.get('/customers/', response_model=List[Customer])
async def get_customers(
    age_min: Optional[int] = Query(None, description="Minimum age"),
    age_max: Optional[int] = Query(None, description="Maximum age")
):
    \"\"\"Get customers with optional age filtering.\"\"\"
    filtered_df = df.copy()
    
    if age_min is not None:
        filtered_df = filtered_df[filtered_df['age'] >= age_min]
    if age_max is not None:
        filtered_df = filtered_df[filtered_df['age'] <= age_max]
    
    return filtered_df.to_dict('records')

@app.post('/customers/', response_model=Customer)
async def create_customer(customer: CustomerCreate):
    \"\"\"Create a new customer.\"\"\"
    new_id = df['id'].max() + 1
    new_customer = {
        'id': new_id,
        'name': customer.name,
        'age': customer.age,
        'income': customer.income
    }
    
    df = df.append(new_customer, ignore_index=True)
    df.to_csv('customer_data.csv', index=False)
    
    return new_customer

@app.get('/analytics/summary')
async def get_analytics_summary():
    \"\"\"Get analytics summary.\"\"\"
    return {
        'total_customers': len(df),
        'avg_age': float(df['age'].mean()),
        'avg_income': float(df['income'].mean()),
        'age_distribution': df['age'].value_counts().to_dict()
    }
```

API Client Example:
```python
import requests
import pandas as pd

# API base URL
base_url = 'http://localhost:5000/api'

# Get all customers
response = requests.get(f'{base_url}/customers')
customers = response.json()
df = pd.DataFrame(customers)

# Get customers with age filter
response = requests.get(f'{base_url}/customers?age_min=25&age_max=35')
filtered_customers = response.json()

# Get analytics summary
response = requests.get(f'{base_url}/analytics/summary')
summary = response.json()

print(f"Total customers: {summary['total_customers']}")
print(f"Average age: {summary['avg_age']:.1f}")
print(f"Average income: ${summary['avg_income']:,.0f}")
```
""")

    # Section 6: Custom Extensions and Plugins
    print_section_header("6. Custom Extensions and Plugins")
    
    print("""
Jupyter's extensible architecture allows you to create custom extensions 
and plugins to enhance functionality and add specialized features.

Extension Types:
1. JupyterLab Extensions: Frontend extensions
2. Server Extensions: Backend extensions
3. Kernel Extensions: Language-specific extensions
4. Widget Extensions: Interactive components
5. Theme Extensions: Visual customization
""")

    # Demonstrate extension development
    print_subsection_header("Extension Development Examples")
    
    print("""
Simple JupyterLab Extension:
```python
# setup.py
from setuptools import setup, find_packages

setup(
    name='my-jupyterlab-extension',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'jupyterlab>=3.0.0',
    ],
    zip_safe=False,
    entry_points={
        'jupyterlab.extension': [
            'my-extension = my_extension',
        ],
    },
)

# my_extension/__init__.py
from ._version import __version__

def _jupyter_labextension_paths():
    return [{
        'src': 'labextension',
        'dest': 'my-extension'
    }]

# my_extension/labextension/package.json
{
  "name": "my-extension",
  "version": "0.1.0",
  "description": "A custom JupyterLab extension",
  "keywords": [
    "jupyter",
    "jupyterlab",
    "jupyterlab-extension:extension"
  ],
  "license": "BSD-3-Clause",
  "files": [
    "lib/**/*.{d.ts,eot,gif,html,jpg,js,js.map,json,png,svg,woff2,ttf}"
  ],
  "main": "lib/index.js",
  "types": "lib/index.d.ts",
  "style": "style/index.css",
  "repository": {
    "type": "git",
    "url": "https://github.com/user/my-extension.git"
  },
  "scripts": {
    "build": "tsc",
    "clean": "rimraf lib tsconfig.tsbuildinfo",
    "prepare": "npm run clean && npm run build",
    "watch": "tsc -w"
  },
  "dependencies": {
    "@jupyterlab/application": "^3.0.0",
    "@jupyterlab/notebook": "^3.0.0"
  },
  "devDependencies": {
    "typescript": "~4.1.3",
    "rimraf": "~3.0.2"
  },
  "jupyterlab": {
    "extension": true
  }
}

# my_extension/labextension/src/index.ts
import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';

import {
  INotebookTracker
} from '@jupyterlab/notebook';

/**
 * Initialization data for the my-extension extension.
 */
const extension: JupyterFrontEndPlugin<void> = {
  id: 'my-extension:plugin',
  autoStart: true,
  requires: [INotebookTracker],
  activate: (app: JupyterFrontEnd, tracker: INotebookTracker) => {
    console.log('My extension is activated!');
    
    // Add custom functionality
    tracker.widgetAdded.connect((sender, panel) => {
      console.log('New notebook opened:', panel.context.path);
    });
  }
};

export default extension;
```

Server Extension Example:
```python
# my_server_extension/__init__.py
from .handlers import setup_handlers

def _jupyter_server_extension_paths():
    return [{
        "module": "my_server_extension"
    }]

def load_jupyter_server_extension(nb_server_app):
    web_app = nb_server_app.web_app
    setup_handlers(web_app)

# my_server_extension/handlers.py
from notebook.utils import url_path_join
from tornado.web import RequestHandler
import json

class MyHandler(RequestHandler):
    def get(self):
        self.write(json.dumps({
            'message': 'Hello from custom server extension!',
            'status': 'success'
        }))

def setup_handlers(web_app):
    host_pattern = ".*$"
    route_pattern = url_path_join(web_app.settings["base_url"], "/my-extension")
    web_app.add_handlers(host_pattern, [(route_pattern, MyHandler)])
```
""")

    # Section 7: Enterprise Features
    print_section_header("7. Enterprise Features")
    
    print("""
Enterprise Jupyter deployments require additional features for security, 
scalability, and management in production environments.

Enterprise Features:
1. Authentication and Authorization
2. Multi-user environments (JupyterHub)
3. Resource management and quotas
4. Monitoring and logging
5. Backup and disaster recovery
6. Integration with enterprise systems
7. Compliance and audit trails
""")

    # Demonstrate enterprise features
    print_subsection_header("Enterprise Configuration Examples")
    
    print("""
JupyterHub Configuration:
```python
# jupyterhub_config.py
c = get_config()

# Authentication
c.JupyterHub.authenticator_class = 'oauthenticator.google.GoogleOAuthenticator'
c.GoogleOAuthenticator.client_id = 'your-client-id'
c.GoogleOAuthenticator.client_secret = 'your-client-secret'

# Spawner configuration
c.JupyterHub.spawner_class = 'dockerspawner.DockerSpawner'
c.DockerSpawner.image = 'jupyter/datascience-notebook:latest'
c.DockerSpawner.network_name = 'jupyterhub'
c.DockerSpawner.volumes = {
    '/home/jovyan/work': '/home/jovyan/work'
}

# Resource limits
c.DockerSpawner.mem_limit = '2G'
c.DockerSpawner.cpu_limit = 1.0

# SSL configuration
c.JupyterHub.ssl_cert = '/path/to/cert.pem'
c.JupyterHub.ssl_key = '/path/to/key.pem'

# Database
c.JupyterHub.db_url = 'postgresql://user:pass@localhost/jupyterhub'

# Logging
c.JupyterHub.log_level = 'INFO'
c.JupyterHub.log_file = '/var/log/jupyterhub.log'
```

Docker Compose for Enterprise:
```yaml
# docker-compose.yml
version: '3.8'

services:
  jupyterhub:
    image: jupyterhub/jupyterhub:latest
    ports:
      - "8000:8000"
    environment:
      - JUPYTER_ENABLE_LAB=yes
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - jupyter_data:/data
    depends_on:
      - postgres
      - redis

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: jupyterhub
      POSTGRES_USER: jupyterhub
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

volumes:
  jupyter_data:
  postgres_data:
  redis_data:
```

Monitoring Configuration:
```python
# monitoring_config.py
import logging
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
notebook_requests = Counter('notebook_requests_total', 'Total notebook requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/jupyter.log'),
        logging.StreamHandler()
    ]
)

# Custom middleware for monitoring
class MonitoringMiddleware:
    def __init__(self, app):
        self.app = app
    
    def __call__(self, environ, start_response):
        notebook_requests.inc()
        
        start_time = time.time()
        def custom_start_response(status, headers, exc_info=None):
            duration = time.time() - start_time
            request_duration.observe(duration)
            return start_response(status, headers, exc_info)
        
        return self.app(environ, custom_start_response)
```
""")

    # Section 8: High-Performance Computing
    print_section_header("8. High-Performance Computing")
    
    print("""
Jupyter can be integrated with high-performance computing (HPC) systems 
for large-scale data processing and scientific computing.

HPC Integration Options:
1. SLURM: Job scheduling system
2. PBS: Portable batch system
3. MPI: Message passing interface
4. GPU Computing: CUDA and OpenCL
5. Cloud Computing: AWS, GCP, Azure
6. Container Orchestration: Kubernetes
""")

    # Demonstrate HPC integration
    print_subsection_header("HPC Integration Examples")
    
    print("""
SLURM Integration:
```python
import subprocess
import json

def submit_slurm_job(script_path, job_name, nodes=1, cpus_per_task=1, memory='2G'):
    \"\"\"Submit a job to SLURM scheduler.\"\"\"
    cmd = [
        'sbatch',
        f'--job-name={job_name}',
        f'--nodes={nodes}',
        f'--cpus-per-task={cpus_per_task}',
        f'--mem={memory}',
        script_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

def get_job_status(job_id):
    \"\"\"Get status of a SLURM job.\"\"\"
    cmd = ['squeue', '--job', str(job_id), '--format', '%j %T %M %N']
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

# Example usage
job_id = submit_slurm_job('my_script.sh', 'data_analysis')
print(f"Submitted job: {job_id}")
print(f"Job status: {get_job_status(job_id)}")
```

MPI Integration:
```python
from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Distribute data
if rank == 0:
    # Master process creates data
    data = np.random.rand(1000, 1000)
else:
    data = None

# Scatter data to all processes
local_data = comm.scatter(data, root=0)

# Process local data
local_result = np.mean(local_data, axis=0)

# Gather results
all_results = comm.gather(local_result, root=0)

if rank == 0:
    # Combine results
    final_result = np.mean(all_results, axis=0)
    print(f"Final result: {final_result}")
```

GPU Computing with CuPy:
```python
import cupy as cp
import numpy as np

# Check GPU availability
if cp.cuda.is_available():
    print(f"GPU: {cp.cuda.Device().name}")
    
    # Create GPU arrays
    x_gpu = cp.random.rand(10000, 10000)
    y_gpu = cp.random.rand(10000, 10000)
    
    # Perform GPU computation
    start_time = cp.cuda.Event()
    end_time = cp.cuda.Event()
    
    start_time.record()
    z_gpu = cp.dot(x_gpu, y_gpu)
    end_time.record()
    
    end_time.synchronize()
    gpu_time = cp.cuda.get_elapsed_time(start_time, end_time)
    
    print(f"GPU computation time: {gpu_time:.2f} ms")
    
    # Compare with CPU
    x_cpu = cp.asnumpy(x_gpu)
    y_cpu = cp.asnumpy(y_gpu)
    
    import time
    start_time = time.time()
    z_cpu = np.dot(x_cpu, y_cpu)
    cpu_time = (time.time() - start_time) * 1000
    
    print(f"CPU computation time: {cpu_time:.2f} ms")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")
else:
    print("No GPU available")
```
""")

    # Section 9: Real-time Collaboration
    print_section_header("9. Real-time Collaboration")
    
    print("""
Real-time collaboration features enable multiple users to work on the same 
notebook simultaneously, enhancing team productivity.

Collaboration Features:
1. Real-time editing
2. Live cursor tracking
3. Chat and comments
4. Version control integration
5. Conflict resolution
6. User presence indicators
""")

    # Demonstrate collaboration features
    print_subsection_header("Collaboration Examples")
    
    print("""
JupyterLab Real-time Collaboration:
```python
# jupyterlab_config.py
c = get_config()

# Enable real-time collaboration
c.ServerApp.collaborative = True
c.ServerApp.collaborative_room_id = 'my-room'

# Configure collaboration settings
c.CollaborativeDrive.collaborative = True
c.CollaborativeDrive.room_id = 'my-room'

# User authentication for collaboration
c.ServerApp.token = ''
c.ServerApp.password = ''

# WebSocket configuration
c.ServerApp.websocket_url = 'ws://localhost:8000'
```

Collaborative Widget Example:
```python
import ipywidgets as widgets
from IPython.display import display
import json

class CollaborativeWidget:
    def __init__(self, room_id):
        self.room_id = room_id
        self.users = {}
        self.messages = []
        
        # Create UI
        self.chat_input = widgets.Text(description='Message:')
        self.chat_output = widgets.HTML(value='<h3>Chat</h3>')
        self.user_list = widgets.HTML(value='<h3>Online Users</h3>')
        
        # Set up event handlers
        self.chat_input.on_submit(self.send_message)
        
        # Display widget
        display(widgets.VBox([
            self.user_list,
            self.chat_output,
            self.chat_input
        ]))
    
    def send_message(self, change):
        \"\"\"Send a message to the room.\"\"\"
        message = change['new']
        if message.strip():
            self.messages.append({
                'user': 'You',
                'message': message,
                'timestamp': datetime.now().isoformat()
            })
            self.update_chat_display()
            self.chat_input.value = ''
    
    def update_chat_display(self):
        \"\"\"Update the chat display.\"\"\"
        chat_html = '<h3>Chat</h3>'
        for msg in self.messages[-10:]:  # Show last 10 messages
            chat_html += f'<p><strong>{msg["user"]}:</strong> {msg["message"]}</p>'
        self.chat_output.value = chat_html

# Usage
collab_widget = CollaborativeWidget('my-room')
```
""")

    # Section 10: Advanced Visualization
    print_section_header("10. Advanced Visualization")
    
    print("""
Advanced visualization capabilities enable creating interactive, 
high-performance visualizations for complex data analysis.

Visualization Libraries:
1. Plotly: Interactive plots
2. Bokeh: Web-based visualization
3. HoloViews: High-level plotting
4. Datashader: Large dataset visualization
5. PyViz: Python visualization ecosystem
6. Three.js: 3D visualization
""")

    # Demonstrate advanced visualization
    print_subsection_header("Advanced Visualization Examples")
    
    print("""
Interactive 3D Visualization:
```python
import plotly.graph_objects as go
import numpy as np

# Create 3D data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# Create 3D surface plot
fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])

fig.update_layout(
    title='3D Surface Plot',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    width=800,
    height=600
)

fig.show()
```

Large Dataset Visualization with Datashader:
```python
import datashader as ds
import datashader.transfer_functions as tf
import pandas as pd
import numpy as np

# Create large dataset
n = 1000000
df = pd.DataFrame({
    'x': np.random.randn(n),
    'y': np.random.randn(n),
    'category': np.random.choice(['A', 'B', 'C'], n)
})

# Create canvas
canvas = ds.Canvas(plot_width=800, plot_height=600)

# Aggregate data
agg = canvas.points(df, 'x', 'y', ds.count_cat('category'))

# Create image
img = tf.shade(agg, color_key=['red', 'green', 'blue'])

# Display
from IPython.display import display
display(img)
```

Interactive Dashboard:
```python
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

# Load data
df = pd.read_csv('customer_data.csv')

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Customer Analytics Dashboard'),
    
    html.Div([
        html.Label('Select Age Range:'),
        dcc.RangeSlider(
            id='age-slider',
            min=df['age'].min(),
            max=df['age'].max(),
            value=[df['age'].min(), df['age'].max()],
            marks={i: str(i) for i in range(0, 101, 10)}
        )
    ]),
    
    dcc.Graph(id='scatter-plot'),
    dcc.Graph(id='histogram')
])

@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('histogram', 'figure')],
    [Input('age-slider', 'value')]
)
def update_graphs(age_range):
    filtered_df = df[(df['age'] >= age_range[0]) & (df['age'] <= age_range[1])]
    
    scatter_fig = px.scatter(
        filtered_df, x='age', y='income', 
        title='Age vs Income',
        color='cluster'
    )
    
    hist_fig = px.histogram(
        filtered_df, x='income', 
        title='Income Distribution',
        nbins=20
    )
    
    return scatter_fig, hist_fig

# Run app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
```
""")

    # Section 11: Summary and Next Steps
    print_section_header("11. Summary and Next Steps")
    
    print("""
Congratulations! You've completed the Jupyter Advanced Features tutorial. Here's what you've learned:

Key Concepts Covered:
✅ Multi-Language Kernels: Working with R, Julia, JavaScript, and more
✅ Parallel Computing: Multiprocessing, IPyParallel, Dask, and Ray
✅ Database Integration: SQLite, PostgreSQL, MongoDB, and more
✅ API Development: Flask, FastAPI, and RESTful services
✅ Custom Extensions: JupyterLab and server extensions
✅ Enterprise Features: Authentication, monitoring, and security
✅ High-Performance Computing: SLURM, MPI, and GPU computing
✅ Real-time Collaboration: Multi-user editing and communication
✅ Advanced Visualization: Interactive and large-scale plotting
✅ Production Deployment: Scalable and secure deployments

Next Steps:

1. Explore Multi-Language Workflows: Combine different kernels for specialized tasks
2. Implement Parallel Computing: Optimize performance for large datasets
3. Build Database Integrations: Connect to your data sources
4. Develop Custom APIs: Create services for your applications
5. Create Custom Extensions: Build specialized functionality
6. Deploy Enterprise Solutions: Scale Jupyter for your organization
7. Master HPC Integration: Leverage high-performance computing resources
8. Enable Real-time Collaboration: Enhance team productivity
9. Create Advanced Visualizations: Build interactive dashboards
10. Optimize for Production: Deploy robust, scalable solutions

Additional Resources:
- Jupyter Documentation: https://jupyter.org/
- JupyterLab Extensions: https://jupyterlab.readthedocs.io/en/stable/user/extensions.html
- IPyParallel: https://ipyparallel.readthedocs.io/
- Dask: https://dask.org/
- FastAPI: https://fastapi.tiangolo.com/
- Plotly: https://plotly.com/python/
- Datashader: https://datashader.org/

Practice Exercises:
1. Set up a multi-language workflow using R and Python kernels
2. Implement parallel processing for a large dataset
3. Create a database integration for your data sources
4. Build a REST API for your analysis results
5. Develop a custom JupyterLab extension
6. Deploy JupyterHub for your team
7. Create an interactive dashboard with real-time data
8. Optimize a notebook for GPU computing
9. Implement real-time collaboration features
10. Build a production-ready Jupyter deployment

Happy Advanced Jupyter-ing! 🚀✨
""")

if __name__ == "__main__":
    # Run the tutorial
    main()
    
    print("\n" + "="*60)
    print(" Tutorial completed successfully!")
    print(" Explore these advanced features in Jupyter!")
    print("="*60) 