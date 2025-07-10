# Jupyter Deployment: Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Converting Notebooks to Scripts](#converting-notebooks-to-scripts)
3. [Automated Execution and Scheduling](#automated-execution-and-scheduling)
4. [Cloud Deployment Options](#cloud-deployment-options)
5. [Containerization with Docker](#containerization-with-docker)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [CI/CD Pipelines](#cicd-pipelines)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## Introduction

Deploying Jupyter notebooks enables reproducible, automated, and scalable workflows for data science and machine learning. This guide covers converting, automating, and deploying notebooks in production environments.

## Converting Notebooks to Scripts

### Using nbconvert

```bash
# Convert notebook to Python script
jupyter nbconvert --to script my_notebook.ipynb

# Convert to HTML, PDF, Markdown, or slides
jupyter nbconvert --to html my_notebook.ipynb
jupyter nbconvert --to pdf my_notebook.ipynb
jupyter nbconvert --to markdown my_notebook.ipynb
jupyter nbconvert --to slides my_notebook.ipynb
```

### Programmatic Conversion

```python
import nbformat
from nbconvert import PythonExporter

with open('my_notebook.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

exporter = PythonExporter()
source, _ = exporter.from_notebook_node(nb)

with open('my_notebook.py', 'w') as f:
    f.write(source)
```

## Automated Execution and Scheduling

### Using Papermill

```bash
pip install papermill
```

```bash
# Run a notebook with parameters
papermill input.ipynb output.ipynb -p param1 value1 -p param2 value2
```

### Using nbconvert for Execution

```bash
# Execute notebook and save output
jupyter nbconvert --to notebook --execute my_notebook.ipynb

# Execute and convert to HTML
jupyter nbconvert --to html --execute my_notebook.ipynb
```

### Scheduling with Cron (Linux/macOS)

```bash
# Edit crontab
crontab -e

# Run notebook every day at 2am
0 2 * * * papermill /path/to/input.ipynb /path/to/output.ipynb
```

### Scheduling with Task Scheduler (Windows)
- Use Task Scheduler to run a batch file or PowerShell script that calls Papermill or nbconvert.

## Cloud Deployment Options

### JupyterHub
- Multi-user Jupyter environment for teams and classrooms
- [JupyterHub Documentation](https://jupyterhub.readthedocs.io/)

### Binder
- Share reproducible environments online
- [Binder Project](https://mybinder.org/)
- Add a `binder/` directory with `environment.yml` or `requirements.txt`

### Google Colab
- Free cloud-hosted Jupyter notebooks
- [Google Colab](https://colab.research.google.com/)

### AWS SageMaker, Azure Notebooks, Databricks
- Managed cloud notebook services for enterprise workflows

## Containerization with Docker

### Basic Dockerfile

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
```

### Build and Run

```bash
# Build Docker image
docker build -t my-jupyter-app .

# Run container
docker run -p 8888:8888 my-jupyter-app
```

### Using docker-compose

```yaml
version: '3.8'
services:
  jupyter:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/app/notebooks
      - ./data:/app/data
    environment:
      - JUPYTER_TOKEN=your_secure_token
    restart: unless-stopped
```

## Monitoring and Logging

### Logging Output

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Notebook started")
```

### Monitoring Resource Usage

```python
import psutil
print(f"CPU: {psutil.cpu_percent()}%")
print(f"Memory: {psutil.virtual_memory().percent}%")
```

### JupyterHub Monitoring
- Use JupyterHub's admin panel for user/session/resource monitoring

## CI/CD Pipelines

### Example: GitHub Actions

```yaml
name: Run Jupyter Notebook
on: [push]
jobs:
  run-notebook:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install papermill
      - name: Run notebook
        run: |
          papermill input.ipynb output.ipynb
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: notebook-output
          path: output.ipynb
```

### Example: GitLab CI

```yaml
jupyter-job:
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - pip install papermill
    - papermill input.ipynb output.ipynb
  artifacts:
    paths:
      - output.ipynb
```

## Best Practices
- Use parameterized notebooks for automation
- Store secrets in environment variables, not notebooks
- Use version control for notebooks and Dockerfiles
- Monitor and log execution for reproducibility
- Test notebooks before deployment

## Troubleshooting

### Common Issues
- **Notebook fails to execute**: Check for missing dependencies or code errors
- **Docker build fails**: Check Dockerfile syntax and requirements
- **Cloud deployment errors**: Check service documentation and logs
- **CI/CD pipeline fails**: Check logs for missing packages or syntax errors

### Getting Help
- [Jupyter Documentation](https://jupyter.org/documentation)
- [Papermill Docs](https://papermill.readthedocs.io/)
- [JupyterHub Docs](https://jupyterhub.readthedocs.io/)
- [Docker Docs](https://docs.docker.com/)

---

**Deploy your Jupyter workflows with confidence!**