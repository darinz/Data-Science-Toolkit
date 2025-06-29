#!/usr/bin/env python3
"""
Jupyter Deployment: From Development to Production

Welcome to the Jupyter Deployment tutorial! This guide covers how to deploy 
Jupyter notebooks and applications to production environments, including 
automated execution, cloud deployment, and monitoring.

This script covers:
- Converting notebooks to scripts
- Automated execution and scheduling
- Cloud deployment options
- Containerization with Docker
- Monitoring and logging
- CI/CD pipelines

Prerequisites:
- Python 3.8 or higher
- Basic understanding of Jupyter (covered in jupyter_basics.py)
- Familiarity with command line and cloud platforms
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
    
    print("Jupyter Deployment Tutorial")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print("Deployment tutorial started successfully!")

    # Section 1: Introduction to Jupyter Deployment
    print_section_header("1. Introduction to Jupyter Deployment")
    
    print("""
Deploying Jupyter notebooks to production involves converting interactive 
notebooks into automated, scalable applications that can run reliably 
in production environments.

Deployment Goals:
- Automate notebook execution
- Scale to handle large workloads
- Ensure reliability and monitoring
- Integrate with existing systems
- Maintain security and compliance

Deployment Options:
1. Script Conversion: Convert notebooks to Python scripts
2. Automated Execution: Schedule and run notebooks automatically
3. Cloud Deployment: Deploy to cloud platforms
4. Containerization: Use Docker for consistent environments
5. Orchestration: Manage with Kubernetes or similar tools
6. Monitoring: Track execution and performance
""")

    # Section 2: Converting Notebooks to Scripts
    print_section_header("2. Converting Notebooks to Scripts")
    
    print("""
Converting Jupyter notebooks to Python scripts is often the first step 
in deployment. This enables automated execution and integration with 
existing systems.

Conversion Methods:
1. nbconvert: Official Jupyter conversion tool
2. Papermill: Parameterized notebook execution
3. Jupytext: Bidirectional conversion
4. Manual conversion: Hand-crafted scripts
""")

    # Demonstrate conversion examples
    print_subsection_header("Notebook Conversion Examples")
    
    print("""
Using nbconvert:
```bash
# Convert notebook to Python script
jupyter nbconvert --to script my_notebook.ipynb

# Convert to HTML for sharing
jupyter nbconvert --to html my_notebook.ipynb

# Convert to PDF (requires LaTeX)
jupyter nbconvert --to pdf my_notebook.ipynb

# Convert with custom template
jupyter nbconvert --to script --template my_template.py my_notebook.ipynb
```

Using Papermill:
```python
import papermill as pm

# Execute notebook with parameters
pm.execute_notebook(
    'input_notebook.ipynb',
    'output_notebook.ipynb',
    parameters={
        'input_file': 'data.csv',
        'output_file': 'results.csv',
        'n_clusters': 5
    }
)

# Execute with custom kernel
pm.execute_notebook(
    'notebook.ipynb',
    'output.ipynb',
    kernel_name='python3',
    parameters={'param1': 'value1'}
)
```

Using Jupytext:
```python
import jupytext

# Convert notebook to Python script
jupytext.write('notebook.ipynb', 'script.py')

# Convert Python script back to notebook
jupytext.write('script.py', 'notebook.ipynb')

# Sync notebook and script
jupytext.sync('notebook.ipynb', 'script.py')
```

Manual Conversion Example:
```python
# Original notebook cell:
# import pandas as pd
# df = pd.read_csv('data.csv')
# print(df.head())

# Converted Python script:
#!/usr/bin/env python3
\"\"\"
Converted from Jupyter notebook
Original: data_analysis.ipynb
Date: 2024-01-15
\"\"\"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    \"\"\"Main function for data analysis.\"\"\"
    # Load data
    df = pd.read_csv('data.csv')
    print(df.head())
    
    # Perform analysis
    results = analyze_data(df)
    
    # Save results
    save_results(results)
    
    return results

def analyze_data(df):
    \"\"\"Analyze the data.\"\"\"
    # Analysis code here
    return {'summary': df.describe()}

def save_results(results):
    \"\"\"Save analysis results.\"\"\"
    with open('results.json', 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    main()
```
""")

    # Section 3: Automated Execution
    print_section_header("3. Automated Execution")
    
    print("""
Automated execution enables notebooks and scripts to run on schedules, 
triggered by events, or as part of larger workflows.

Scheduling Options:
1. Cron: Unix/Linux task scheduler
2. Windows Task Scheduler: Windows automation
3. Airflow: Workflow orchestration
4. Prefect: Modern workflow management
5. GitHub Actions: CI/CD automation
6. Cloud schedulers: AWS EventBridge, GCP Cloud Scheduler
""")

    # Demonstrate automated execution
    print_subsection_header("Automated Execution Examples")
    
    print("""
Cron Job Example:
```bash
# Edit crontab
crontab -e

# Run notebook daily at 2 AM
0 2 * * * /usr/bin/python3 /path/to/execute_notebook.py

# Run with specific environment
0 2 * * * /path/to/conda/envs/myenv/bin/python /path/to/script.py

# Run with logging
0 2 * * * /usr/bin/python3 /path/to/script.py >> /var/log/notebook.log 2>&1
```

Python Script for Automated Execution:
```python
#!/usr/bin/env python3
\"\"\"
Automated notebook execution script
\"\"\"

import subprocess
import logging
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('notebook_execution.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def execute_notebook(notebook_path, output_path=None):
    \"\"\"Execute a Jupyter notebook.\"\"\"
    try:
        logging.info(f'Starting execution of {notebook_path}')
        
        cmd = [
            'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute', notebook_path
        ]
        
        if output_path:
            cmd.extend(['--output', output_path])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logging.info(f'Successfully executed {notebook_path}')
            return True
        else:
            logging.error(f'Failed to execute {notebook_path}: {result.stderr}')
            return False
            
    except Exception as e:
        logging.error(f'Error executing {notebook_path}: {e}')
        return False

def main():
    \"\"\"Main execution function.\"\"\"
    notebooks = [
        'data_processing.ipynb',
        'model_training.ipynb',
        'report_generation.ipynb'
    ]
    
    success_count = 0
    total_count = len(notebooks)
    
    for notebook in notebooks:
        if Path(notebook).exists():
            if execute_notebook(notebook):
                success_count += 1
        else:
            logging.warning(f'Notebook not found: {notebook}')
    
    logging.info(f'Execution completed: {success_count}/{total_count} successful')
    
    # Exit with error code if any notebook failed
    if success_count < total_count:
        sys.exit(1)

if __name__ == '__main__':
    main()
```

Airflow DAG Example:
```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import papermill as pm

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'notebook_pipeline',
    default_args=default_args,
    description='Automated notebook execution pipeline',
    schedule_interval=timedelta(days=1),
    catchup=False
)

def execute_data_processing():
    \"\"\"Execute data processing notebook.\"\"\"
    pm.execute_notebook(
        'notebooks/data_processing.ipynb',
        'outputs/data_processing_output.ipynb',
        parameters={'date': '{{ ds }}'}
    )

def execute_model_training():
    \"\"\"Execute model training notebook.\"\"\"
    pm.execute_notebook(
        'notebooks/model_training.ipynb',
        'outputs/model_training_output.ipynb',
        parameters={'date': '{{ ds }}'}
    )

def execute_reporting():
    \"\"\"Execute reporting notebook.\"\"\"
    pm.execute_notebook(
        'notebooks/reporting.ipynb',
        'outputs/reporting_output.ipynb',
        parameters={'date': '{{ ds }}'}
    )

# Define tasks
data_processing_task = PythonOperator(
    task_id='data_processing',
    python_callable=execute_data_processing,
    dag=dag
)

model_training_task = PythonOperator(
    task_id='model_training',
    python_callable=execute_model_training,
    dag=dag
)

reporting_task = PythonOperator(
    task_id='reporting',
    python_callable=execute_reporting,
    dag=dag
)

# Define dependencies
data_processing_task >> model_training_task >> reporting_task
```
""")

    # Section 4: Cloud Deployment
    print_section_header("4. Cloud Deployment")
    
    print("""
Cloud deployment enables scalable, managed execution of Jupyter notebooks 
and applications across various cloud platforms.

Cloud Platforms:
1. AWS: SageMaker, Lambda, ECS, EKS
2. Google Cloud: Vertex AI, Cloud Run, GKE
3. Azure: Machine Learning, Functions, AKS
4. IBM Cloud: Watson Studio, Cloud Functions
5. DigitalOcean: App Platform, Kubernetes
""")

    # Demonstrate cloud deployment
    print_subsection_header("Cloud Deployment Examples")
    
    print("""
AWS SageMaker Example:
```python
import sagemaker
from sagemaker import get_execution_role
from sagemaker.sklearn import SKLearn

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Create SKLearn estimator
sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version='0.23-1',
    py_version='py3'
)

# Train model
sklearn_estimator.fit({'train': 's3://bucket/train-data'})

# Deploy model
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)
```

Google Cloud Functions Example:
```python
# main.py
import functions_framework
import pandas as pd
import json

@functions_framework.http
def process_data(request):
    \"\"\"HTTP Cloud Function for data processing.\"\"\"
    # Get request data
    request_json = request.get_json(silent=True)
    
    if request_json and 'data' in request_json:
        # Process data
        df = pd.DataFrame(request_json['data'])
        result = df.describe().to_dict()
        
        return json.dumps(result)
    else:
        return json.dumps({'error': 'No data provided'})

# requirements.txt
# functions-framework==3.*
# pandas==1.*
# numpy==1.*

# Deploy command
# gcloud functions deploy process_data --runtime python39 --trigger-http --allow-unauthenticated
```

Azure Functions Example:
```python
# function_app.py
import azure.functions as func
import pandas as pd
import json

def main(req: func.HttpRequest) -> func.HttpResponse:
    \"\"\"Azure Function for data processing.\"\"\"
    try:
        # Get request data
        req_body = req.get_json()
        
        if 'data' in req_body:
            # Process data
            df = pd.DataFrame(req_body['data'])
            result = df.describe().to_dict()
            
            return func.HttpResponse(
                json.dumps(result),
                mimetype='application/json'
            )
        else:
            return func.HttpResponse(
                json.dumps({'error': 'No data provided'}),
                status_code=400,
                mimetype='application/json'
            )
    except Exception as e:
        return func.HttpResponse(
            json.dumps({'error': str(e)}),
            status_code=500,
            mimetype='application/json'
        )

# requirements.txt
# azure-functions
# pandas
# numpy
```

Docker Deployment Example:
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "app.py"]
```

# requirements.txt
# jupyter==1.*
# papermill==2.*
# pandas==1.*
# numpy==1.*
# matplotlib==3.*
# seaborn==0.*
# scikit-learn==1.*

# docker-compose.yml
version: '3.8'

services:
  jupyter-app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./notebooks:/app/notebooks
      - ./data:/app/data
      - ./outputs:/app/outputs
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - JUPYTER_TOKEN=your_token_here
    restart: unless-stopped
```
""")

    # Section 5: Containerization
    print_section_header("5. Containerization")
    
    print("""
Containerization provides consistent, portable environments for Jupyter 
applications across different platforms and deployment scenarios.

Container Benefits:
- Consistent environments
- Easy deployment and scaling
- Version control for environments
- Isolation and security
- Resource management
""")

    # Demonstrate containerization
    print_subsection_header("Containerization Examples")
    
    print("""
Multi-stage Dockerfile:
```dockerfile
# Build stage
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Add local bin to PATH
ENV PATH=/root/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8000", "--no-browser", "--allow-root"]
```

Kubernetes Deployment:
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jupyter-deployment
  labels:
    app: jupyter
spec:
  replicas: 3
  selector:
    matchLabels:
      app: jupyter
  template:
    metadata:
      labels:
        app: jupyter
    spec:
      containers:
      - name: jupyter
        image: my-jupyter-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: JUPYTER_TOKEN
          valueFrom:
            secretKeyRef:
              name: jupyter-secrets
              key: token
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: jupyter-service
spec:
  selector:
    app: jupyter
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: Secret
metadata:
  name: jupyter-secrets
type: Opaque
data:
  token: <base64-encoded-token>
```

Docker Compose for Development:
```yaml
# docker-compose.yml
version: '3.8'

services:
  jupyter:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./data:/home/jovyan/data
      - ./outputs:/home/jovyan/outputs
    environment:
      - JUPYTER_ENABLE_LAB=yes
      - JUPYTER_TOKEN=dev_token
    restart: unless-stopped
    networks:
      - jupyter-network

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: jupyter
      POSTGRES_USER: jupyter
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - jupyter-network

  redis:
    image: redis:6-alpine
    networks:
      - jupyter-network

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - jupyter
    networks:
      - jupyter-network

volumes:
  postgres_data:

networks:
  jupyter-network:
    driver: bridge
```
""")

    # Section 6: Monitoring and Logging
    print_section_header("6. Monitoring and Logging")
    
    print("""
Monitoring and logging are essential for production deployments to track 
performance, detect issues, and ensure reliability.

Monitoring Components:
1. Application metrics
2. System metrics
3. Error tracking
4. Performance monitoring
5. User analytics
6. Alerting systems
""")

    # Demonstrate monitoring and logging
    print_subsection_header("Monitoring and Logging Examples")
    
    print("""
Application Monitoring:
```python
import logging
import time
import psutil
import os
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
notebook_executions = Counter('notebook_executions_total', 'Total notebook executions')
execution_duration = Histogram('notebook_execution_duration_seconds', 'Notebook execution time')
memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/jupyter.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def monitor_execution(func):
    \"\"\"Decorator to monitor function execution.\"\"\"
    def wrapper(*args, **kwargs):
        start_time = time.time()
        notebook_executions.inc()
        
        try:
            result = func(*args, **kwargs)
            execution_duration.observe(time.time() - start_time)
            logger.info(f'Successfully executed {func.__name__}')
            return result
        except Exception as e:
            logger.error(f'Error executing {func.__name__}: {e}')
            raise
    
    return wrapper

def update_system_metrics():
    \"\"\"Update system metrics.\"\"\"
    process = psutil.Process(os.getpid())
    memory_usage.set(process.memory_info().rss)
    cpu_usage.set(psutil.cpu_percent())

@monitor_execution
def execute_notebook(notebook_path):
    \"\"\"Execute a notebook with monitoring.\"\"\"
    update_system_metrics()
    
    # Notebook execution code here
    time.sleep(2)  # Simulate execution
    
    update_system_metrics()
    return True

# Start metrics server
start_http_server(8000)

# Example usage
if __name__ == '__main__':
    execute_notebook('test.ipynb')
```

ELK Stack Configuration:
```yaml
# docker-compose.yml for ELK stack
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.17.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:7.17.0
    ports:
      - "5044:5044"
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:7.17.0
    ports:
      - "5601:5601"
    environment:
      ELASTICSEARCH_HOSTS: http://elasticsearch:9200
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

# logstash.conf
input {
  file {
    path => "/var/log/jupyter.log"
    type => "jupyter"
  }
}

filter {
  if [type] == "jupyter" {
    grok {
      match => { "message" => "%{TIMESTAMP_ISO8601:timestamp} - %{WORD:logger} - %{LOGLEVEL:level} - %{GREEDYDATA:message}" }
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "jupyter-logs-%{+YYYY.MM.dd}"
  }
}
```

Grafana Dashboard:
```json
{
  "dashboard": {
    "title": "Jupyter Monitoring Dashboard",
    "panels": [
      {
        "title": "Notebook Executions",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(notebook_executions_total[5m])",
            "legendFormat": "Executions/sec"
          }
        ]
      },
      {
        "title": "Execution Duration",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(notebook_execution_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "memory_usage_bytes",
            "legendFormat": "Memory (bytes)"
          }
        ]
      },
      {
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "cpu_usage_percent",
            "legendFormat": "CPU (%)"
          }
        ]
      }
    ]
  }
}
```
""")

    # Section 7: CI/CD Pipelines
    print_section_header("7. CI/CD Pipelines")
    
    print("""
Continuous Integration and Continuous Deployment (CI/CD) pipelines automate 
the testing, building, and deployment of Jupyter applications.

CI/CD Benefits:
- Automated testing
- Consistent deployments
- Version control integration
- Rollback capabilities
- Quality assurance
""")

    # Demonstrate CI/CD pipelines
    print_subsection_header("CI/CD Pipeline Examples")
    
    print("""
GitHub Actions Workflow:
```yaml
# .github/workflows/deploy.yml
name: Deploy Jupyter Application

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest --cov=./ --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v1
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v1
    
    - name: Login to Docker Hub
      uses: docker/login-action@v1
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v2
      with:
        context: .
        push: true
        tags: |
          myusername/jupyter-app:latest
          myusername/jupyter-app:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Deploy to Kubernetes
      uses: steebchen/kubectl@v2
      with:
        config: ${{ secrets.KUBE_CONFIG_DATA }}
        command: apply -f k8s/
    
    - name: Update deployment
      uses: steebchen/kubectl@v2
      with:
        config: ${{ secrets.KUBE_CONFIG_DATA }}
        command: set image deployment/jupyter-deployment jupyter=myusername/jupyter-app:${{ github.sha }}
```

GitLab CI/CD Pipeline:
```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

test:
  stage: test
  image: python:3.9
  script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov
    - pytest --cov=./ --cov-report=xml
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - docker tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA $CI_REGISTRY_IMAGE:latest
    - docker push $CI_REGISTRY_IMAGE:latest
  only:
    - main

deploy:
  stage: deploy
  image: alpine:latest
  before_script:
    - apk add --no-cache curl
  script:
    - curl -X POST $DEPLOY_WEBHOOK_URL
  only:
    - main
```

Jenkins Pipeline:
```groovy
// Jenkinsfile
pipeline {
    agent any
    
    environment {
        DOCKER_IMAGE = 'myusername/jupyter-app'
        DOCKER_TAG = "${env.BUILD_NUMBER}"
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Test') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pip install pytest pytest-cov'
                sh 'pytest --cov=./ --cov-report=xml'
            }
            post {
                always {
                    publishCoverage adapters: [coberturaAdapter('coverage.xml')]
                }
            }
        }
        
        stage('Build') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}")
                }
            }
        }
        
        stage('Push') {
            steps {
                script {
                    docker.withRegistry('https://registry.hub.docker.com', 'docker-hub-credentials') {
                        docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").push()
                        docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").push('latest')
                    }
                }
            }
        }
        
        stage('Deploy') {
            steps {
                script {
                    sh "kubectl set image deployment/jupyter-deployment jupyter=${DOCKER_IMAGE}:${DOCKER_TAG}"
                }
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
    }
}
```
""")

    # Section 8: Security Best Practices
    print_section_header("8. Security Best Practices")
    
    print("""
Security is crucial for production deployments. Implement appropriate 
security measures to protect data, applications, and infrastructure.

Security Considerations:
1. Authentication and authorization
2. Data encryption
3. Network security
4. Container security
5. Secrets management
6. Audit logging
""")

    # Demonstrate security practices
    print_subsection_header("Security Examples")
    
    print("""
Secrets Management:
```python
# Using environment variables
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL')
API_KEY = os.getenv('API_KEY')
SECRET_KEY = os.getenv('SECRET_KEY')

# Using AWS Secrets Manager
import boto3
import json

def get_secret(secret_name):
    \"\"\"Retrieve secret from AWS Secrets Manager.\"\"\"
    client = boto3.client('secretsmanager')
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
        return json.loads(response['SecretString'])
    except Exception as e:
        print(f'Error retrieving secret: {e}')
        return None

# Usage
secrets = get_secret('jupyter-secrets')
database_url = secrets['database_url']
api_key = secrets['api_key']
```

Kubernetes Secrets:
```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: jupyter-secrets
type: Opaque
data:
  database-url: <base64-encoded-url>
  api-key: <base64-encoded-key>
  jwt-secret: <base64-encoded-secret>

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jupyter-deployment
spec:
  template:
    spec:
      containers:
      - name: jupyter
        image: jupyter-app:latest
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: jupyter-secrets
              key: database-url
        - name: API_KEY
          valueFrom:
            secretKeyRef:
              name: jupyter-secrets
              key: api-key
```

Network Security:
```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: jupyter-network-policy
spec:
  podSelector:
    matchLabels:
      app: jupyter
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: frontend
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
```

SSL/TLS Configuration:
```python
# SSL configuration for Jupyter
c = get_config()

# SSL certificate configuration
c.ServerApp.certfile = '/path/to/cert.pem'
c.ServerApp.keyfile = '/path/to/key.pem'

# Security headers
c.ServerApp.headers = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
}

# Authentication
c.ServerApp.token = ''
c.ServerApp.password = 'hashed_password'
c.ServerApp.allow_origin = ['https://yourdomain.com']
```
""")

    # Section 9: Performance Optimization
    print_section_header("9. Performance Optimization")
    
    print("""
Performance optimization ensures that deployed Jupyter applications 
run efficiently and can handle expected workloads.

Optimization Areas:
1. Application performance
2. Resource utilization
3. Caching strategies
4. Load balancing
5. Database optimization
6. Network optimization
""")

    # Demonstrate performance optimization
    print_subsection_header("Performance Optimization Examples")
    
    print("""
Application Caching:
```python
import redis
import pickle
import hashlib
from functools import wraps

# Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cache_result(expire_time=3600):
    \"\"\"Cache decorator for function results.\"\"\"
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key_data = f'{func.__name__}:{args}:{kwargs}'
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return pickle.loads(cached_result)
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, expire_time, pickle.dumps(result))
            
            return result
        return wrapper
    return decorator

@cache_result(expire_time=1800)
def expensive_calculation(data):
    \"\"\"Expensive calculation that should be cached.\"\"\"
    # Simulate expensive computation
    import time
    time.sleep(2)
    return {'result': sum(data), 'count': len(data)}
```

Load Balancing Configuration:
```yaml
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream jupyter_backend {
        least_conn;
        server jupyter1:8000;
        server jupyter2:8000;
        server jupyter3:8000;
    }
    
    server {
        listen 80;
        server_name jupyter.example.com;
        
        location / {
            proxy_pass http://jupyter_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

Database Optimization:
```python
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd

class DatabaseManager:
    def __init__(self, connection_string):
        self.connection_string = connection_string
    
    def get_connection(self):
        return psycopg2.connect(self.connection_string)
    
    def execute_query(self, query, params=None):
        \"\"\"Execute query with connection pooling.\"\"\"
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return cur.fetchall()
    
    def batch_insert(self, table_name, data, batch_size=1000):
        \"\"\"Insert data in batches for better performance.\"\"\"
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for i in range(0, len(data), batch_size):
                    batch = data[i:i + batch_size]
                    placeholders = ','.join(['%s'] * len(batch[0]))
                    columns = ','.join(batch[0].keys())
                    
                    query = f\"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})\"
                    cur.executemany(query, [tuple(row.values()) for row in batch])
                
                conn.commit()
    
    def create_indexes(self):
        \"\"\"Create indexes for better query performance.\"\"\"
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_customer_age ON customers(age)',
            'CREATE INDEX IF NOT EXISTS idx_customer_income ON customers(income)',
            'CREATE INDEX IF NOT EXISTS idx_customer_cluster ON customers(cluster)'
        ]
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for index in indexes:
                    cur.execute(index)
                conn.commit()
```

Resource Monitoring:
```python
import psutil
import time
import threading
from collections import deque

class ResourceMonitor:
    def __init__(self, max_history=1000):
        self.max_history = max_history
        self.cpu_history = deque(maxlen=max_history)
        self.memory_history = deque(maxlen=max_history)
        self.disk_history = deque(maxlen=max_history)
        self.running = False
    
    def start_monitoring(self):
        \"\"\"Start resource monitoring in background thread.\"\"\"
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        \"\"\"Stop resource monitoring.\"\"\"
        self.running = False
    
    def _monitor_loop(self):
        \"\"\"Main monitoring loop.\"\"\"
        while self.running:
            timestamp = time.time()
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_history.append((timestamp, cpu_percent))
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_history.append((timestamp, memory.percent))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.disk_history.append((timestamp, disk.percent))
            
            time.sleep(5)  # Monitor every 5 seconds
    
    def get_current_stats(self):
        \"\"\"Get current resource statistics.\"\"\"
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'cpu_history': list(self.cpu_history),
            'memory_history': list(self.memory_history),
            'disk_history': list(self.disk_history)
        }
```
""")

    # Section 10: Summary and Next Steps
    print_section_header("10. Summary and Next Steps")
    
    print("""
Congratulations! You've completed the Jupyter Deployment tutorial. Here's what you've learned:

Key Concepts Covered:
✅ Notebook Conversion: Converting notebooks to executable scripts
✅ Automated Execution: Scheduling and automating notebook runs
✅ Cloud Deployment: Deploying to various cloud platforms
✅ Containerization: Using Docker and Kubernetes
✅ Monitoring and Logging: Tracking application performance
✅ CI/CD Pipelines: Automated testing and deployment
✅ Security Best Practices: Protecting applications and data
✅ Performance Optimization: Optimizing for production workloads
✅ Resource Management: Efficient resource utilization
✅ Production Readiness: Enterprise-grade deployments

Next Steps:

1. Start with Script Conversion: Convert your notebooks to scripts
2. Implement Automated Execution: Set up scheduled runs
3. Choose Cloud Platform: Deploy to your preferred cloud
4. Containerize Applications: Use Docker for consistency
5. Set Up Monitoring: Implement comprehensive monitoring
6. Establish CI/CD: Automate your deployment pipeline
7. Implement Security: Apply security best practices
8. Optimize Performance: Fine-tune for production
9. Scale Applications: Handle increased workloads
10. Maintain and Update: Keep deployments current

Additional Resources:
- Jupyter nbconvert: https://nbconvert.readthedocs.io/
- Papermill: https://papermill.readthedocs.io/
- Docker: https://docs.docker.com/
- Kubernetes: https://kubernetes.io/docs/
- Prometheus: https://prometheus.io/docs/
- Grafana: https://grafana.com/docs/
- GitHub Actions: https://docs.github.com/en/actions
- GitLab CI/CD: https://docs.gitlab.com/ee/ci/

Practice Exercises:
1. Convert a notebook to a Python script using nbconvert
2. Set up automated execution with cron or Airflow
3. Deploy a Jupyter application to a cloud platform
4. Containerize your application with Docker
5. Set up monitoring with Prometheus and Grafana
6. Create a CI/CD pipeline for automated deployment
7. Implement security measures for your deployment
8. Optimize performance for production workloads
9. Scale your application to handle more users
10. Set up disaster recovery and backup procedures

Happy Deploying! 🚀✨
""")

if __name__ == "__main__":
    # Run the tutorial
    main()
    
    print("\n" + "="*60)
    print(" Tutorial completed successfully!")
    print(" Start deploying your Jupyter applications!")
    print("="*60) 