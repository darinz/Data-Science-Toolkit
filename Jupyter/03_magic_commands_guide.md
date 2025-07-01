# Jupyter Magic Commands: Complete Guide

## Table of Contents
1. [Introduction to Magic Commands](#introduction-to-magic-commands)
2. [Line Magic Commands](#line-magic-commands)
3. [Cell Magic Commands](#cell-magic-commands)
4. [Built-in Magic Commands](#built-in-magic-commands)
5. [Custom Magic Commands](#custom-magic-commands)
6. [Performance Magic Commands](#performance-magic-commands)
7. [System Integration](#system-integration)
8. [Advanced Magic Features](#advanced-magic-features)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Introduction to Magic Commands

Magic commands are special commands in Jupyter that provide enhanced functionality beyond regular Python code. They start with `%` (line magic) or `%%` (cell magic) and are processed by IPython before execution.

### Types of Magic Commands

1. **Line Magic**: Start with `%` and operate on a single line
2. **Cell Magic**: Start with `%%` and operate on entire cells
3. **Built-in Magic**: Come with IPython/Jupyter
4. **Custom Magic**: User-defined magic commands

### Magic Command Syntax

```python
# Line magic (single line)
%magic_name argument1 argument2

# Cell magic (entire cell)
%%magic_name argument1 argument2
cell content here
multiple lines
```

### Getting Help with Magic Commands

```python
# List all magic commands
%lsmagic

# Get help for specific magic
%magic_name?

# Get detailed help
%magic_name??

# Search for magic commands
%magic_name*
```

## Line Magic Commands

### Basic Information Commands

```python
# Get current working directory
%pwd

# Change directory
%cd /path/to/directory

# List files in current directory
%ls

# List files with details
%ls -la

# Get system information
%system_info

# Get Python version
%python_version

# Get IPython version
%ipython_version
```

### Variable and Object Commands

```python
# Display detailed information about object
%pinfo object_name

# Display source code of function/class
%psource function_name

# Display docstring
%pdoc object_name

# Display file where object is defined
%pfile object_name

# Get type information
%pinfo2 object_name

# Display object attributes
%pdef function_name
```

### History Commands

```python
# Display command history
%history

# Display history with line numbers
%history -n

# Display history with timestamps
%history -t

# Display history for specific session
%history -g pattern

# Display last N commands
%history -l 10

# Save history to file
%history -f history.txt
```

### Bookmark Commands

```python
# Create bookmark
%bookmark bookmark_name /path/to/directory

# List bookmarks
%bookmark -l

# Go to bookmark
%cd -b bookmark_name

# Delete bookmark
%bookmark -d bookmark_name

# Delete all bookmarks
%bookmark -r
```

### Alias Commands

```python
# Create alias
%alias alias_name command

# Create alias with arguments
%alias ll ls -la

# List aliases
%alias

# Delete alias
%unalias alias_name

# Example: Create git status alias
%alias gs git status
%gs  # Now runs git status
```

## Cell Magic Commands

### Code Execution Magic

```python
# Execute code in different language
%%python
print("This is Python code")

%%bash
echo "This is bash code"
ls -la

%%javascript
console.log("This is JavaScript code");

%%html
<h1>This is HTML</h1>
<p>This is a paragraph</p>

%%latex
\begin{equation}
E = mc^2
\end{equation}
```

### File Operations Magic

```python
# Write cell content to file
%%writefile filename.py
def hello_world():
    print("Hello, World!")

hello_world()

# Append to file
%%writefile -a filename.py
# Additional code here

# Read file content
%%readfile filename.py

# Load file content into variable
%%capture file_content
%%readfile filename.py
```

### Performance Magic

```python
# Time execution
%%time
import numpy as np
x = np.random.randn(1000000)
y = np.random.randn(1000000)
z = x + y

# Time execution with more details
%%timeit
import numpy as np
x = np.random.randn(1000)
y = np.random.randn(1000)
z = x + y

# Profile code
%%prun
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

fibonacci(30)
```

### Debugging Magic

```python
# Debug code
%%debug
def divide(a, b):
    return a / b

result = divide(10, 0)

# Run with pdb
%%pdb
def risky_function():
    x = 1
    y = 0
    return x / y

risky_function()
```

## Built-in Magic Commands

### System Commands

```python
# Run system command
%system command

# Run system command and capture output
%system command

# Example system commands
%system ls -la
%system pwd
%system echo $PATH
%system python --version
%system pip list
```

### Environment Commands

```python
# Set environment variable
%env VARIABLE_NAME=value

# Get environment variable
%env VARIABLE_NAME

# List all environment variables
%env

# Example: Set Python path
%env PYTHONPATH=/path/to/modules
```

### Configuration Commands

```python
# Configure IPython
%config Class.attribute=value

# List configuration
%config

# Example: Configure matplotlib
%config InlineBackend.figure_format='retina'
%config InlineBackend.rc={'figure.figsize': (10, 6)}
```

### Extension Commands

```python
# Load extension
%load_ext extension_name

# Reload extension
%reload_ext extension_name

# List loaded extensions
%extensions

# Example: Load line profiler
%load_ext line_profiler
%lprun -f function_name function_call()
```

## Custom Magic Commands

### Creating Simple Magic Commands

```python
# Define line magic
from IPython.core.magic import register_line_magic

@register_line_magic
def hello(line):
    """Simple hello magic command."""
    return f"Hello, {line}!"

# Usage
%hello World

# Define cell magic
from IPython.core.magic import register_cell_magic

@register_cell_magic
def repeat(line, cell):
    """Repeat cell content N times."""
    try:
        n = int(line)
        return cell * n
    except ValueError:
        return f"Error: '{line}' is not a valid number"

# Usage
%%repeat 3
print("Hello, World!")
```

### Advanced Custom Magic

```python
from IPython.core.magic import register_line_magic, register_cell_magic
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

@register_line_magic
@magic_arguments(
    argument('name', help='Name to greet'),
    argument('-u', '--uppercase', action='store_true', help='Convert to uppercase')
)
def greet(line):
    """Greet someone with optional uppercase."""
    args = parse_argstring(greet, line)
    message = f"Hello, {args.name}!"
    if args.uppercase:
        message = message.upper()
    return message

# Usage
%greet Alice
%greet -u Bob
```

### Magic with Arguments

```python
from IPython.core.magic import register_cell_magic
import json

@register_cell_magic
def json_format(line, cell):
    """Format JSON content."""
    try:
        data = json.loads(cell)
        return json.dumps(data, indent=2)
    except json.JSONDecodeError as e:
        return f"Error: {e}"

# Usage
%%json_format
{"name": "John", "age": 30, "city": "New York"}
```

### Magic with File Operations

```python
from IPython.core.magic import register_cell_magic
import os

@register_cell_magic
def save_to_file(line, cell):
    """Save cell content to file."""
    if not line.strip():
        return "Error: Please provide a filename"
    
    filename = line.strip()
    try:
        with open(filename, 'w') as f:
            f.write(cell)
        return f"Content saved to {filename}"
    except Exception as e:
        return f"Error saving file: {e}"

# Usage
%%save_to_file my_script.py
def hello():
    print("Hello, World!")

hello()
```

## Performance Magic Commands

### Timing and Profiling

```python
# Basic timing
%time expression

# Detailed timing
%timeit expression

# Time with setup
%timeit -s "import numpy as np" "np.random.randn(1000)"

# Profile function
%prun function_call()

# Line-by-line profiling
%load_ext line_profiler
%lprun -f function_name function_call()

# Memory profiling
%load_ext memory_profiler
%mprun -f function_name function_call()
```

### Performance Examples

```python
# Compare different approaches
import numpy as np

# Method 1: List comprehension
%timeit [x**2 for x in range(1000)]

# Method 2: NumPy
%timeit np.arange(1000)**2

# Method 3: Map function
%timeit list(map(lambda x: x**2, range(1000)))

# Profile complex function
def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

%prun fibonacci_recursive(30)
```

### Memory Profiling

```python
# Memory usage tracking
%load_ext memory_profiler

def create_large_list(n):
    return list(range(n))

%mprun -f create_large_list create_large_list(1000000)

# Monitor memory usage
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

print(f"Current memory usage: {get_memory_usage():.2f} MB")
```

## System Integration

### Shell Commands

```python
# Run shell commands
!ls -la
!pwd
!echo $PATH

# Capture output
output = !ls -la
print(output)

# Use variables in shell commands
filename = "data.csv"
!head -n 5 {filename}

# Multiple commands
!cd /tmp && ls -la
```

### File System Operations

```python
# File operations
!touch new_file.txt
!echo "Hello, World!" > new_file.txt
!cat new_file.txt
!rm new_file.txt

# Directory operations
!mkdir new_directory
!ls -la new_directory
!rmdir new_directory

# Copy and move files
!cp source.txt destination.txt
!mv old_name.txt new_name.txt
```

### Package Management

```python
# Install packages
!pip install package_name
!conda install package_name

# List installed packages
!pip list
!conda list

# Upgrade packages
!pip install --upgrade package_name
!conda update package_name

# Check package versions
!pip show package_name
!conda list package_name
```

### Git Integration

```python
# Git commands
!git status
!git add .
!git commit -m "Update notebook"
!git push

# Check git log
!git log --oneline -5

# Check branches
!git branch -a

# Create and switch branches
!git checkout -b new_branch
!git checkout main
```

## Advanced Magic Features

### Magic with External Tools

```python
# Docker integration
!docker ps
!docker run -it python:3.9 python -c "print('Hello from Docker')"

# Database connections
import sqlite3
%sql sqlite:///database.db
%sql SELECT * FROM table_name LIMIT 5

# API calls
import requests
response = requests.get('https://api.github.com/users/octocat')
print(response.json())
```

### Magic with Visualization

```python
# Matplotlib configuration
%matplotlib inline
%config InlineBackend.figure_format='retina'

# Plotly integration
%load_ext plotly_magic
%%plotly
import plotly.express as px
fig = px.scatter(x=[1, 2, 3], y=[4, 5, 6])
fig.show()

# Bokeh integration
%load_ext bokeh_magic
%%bokeh
from bokeh.plotting import figure, show
p = figure()
p.circle([1, 2, 3], [4, 5, 6])
show(p)
```

### Magic with Data Science Tools

```python
# Pandas magic
%load_ext pandas_magic
%%pandas
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df.head()

# R magic
%load_ext rpy2.ipython
%%R
data <- data.frame(x = 1:10, y = rnorm(10))
plot(data$x, data$y)

# Julia magic
%load_ext julia.magic
%%julia
using Plots
plot(1:10, rand(10))
```

## Best Practices

### Magic Command Organization

```python
# Group related magic commands
# Setup and configuration
%matplotlib inline
%config InlineBackend.figure_format='retina'
%load_ext line_profiler

# Environment setup
%env PYTHONPATH=/path/to/modules
%env DISPLAY=:0

# Aliases for common commands
%alias ll ls -la
%alias gs git status
%alias gp git push
```

### Performance Optimization

```python
# Use appropriate timing commands
# For quick timing
%time expression

# For accurate timing
%timeit expression

# For profiling
%prun function_call()

# For line-by-line profiling
%lprun -f function_name function_call()

# For memory profiling
%mprun -f function_name function_call()
```

### Error Handling

```python
# Handle magic command errors
try:
    %magic_command
except Exception as e:
    print(f"Magic command failed: {e}")

# Check if magic is available
if 'magic_name' in get_ipython().magics_manager.magics['line']:
    %magic_name
else:
    print("Magic command not available")
```

### Documentation

```python
# Document custom magic commands
@register_line_magic
def my_magic(line):
    """
    My custom magic command.
    
    Parameters:
    -----------
    line : str
        Command line arguments
        
    Returns:
    --------
    str
        Result of the magic command
        
    Examples:
    ---------
    %my_magic argument
    """
    return f"Processed: {line}"
```

## Troubleshooting

### Common Issues

#### 1. Magic Command Not Found

```python
# Check available magic commands
%lsmagic

# Check if extension is loaded
%extensions

# Load missing extension
%load_ext extension_name

# Check IPython version
%ipython_version
```

#### 2. Performance Issues

```python
# Check if profiling extensions are available
%load_ext line_profiler
%load_ext memory_profiler

# Use appropriate profiling tools
%timeit expression  # For timing
%prun function_call()  # For profiling
%lprun -f function_name function_call()  # For line profiling
```

#### 3. System Command Issues

```python
# Check system command availability
!which command_name

# Use full path
!/usr/bin/command_name

# Check permissions
!ls -la /path/to/command

# Use shell=True for complex commands
import subprocess
subprocess.run("complex command", shell=True)
```

### Getting Help

```python
# Built-in help
%magic_name?

# Detailed help
%magic_name??

# Search for magic commands
%magic_name*

# IPython help
%help

# System help
!command_name --help
```

### Debugging Magic Commands

```python
# Debug custom magic commands
import traceback

@register_line_magic
def debug_magic(line):
    """Debug magic command."""
    try:
        # Your magic command logic here
        result = process_line(line)
        return result
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None
```

## Conclusion

Magic commands are powerful tools that enhance Jupyter's functionality. Key takeaways:

- **Line Magic**: Use `%` for single-line operations
- **Cell Magic**: Use `%%` for multi-line operations
- **Built-in Magic**: Leverage pre-built functionality
- **Custom Magic**: Create your own commands
- **Performance Magic**: Profile and optimize code
- **System Integration**: Connect with external tools

### Next Steps
- Explore more built-in magic commands
- Create custom magic for your workflow
- Integrate with external tools and APIs
- Contribute to the magic command ecosystem

### Resources
- [IPython Magic Commands Documentation](https://ipython.readthedocs.io/en/stable/interactive/magics.html)
- [Jupyter Magic Commands](https://jupyter.readthedocs.io/en/latest/development/magics.html)
- [Custom Magic Commands Guide](https://ipython.readthedocs.io/en/stable/config/custommagics.html)
- [IPython GitHub](https://github.com/ipython/ipython)

---

**Happy Magic Commanding!** ✨ 