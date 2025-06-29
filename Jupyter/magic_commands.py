#!/usr/bin/env python3
"""
Magic Commands: Enhancing Your Jupyter Workflow

Welcome to the Magic Commands tutorial! Magic commands are special commands in 
Jupyter that provide enhanced functionality beyond regular Python code. They 
start with % (line magic) or %% (cell magic) and can significantly improve 
your productivity and workflow.

This script covers:
- Line and cell magic commands
- Built-in magic functions
- Custom magic commands
- Performance profiling
- System integration
- Advanced magic features

Prerequisites:
- Python 3.8 or higher
- Basic understanding of Jupyter (covered in jupyter_basics.py)
- Familiarity with Python programming
"""

import os
import sys
import time
import subprocess
import cProfile
import pstats
from io import StringIO

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_subsection_header(title):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")

def simulate_magic_command(command, description, output=""):
    """Simulate a magic command execution."""
    print(f"Magic Command: {command}")
    print(f"Description: {description}")
    if output:
        print(f"Output:\n{output}")
    print("-" * 40)

def main():
    """Main function to run all tutorial sections."""
    
    print("Magic Commands Tutorial")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print("Magic commands tutorial started successfully!")

    # Section 1: Introduction to Magic Commands
    print_section_header("1. Introduction to Magic Commands")
    
    print("""
Magic commands are special commands in Jupyter that provide enhanced functionality 
beyond regular Python code. They start with % (line magic) or %% (cell magic) and 
can significantly improve your productivity and workflow.

Key Features:
- Line Magic (%): Operate on single lines
- Cell Magic (%%): Operate on entire cells
- Built-in Functions: System integration, timing, debugging
- Custom Magic: Create your own magic commands
- Performance Profiling: Analyze code performance
- File Operations: Read, write, and manipulate files

Benefits:
✅ Enhanced productivity and workflow
✅ System integration and file operations
✅ Performance analysis and optimization
✅ Debugging and troubleshooting
✅ Custom functionality and automation
✅ Better code organization and documentation
""")

    # Section 2: Line Magic Commands
    print_section_header("2. Line Magic Commands")
    
    print("""
Line magic commands operate on single lines and are prefixed with %.
They provide quick access to common operations and system functions.
""")

    # Demonstrate line magic commands
    print_subsection_header("Common Line Magic Commands")
    
    line_magics = [
        ("%ls", "List files in current directory", "file1.py\nfile2.ipynb\nREADME.md"),
        ("%pwd", "Print working directory", "/home/user/project"),
        ("%cd", "Change directory", "Changed directory to /home/user/project/data"),
        ("%run", "Run a Python script", "Running script.py...\nScript completed successfully"),
        ("%time", "Time execution of a single statement", "CPU times: user 2.34 ms, sys: 1.12 ms, total: 3.46 ms\nWall time: 3.47 ms"),
        ("%timeit", "Time execution with multiple runs", "1.23 µs ± 45.6 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)"),
        ("%who", "Show all variables", "a\tb\tdata\tdf\tx\ty"),
        ("%whos", "Show detailed variable information", "Variable   Type    Data/Info\nx          int     42\ny          float   3.14"),
        ("%reset", "Reset namespace by removing all names", "Once deleted, variables cannot be recovered. Proceed (y/[n])? y\nDeleted all variables."),
        ("%history", "Show command history", "1: import numpy as np\n2: x = np.array([1, 2, 3])\n3: print(x)"),
        ("%bookmark", "Create a bookmark for a directory", "Current directory bookmarked as 'data'"),
        ("%bookmark -l", "List all bookmarks", "Available bookmarks:\ndata -> /home/user/project/data\nsrc -> /home/user/project/src"),
        ("%cd -b data", "Change to bookmarked directory", "Changed directory to /home/user/project/data"),
    ]
    
    for command, description, output in line_magics:
        simulate_magic_command(command, description, output)

    # Section 3: Cell Magic Commands
    print_section_header("3. Cell Magic Commands")
    
    print("""
Cell magic commands operate on entire cells and are prefixed with %%.
They provide powerful functionality for code execution, file operations, 
and system integration.
""")

    # Demonstrate cell magic commands
    print_subsection_header("Common Cell Magic Commands")
    
    cell_magics = [
        ("%%time", "Time execution of entire cell", "CPU times: user 45.2 ms, sys: 12.3 ms, total: 57.5 ms\nWall time: 58.1 ms"),
        ("%%timeit", "Time execution with multiple runs", "45.2 µs ± 2.34 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)"),
        ("%%writefile", "Write cell content to file", "Writing data.csv"),
        ("%%writefile -a", "Append cell content to file", "Appending to data.csv"),
        ("%%capture", "Capture output (suppress display)", "Output captured (not displayed)"),
        ("%%capture --no-stderr", "Capture only stdout", "Only stdout captured"),
        ("%%capture --no-stdout", "Capture only stderr", "Only stderr captured"),
        ("%%bash", "Execute bash commands", "Running bash command...\nCommand output:\nfile1.txt\nfile2.txt"),
        ("%%system", "Execute system commands", "Running system command...\nCommand completed"),
        ("%%javascript", "Execute JavaScript code", "JavaScript executed in browser"),
        ("%%html", "Display HTML content", "HTML content rendered"),
        ("%%latex", "Display LaTeX content", "LaTeX content rendered"),
        ("%%markdown", "Display Markdown content", "Markdown content rendered"),
        ("%%sql", "Execute SQL queries", "SQL query executed\nResults displayed"),
        ("%%cython", "Execute Cython code", "Cython code compiled and executed"),
        ("%%perl", "Execute Perl code", "Perl code executed"),
        ("%%ruby", "Execute Ruby code", "Ruby code executed"),
    ]
    
    for command, description, output in cell_magics:
        simulate_magic_command(command, description, output)

    # Section 4: Performance Profiling Magic
    print_section_header("4. Performance Profiling Magic")
    
    print("""
Jupyter provides powerful magic commands for performance profiling and optimization.
These commands help you identify bottlenecks and optimize your code.
""")

    # Demonstrate profiling magic commands
    print_subsection_header("Profiling Magic Commands")
    
    profiling_magics = [
        ("%prun", "Profile code with cProfile", "Profile results:\n         123 function calls in 0.045 seconds\n\n   Ordered by: internal time\n\n   ncalls  tottime  percall  cumtime  percall filename:lineno(function)\n        1    0.030    0.030    0.045    0.045 <ipython-input-1>:1(<module>)"),
        ("%lprun", "Profile line by line (requires line_profiler)", "Line #      Hits         Time  Per Hit   % Time  Line Contents\n     1                                           def slow_function():\n     2      1000         1234     1.234    45.6     time.sleep(0.001)\n     3      1000         1478     1.478    54.4     return sum(range(100))"),
        ("%mprun", "Profile memory usage (requires memory_profiler)", "Line #    Mem usage    Increment  Line Contents\n     1     45.2 MiB      0.0 MiB   def memory_intensive():\n     2     45.2 MiB      0.0 MiB       data = []\n     3     89.4 MiB     44.2 MiB       for i in range(1000000):\n     4     89.4 MiB      0.0 MiB           data.append(i)"),
        ("%%prun", "Profile entire cell", "Profile results for entire cell execution"),
        ("%%timeit -n 1000 -r 5", "Custom timeit with 1000 loops, 5 runs", "1.23 µs ± 45.6 ns per loop (mean ± std. dev. of 5 runs, 1000 loops each)"),
        ("%%timeit -o", "Return timing object for analysis", "Timing object created for further analysis"),
    ]
    
    for command, description, output in profiling_magics:
        simulate_magic_command(command, description, output)

    # Demonstrate profiling example
    print_subsection_header("Profiling Example")
    
    print("Example: Profiling a slow function")
    print("""
```python
def slow_function():
    result = 0
    for i in range(10000):
        result += i ** 2
    return result

# Profile the function
%prun slow_function()
```

Output:
```
         10003 function calls in 0.045 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.030    0.030    0.045    0.045 <ipython-input-1>:1(slow_function)
    10000    0.015    0.000    0.015    0.000 <built-in method builtins.pow>
        1    0.000    0.000    0.045    0.045 <string>:1(<module>)
        1    0.000    0.000    0.045    0.045 {built-in method builtins.exec}
```
""")

    # Section 5: System Integration Magic
    print_section_header("5. System Integration Magic")
    
    print("""
Magic commands provide powerful system integration capabilities, allowing you 
to interact with the operating system, execute shell commands, and manage files.
""")

    # Demonstrate system integration
    print_subsection_header("System Integration Examples")
    
    system_examples = [
        ("%ls -la", "List files with details", "total 1234\ndrwxr-xr-x  2 user group  4096 Jan 15 10:30 .\ndrwxr-xr-x 10 user group  4096 Jan 15 10:30 ..\n-rw-r--r--  1 user group   123 Jan 15 10:30 file1.txt"),
        ("%pwd", "Show current directory", "/home/user/project"),
        ("%cd ../data", "Change to data directory", "Changed directory to /home/user/project/data"),
        ("%run script.py", "Execute Python script", "Running script.py...\nScript completed successfully"),
        ("%%bash", "Execute bash commands in cell", "#!/bin/bash\necho 'Current directory:'\npwd\necho 'Files:'\nls -la"),
        ("%%system", "Execute system commands", "echo 'System command executed'\ndate\nwhoami"),
        ("%env", "Show environment variables", "PATH=/usr/local/bin:/usr/bin:/bin\nPYTHONPATH=/home/user/project\nHOME=/home/user"),
        ("%env VAR=value", "Set environment variable", "Environment variable VAR set to value"),
        ("%pip install package", "Install Python package", "Collecting package\nInstalling collected packages: package\nSuccessfully installed package-1.0.0"),
        ("%conda install package", "Install conda package", "Collecting package metadata\nInstalling package...\nDone"),
    ]
    
    for command, description, output in system_examples:
        simulate_magic_command(command, description, output)

    # Section 6: File Operations Magic
    print_section_header("6. File Operations Magic")
    
    print("""
Magic commands provide convenient ways to read, write, and manipulate files 
directly from Jupyter notebooks.
""")

    # Demonstrate file operations
    print_subsection_header("File Operation Examples")
    
    file_examples = [
        ("%%writefile data.txt", "Write cell content to file", "Writing data.txt"),
        ("%%writefile -a data.txt", "Append to file", "Appending to data.txt"),
        ("%cat data.txt", "Display file contents", "This is the content of data.txt\nLine 2\nLine 3"),
        ("%head -5 data.txt", "Show first 5 lines", "This is the content of data.txt\nLine 2\nLine 3\nLine 4\nLine 5"),
        ("%tail -3 data.txt", "Show last 3 lines", "Line 8\nLine 9\nLine 10"),
        ("%less data.txt", "View file with pager", "File displayed in pager (interactive)"),
        ("%more data.txt", "View file with more", "File displayed with more (interactive)"),
        ("%wc -l data.txt", "Count lines in file", "10 data.txt"),
        ("%grep 'pattern' data.txt", "Search for pattern", "Line containing pattern"),
        ("%sed 's/old/new/g' data.txt", "Replace text", "Text replaced in file"),
    ]
    
    for command, description, output in file_examples:
        simulate_magic_command(command, description, output)

    # Demonstrate file writing example
    print_subsection_header("File Writing Example")
    
    print("Example: Writing data to CSV file")
    print("""
```python
%%writefile data.csv
name,age,city
Alice,25,New York
Bob,30,San Francisco
Charlie,35,Chicago
```

Output:
```
Writing data.csv
```

```python
%cat data.csv
```

Output:
```
name,age,city
Alice,25,New York
Bob,30,San Francisco
Charlie,35,Chicago
```
""")

    # Section 7: Custom Magic Commands
    print_section_header("7. Custom Magic Commands")
    
    print("""
You can create your own magic commands to automate common tasks and enhance 
your workflow. Custom magic commands are defined using Python functions.
""")

    # Demonstrate custom magic creation
    print_subsection_header("Creating Custom Magic Commands")
    
    print("Example: Creating a custom magic command")
    print("""
```python
from IPython.core.magic import register_line_magic, register_cell_magic

@register_line_magic
def hello(line):
    \"\"\"Say hello to someone.\"\"\"
    name = line.strip() or 'World'
    return f'Hello, {name}!'

@register_cell_magic
def repeat(line, cell):
    \"\"\"Repeat cell content multiple times.\"\"\"
    try:
        count = int(line.strip())
    except ValueError:
        count = 1
    
    result = []
    for i in range(count):
        result.append(f'[{i+1}] {cell.strip()}')
    
    return '\\n'.join(result)
```

Usage:
```python
%hello Alice
# Output: Hello, Alice!

%%repeat 3
This will be repeated
# Output:
# [1] This will be repeated
# [2] This will be repeated
# [3] This will be repeated
```
""")

    # Section 8: Advanced Magic Features
    print_section_header("8. Advanced Magic Features")
    
    print("""
Magic commands support advanced features for complex workflows and automation.
""")

    # Demonstrate advanced features
    print_subsection_header("Advanced Magic Features")
    
    advanced_features = [
        ("%automagic", "Enable automatic magic (no % prefix needed)", "Automagic is ON, % prefix NOT needed for magic commands"),
        ("%automagic 0", "Disable automatic magic", "Automagic is OFF, % prefix needed for magic commands"),
        ("%magic", "Show information about magic commands", "Available magic commands:\n%alias, %autocall, %automagic, %bookmark..."),
        ("%lsmagic", "List all available magic commands", "Available line magics:\n%alias  %autocall  %automagic  %bookmark...\n\nAvailable cell magics:\n%%!  %%HTML  %%SVG  %%bash..."),
        ("%alias", "Create alias for magic command", "Alias 'll' created for 'ls -la'"),
        ("%unalias", "Remove alias", "Alias 'll' removed"),
        ("%macro", "Create macro from history", "Macro 'my_macro' created"),
        ("%store", "Store variable for later use", "Variable 'data' stored"),
        ("%store -r", "Restore stored variables", "Variables restored"),
        ("%store -d", "Delete stored variable", "Variable 'data' deleted"),
    ]
    
    for command, description, output in advanced_features:
        simulate_magic_command(command, description, output)

    # Section 9: Magic Command Best Practices
    print_section_header("9. Magic Command Best Practices")
    
    print("""
Follow these best practices when using magic commands:

1. Use Appropriate Magic Commands:
   - Line magic for single operations
   - Cell magic for multi-line operations
   - System magic for OS integration
   - Profiling magic for performance analysis

2. Performance Considerations:
   - Use %timeit for accurate timing
   - Use %prun for detailed profiling
   - Avoid magic commands in production code
   - Profile before optimizing

3. File Operations:
   - Use %%writefile for creating files
   - Use %cat for viewing small files
   - Use %less for large files
   - Be careful with file paths

4. System Integration:
   - Use %run for Python scripts
   - Use %%bash for shell commands
   - Use %pip for package installation
   - Check command output carefully

5. Custom Magic:
   - Keep custom magic simple
   - Document your magic commands
   - Test thoroughly before using
   - Share useful magic with team
""")

    # Section 10: Troubleshooting Magic Commands
    print_section_header("10. Troubleshooting Magic Commands")
    
    print("""
Common issues with magic commands and how to resolve them:

1. Magic Command Not Found:
   - Check spelling and syntax
   - Use %lsmagic to see available commands
   - Install required packages
   - Check IPython version

2. Permission Issues:
   - Check file permissions
   - Use sudo for system commands
   - Verify working directory
   - Check user privileges

3. Import Errors:
   - Install missing packages
   - Check Python environment
   - Verify import paths
   - Restart kernel if needed

4. Performance Issues:
   - Use appropriate profiling tools
   - Check system resources
   - Optimize code before profiling
   - Use sampling for large datasets

5. Output Problems:
   - Check command syntax
   - Verify file paths
   - Use verbose options
   - Check error messages
""")

    # Section 11: Magic Commands in Production
    print_section_header("11. Magic Commands in Production")
    
    print("""
Using magic commands in production environments requires careful consideration:

1. When to Use Magic Commands:
   - Development and testing
   - Interactive data analysis
   - Prototyping and exploration
   - Documentation and tutorials

2. When to Avoid Magic Commands:
   - Production scripts
   - Automated workflows
   - CI/CD pipelines
   - Performance-critical code

3. Converting Magic to Python:
   - Replace %run with subprocess
   - Replace %timeit with timeit module
   - Replace file magic with file operations
   - Replace system magic with os module

4. Best Practices:
   - Document magic usage
   - Provide Python alternatives
   - Test both versions
   - Consider maintainability
""")

    # Demonstrate conversion example
    print_subsection_header("Magic to Python Conversion Example")
    
    print("""
Magic Command:
```python
%timeit sum(range(1000))
```

Python Equivalent:
```python
import timeit
result = timeit.timeit('sum(range(1000))', number=1000)
print(f'Average time: {result/1000:.6f} seconds')
```

Magic Command:
```python
%%writefile data.txt
Hello, World!
```

Python Equivalent:
```python
with open('data.txt', 'w') as f:
    f.write('Hello, World!')
```
""")

    # Section 12: Summary and Next Steps
    print_section_header("12. Summary and Next Steps")
    
    print("""
Congratulations! You've completed the Magic Commands tutorial. Here's what you've learned:

Key Concepts Covered:
✅ Line Magic Commands: Single-line operations with %
✅ Cell Magic Commands: Multi-line operations with %%
✅ Performance Profiling: Timing and profiling tools
✅ System Integration: OS and shell command integration
✅ File Operations: Reading, writing, and manipulating files
✅ Custom Magic Commands: Creating your own magic functions
✅ Advanced Features: Automation and workflow enhancement
✅ Best Practices: Effective magic command usage
✅ Troubleshooting: Common issues and solutions
✅ Production Usage: When and how to use magic commands

Next Steps:

1. Practice with Built-in Magic: Explore all available commands
2. Create Custom Magic: Automate your common tasks
3. Master Profiling: Optimize your code performance
4. Integrate with System: Use magic for system operations
5. Follow Best Practices: Use magic commands effectively
6. Explore Advanced Features: Leverage advanced capabilities

Additional Resources:
- IPython Magic Commands: https://ipython.readthedocs.io/en/stable/interactive/magics.html
- Jupyter Magic Commands: https://jupyter.org/
- Custom Magic Development: https://ipython.readthedocs.io/en/stable/config/custommagics.html
- Performance Profiling: https://docs.python.org/3/library/profile.html

Practice Exercises:
1. Create a custom magic command for data loading
2. Profile a slow function and optimize it
3. Use magic commands to automate file operations
4. Create a workflow using multiple magic commands
5. Convert magic commands to Python code
6. Build a custom magic command library

Happy Magic Command-ing! ✨
""")

if __name__ == "__main__":
    # Run the tutorial
    main()
    
    print("\n" + "="*60)
    print(" Tutorial completed successfully!")
    print(" Try running magic commands in Jupyter!")
    print("="*60) 