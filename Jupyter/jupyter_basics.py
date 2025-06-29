#!/usr/bin/env python3
"""
Jupyter Basics: A Comprehensive Introduction

Welcome to the Jupyter basics tutorial! Jupyter is an open-source web application 
that allows you to create and share documents that contain live code, equations, 
visualizations, and narrative text.

This script covers:
- Introduction to Jupyter notebooks
- Cell types and execution modes
- Basic notebook operations
- Keyboard shortcuts and tips
- Working with kernels
- Markdown and documentation

Prerequisites:
- Python 3.8 or higher
- Basic understanding of Python programming
- Familiarity with command line operations
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

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
    
    print("Jupyter Basics Tutorial")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print("Jupyter tutorial started successfully!")

    # Section 1: Introduction to Jupyter
    print_section_header("1. Introduction to Jupyter")
    
    print("""
Jupyter is an open-source web application that allows you to create and share 
documents that contain live code, equations, visualizations, and narrative text.

Key Components:
- Jupyter Notebook: The classic web-based interface
- JupyterLab: The next-generation web-based interface
- Kernels: The computational engines that execute code
- Extensions: Additional functionality and customization

Benefits:
✅ Interactive computing and immediate feedback
✅ Rich media support (images, videos, interactive plots)
✅ Markdown documentation alongside code
✅ Reproducible research and analysis
✅ Easy sharing and collaboration
✅ Support for multiple programming languages
""")

    # Section 2: Jupyter Architecture
    print_section_header("2. Jupyter Architecture")
    
    print("""
Jupyter follows a client-server architecture:

1. Jupyter Server: Runs the web application and manages kernels
2. Web Browser: Provides the user interface
3. Kernels: Execute code in different programming languages
4. Notebook Files: Store code, output, and documentation

Notebook Structure:
- Cells: Individual code or text blocks
- Output: Results from code execution
- Metadata: Information about the notebook
- Kernel: The computational engine
""")

    # Section 3: Cell Types
    print_section_header("3. Cell Types")
    
    print("""
Jupyter notebooks contain different types of cells:

1. Code Cells:
   - Execute Python (or other language) code
   - Display output below the cell
   - Can contain multiple statements
   - Support for rich output (plots, tables, etc.)

2. Markdown Cells:
   - Write formatted text using Markdown
   - Support for headers, lists, links, images
   - Mathematical equations with LaTeX
   - HTML and CSS for advanced formatting

3. Raw Cells:
   - Unprocessed text that passes through unchanged
   - Useful for custom output formats
   - Not commonly used in most workflows
""")

    # Demonstrate code cell simulation
    print_subsection_header("Code Cell Example")
    
    # Simulate code execution
    print("Code Cell:")
    print("```python")
    print("import numpy as np")
    print("import matplotlib.pyplot as plt")
    print("")
    print("# Generate sample data")
    print("x = np.linspace(0, 10, 100)")
    print("y = np.sin(x)")
    print("")
    print("# Create a plot")
    print("plt.figure(figsize=(8, 6))")
    print("plt.plot(x, y, 'b-', linewidth=2)")
    print("plt.title('Sine Wave')")
    print("plt.xlabel('x')")
    print("plt.ylabel('sin(x)')")
    print("plt.grid(True, alpha=0.3)")
    print("plt.show()")
    print("```")
    
    print("\nOutput:")
    print("(A plot window would appear here)")
    print("Figure saved as 'sine_wave.png'")

    # Demonstrate markdown cell
    print_subsection_header("Markdown Cell Example")
    
    print("Markdown Cell:")
    print("```markdown")
    print("# Data Analysis Report")
    print("")
    print("## Introduction")
    print("This notebook demonstrates basic data analysis techniques.")
    print("")
    print("### Key Findings:")
    print("- The sine wave has a period of 2π")
    print("- Maximum amplitude is 1.0")
    print("- The function is continuous and smooth")
    print("")
    print("Mathematical expression: $f(x) = \\sin(x)$")
    print("")
    print("[View full documentation](https://jupyter.org/)")
    print("```")
    
    print("\nRendered Output:")
    print("=" * 50)
    print("# Data Analysis Report")
    print("")
    print("## Introduction")
    print("This notebook demonstrates basic data analysis techniques.")
    print("")
    print("### Key Findings:")
    print("- The sine wave has a period of 2π")
    print("- Maximum amplitude is 1.0")
    print("- The function is continuous and smooth")
    print("")
    print("Mathematical expression: f(x) = sin(x)")
    print("")
    print("View full documentation")

    # Section 4: Execution Modes
    print_section_header("4. Execution Modes")
    
    print("""
Jupyter notebooks support different execution modes:

1. Interactive Mode:
   - Execute cells one at a time
   - See immediate results
   - Modify and re-run as needed
   - Perfect for exploration and development

2. Batch Mode:
   - Run entire notebook from start to finish
   - Useful for automated execution
   - Can be scheduled or triggered programmatically
   - Good for reproducible workflows

3. Kernel Modes:
   - Restart: Clear all variables and start fresh
   - Restart & Clear Output: Remove all output
   - Restart & Run All: Execute all cells in order
   - Interrupt: Stop long-running computations
""")

    # Section 5: Basic Operations
    print_section_header("5. Basic Operations")
    
    print("""
Essential Jupyter operations:

1. Creating and Opening Notebooks:
   - New notebook from Jupyter interface
   - Open existing .ipynb files
   - Import from other formats (Python scripts, etc.)

2. Cell Operations:
   - Insert new cells (above/below)
   - Delete cells
   - Copy and paste cells
   - Move cells up/down
   - Merge and split cells

3. Execution:
   - Run current cell (Shift + Enter)
   - Run current cell and select below (Alt + Enter)
   - Run all cells above current
   - Run all cells in notebook

4. Saving and Exporting:
   - Auto-save functionality
   - Manual save (Ctrl/Cmd + S)
   - Export to various formats (HTML, PDF, Python script)
""")

    # Section 6: Keyboard Shortcuts
    print_section_header("6. Keyboard Shortcuts")
    
    print("""
Essential Jupyter keyboard shortcuts:

Command Mode (press Esc to enter):
- A: Insert cell above
- B: Insert cell below
- D, D: Delete cell (press D twice)
- Z: Undo cell deletion
- Shift + V: Paste cell above
- V: Paste cell below
- X: Cut cell
- C: Copy cell
- Shift + M: Merge cells
- Shift + -: Split cell

Edit Mode (press Enter to enter):
- Shift + Enter: Run cell, select below
- Alt + Enter: Run cell, insert below
- Ctrl + Enter: Run cell, stay on cell
- Ctrl + Shift + -: Split cell at cursor
- Ctrl + A: Select all
- Ctrl + Z: Undo
- Ctrl + Y: Redo

General:
- Ctrl + S: Save notebook
- Ctrl + Shift + P: Command palette
- H: Show all shortcuts
- 0, 0: Restart kernel (press 0 twice)
- 0, 0, 0: Restart kernel and clear output
""")

    # Section 7: Working with Kernels
    print_section_header("7. Working with Kernels")
    
    print("""
Kernels are the computational engines that execute code:

1. Kernel Types:
   - Python (ipykernel): Most common
   - R (IRkernel): For R programming
   - Julia (IJulia): For Julia programming
   - JavaScript (ijavascript): For Node.js
   - Custom kernels: For other languages

2. Kernel Operations:
   - Start kernel: Initialize computational environment
   - Restart kernel: Clear memory and restart
   - Interrupt kernel: Stop execution
   - Shutdown kernel: Close kernel completely
   - Change kernel: Switch to different language/environment

3. Kernel Management:
   - Multiple kernels can run simultaneously
   - Each notebook connects to one kernel
   - Kernels can be shared between notebooks
   - Resource monitoring and cleanup
""")

    # Demonstrate kernel information
    print_subsection_header("Kernel Information")
    
    print("Current Python Environment:")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check for common data science packages
    packages_to_check = ['numpy', 'pandas', 'matplotlib', 'scipy', 'scikit-learn']
    print("\nInstalled packages:")
    for package in packages_to_check:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"  ✓ {package}: {version}")
        except ImportError:
            print(f"  ✗ {package}: not installed")

    # Section 8: Markdown and Documentation
    print_section_header("8. Markdown and Documentation")
    
    print("""
Markdown is essential for creating well-documented notebooks:

1. Basic Markdown:
   - Headers: #, ##, ###
   - Emphasis: *italic*, **bold**
   - Lists: bullet points and numbered lists
   - Links: [text](url)
   - Images: ![alt text](image_url)

2. Advanced Markdown:
   - Code blocks: ```python
   - Inline code: `variable_name`
   - Tables: | Column 1 | Column 2 |
   - Blockquotes: > quoted text
   - Horizontal rules: ---

3. Mathematical Expressions:
   - Inline math: $E = mc^2$
   - Display math: $$\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}$$
   - LaTeX support for complex equations

4. HTML and CSS:
   - Custom styling and formatting
   - Embedded videos and interactive content
   - Custom HTML elements
""")

    # Demonstrate markdown examples
    print_subsection_header("Markdown Examples")
    
    markdown_examples = [
        ("Headers", "# Main Title\n## Section\n### Subsection"),
        ("Emphasis", "This is *italic* and this is **bold**"),
        ("Lists", "1. First item\n2. Second item\n   - Subitem\n   - Another subitem"),
        ("Code", "Use `print('Hello World')` for output"),
        ("Links", "[Jupyter Documentation](https://jupyter.org/)"),
        ("Math", "The quadratic formula: $x = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$")
    ]
    
    for title, example in markdown_examples:
        print(f"\n{title}:")
        print(f"```markdown")
        print(example)
        print("```")

    # Section 9: Best Practices
    print_section_header("9. Best Practices")
    
    print("""
Follow these best practices for effective Jupyter usage:

1. Notebook Organization:
   - Clear, descriptive titles
   - Logical cell ordering
   - Consistent formatting
   - Regular saves and backups

2. Code Quality:
   - Write clean, readable code
   - Use meaningful variable names
   - Add comments and documentation
   - Follow PEP 8 style guidelines

3. Documentation:
   - Explain your approach and methodology
   - Document assumptions and limitations
   - Include references and citations
   - Use markdown for narrative text

4. Performance:
   - Avoid long-running cells
   - Use appropriate data structures
   - Profile code when needed
   - Consider using %%timeit magic

5. Collaboration:
   - Use version control (Git)
   - Share notebooks with clear instructions
   - Include requirements and dependencies
   - Document environment setup
""")

    # Section 10: Common Issues and Solutions
    print_section_header("10. Common Issues and Solutions")
    
    print("""
Common Jupyter issues and how to resolve them:

1. Kernel Issues:
   - Kernel won't start: Check Python installation and dependencies
   - Kernel dies unexpectedly: Check for memory issues or infinite loops
   - Wrong kernel: Select correct kernel from kernel menu

2. Import Errors:
   - Package not found: Install missing packages
   - Version conflicts: Use virtual environments
   - Path issues: Check PYTHONPATH and working directory

3. Display Issues:
   - Plots not showing: Use plt.show() or %matplotlib inline
   - Output too large: Use display options or save to file
   - Formatting problems: Check markdown syntax

4. Performance Issues:
   - Slow execution: Profile code and optimize
   - Memory problems: Clear variables or restart kernel
   - Large datasets: Use appropriate data structures

5. Saving Issues:
   - Auto-save not working: Check file permissions
   - Export problems: Install required packages (pandoc, etc.)
   - Version conflicts: Use consistent Jupyter versions
""")

    # Section 11: Getting Started Commands
    print_section_header("11. Getting Started Commands")
    
    print("""
Essential commands to get started with Jupyter:

1. Installation:
   ```bash
   # Install Jupyter
   pip install jupyter
   
   # Install JupyterLab (recommended)
   pip install jupyterlab
   
   # Install additional packages
   pip install numpy pandas matplotlib
   ```

2. Starting Jupyter:
   ```bash
   # Start Jupyter Notebook
   jupyter notebook
   
   # Start JupyterLab
   jupyter lab
   
   # Start with specific port
   jupyter lab --port=8888
   
   # Start with no browser
   jupyter lab --no-browser
   ```

3. Creating Notebooks:
   ```bash
   # Create new notebook
   jupyter notebook --generate-config
   
   # List running servers
   jupyter notebook list
   
   # Stop all servers
   jupyter notebook stop
   ```

4. Package Management:
   ```bash
   # Install kernel for current environment
   python -m ipykernel install --user --name=myenv
   
   # List available kernels
   jupyter kernelspec list
   
   # Remove kernel
   jupyter kernelspec remove myenv
   ```
""")

    # Section 12: Summary and Next Steps
    print_section_header("12. Summary and Next Steps")
    
    print("""
Congratulations! You've completed the Jupyter basics tutorial. Here's what you've learned:

Key Concepts Covered:
✅ Jupyter Architecture: Understanding the client-server model
✅ Cell Types: Code, Markdown, and Raw cells
✅ Execution Modes: Interactive and batch processing
✅ Basic Operations: Creating, editing, and running notebooks
✅ Keyboard Shortcuts: Efficient navigation and editing
✅ Kernel Management: Working with computational engines
✅ Markdown Documentation: Creating rich, formatted text
✅ Best Practices: Writing maintainable notebooks
✅ Troubleshooting: Common issues and solutions

Next Steps:

1. Explore JupyterLab: Master the modern interface
2. Learn Magic Commands: Enhance your workflow
3. Create Interactive Widgets: Build dynamic visualizations
4. Follow Best Practices: Write professional notebooks
5. Master Advanced Features: Leverage advanced capabilities
6. Deploy to Production: Use notebooks in real-world scenarios

Additional Resources:
- Jupyter Official Documentation: https://jupyter.org/
- JupyterLab User Guide: https://jupyterlab.readthedocs.io/
- Jupyter Notebook Documentation: https://jupyter-notebook.readthedocs.io/
- Jupyter Widgets: https://ipywidgets.readthedocs.io/

Practice Exercises:
1. Create a new notebook and write a simple data analysis
2. Practice using different cell types and markdown formatting
3. Experiment with keyboard shortcuts and navigation
4. Create a well-documented notebook with clear explanations
5. Share your notebook with others and get feedback

Happy Interactive Computing!
""")

if __name__ == "__main__":
    # Run the tutorial
    main()
    
    print("\n" + "="*60)
    print(" Tutorial completed successfully!")
    print(" To start Jupyter, run: jupyter lab")
    print("="*60) 