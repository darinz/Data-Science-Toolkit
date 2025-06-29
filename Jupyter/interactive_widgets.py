#!/usr/bin/env python3
"""
Interactive Widgets: Creating Dynamic Visualizations

Welcome to the Interactive Widgets tutorial! Jupyter widgets provide interactive 
HTML widgets for Jupyter notebooks and JupyterLab, enabling you to create 
dynamic, interactive visualizations and user interfaces.

This script covers:
- Introduction to Jupyter widgets
- Basic widget types and properties
- Event handling and callbacks
- Layout and styling
- Custom widget development
- Advanced widget features

Prerequisites:
- Python 3.8 or higher
- Basic understanding of Jupyter (covered in jupyter_basics.py)
- Familiarity with HTML, CSS, and JavaScript (helpful)
"""

import os
import sys
import json
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
    
    print("Interactive Widgets Tutorial")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print("Interactive widgets tutorial started successfully!")

    # Section 1: Introduction to Jupyter Widgets
    print_section_header("1. Introduction to Jupyter Widgets")
    
    print("""
Jupyter widgets are interactive HTML widgets for Jupyter notebooks and JupyterLab. 
They enable you to create dynamic, interactive visualizations and user interfaces 
directly in your notebooks.

Key Features:
- Interactive HTML widgets
- Real-time data visualization
- User input and controls
- Event-driven programming
- Custom widget development
- Integration with Python code

Benefits:
✅ Interactive data exploration
✅ Dynamic visualizations
✅ User-friendly interfaces
✅ Real-time feedback
✅ Customizable components
✅ Cross-platform compatibility
""")

    # Section 2: Basic Widget Types
    print_section_header("2. Basic Widget Types")
    
    print("""
Jupyter widgets come in several categories, each serving different purposes:

1. Numeric Widgets:
   - IntSlider: Integer slider
   - FloatSlider: Float slider
   - IntRangeSlider: Range slider for integers
   - FloatRangeSlider: Range slider for floats
   - IntText: Integer text input
   - FloatText: Float text input
   - BoundedIntText: Bounded integer input
   - BoundedFloatText: Bounded float input

2. Boolean Widgets:
   - Checkbox: True/false checkbox
   - ToggleButton: Toggle button
   - Valid: Validation widget

3. Selection Widgets:
   - Dropdown: Dropdown selection
   - Select: Single selection
   - SelectMultiple: Multiple selection
   - RadioButtons: Radio button group
   - SelectionSlider: Slider with discrete values

4. String Widgets:
   - Text: Single-line text input
   - Textarea: Multi-line text input
   - Password: Password input
   - Combobox: Combo box with suggestions

5. Action Widgets:
   - Button: Clickable button
   - FileUpload: File upload widget
   - ColorPicker: Color picker
   - DatePicker: Date picker
   - Play: Play/pause button with slider
""")

    # Demonstrate widget examples
    print_subsection_header("Widget Examples")
    
    widget_examples = [
        ("IntSlider", "widgets.IntSlider(value=50, min=0, max=100, step=1, description='Value:')"),
        ("FloatSlider", "widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.1, description='Probability:')"),
        ("Checkbox", "widgets.Checkbox(value=False, description='Enable feature')"),
        ("Dropdown", "widgets.Dropdown(options=['Option 1', 'Option 2', 'Option 3'], value='Option 1', description='Select:')"),
        ("Text", "widgets.Text(value='Hello World', description='Input:')"),
        ("Button", "widgets.Button(description='Click me!', button_style='success')"),
        ("ColorPicker", "widgets.ColorPicker(concise=False, description='Pick a color', value='red')"),
        ("Play", "widgets.Play(value=0, min=0, max=100, step=1, description='Play', disabled=False)"),
    ]
    
    for widget_type, example in widget_examples:
        print(f"\n{widget_type}:")
        print(f"```python")
        print(f"import ipywidgets as widgets")
        print(f"{example}")
        print(f"```")

    # Section 3: Widget Properties and Attributes
    print_section_header("3. Widget Properties and Attributes")
    
    print("""
Widgets have various properties that control their appearance and behavior:

1. Value Properties:
   - value: Current value of the widget
   - min/max: Minimum and maximum values
   - step: Step size for numeric widgets
   - options: Available options for selection widgets

2. Display Properties:
   - description: Label displayed next to the widget
   - placeholder: Placeholder text for text inputs
   - disabled: Whether the widget is disabled
   - continuous_update: Update value continuously

3. Style Properties:
   - button_style: Button style ('primary', 'success', 'info', 'warning', 'danger')
   - layout: Layout properties (width, height, margin, padding)
   - style: Visual style properties
   - icon: Icon for buttons

4. Event Properties:
   - on_click: Callback for button clicks
   - on_value_change: Callback for value changes
   - on_submit: Callback for form submission
""")

    # Demonstrate property examples
    print_subsection_header("Property Examples")
    
    print("""
```python
import ipywidgets as widgets

# Basic slider with properties
slider = widgets.IntSlider(
    value=50,
    min=0,
    max=100,
    step=5,
    description='Value:',
    disabled=False,
    continuous_update=True,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

# Button with style
button = widgets.Button(
    description='Submit',
    disabled=False,
    button_style='success',
    tooltip='Click to submit',
    icon='check'
)

# Dropdown with options
dropdown = widgets.Dropdown(
    options=['Red', 'Green', 'Blue'],
    value='Red',
    description='Color:',
    disabled=False,
    layout=widgets.Layout(width='200px')
)
```
""")

    # Section 4: Event Handling and Callbacks
    print_section_header("4. Event Handling and Callbacks")
    
    print("""
Widgets support event-driven programming through callbacks. You can attach 
functions to widget events to respond to user interactions.

Common Events:
- on_click: Button click events
- on_value_change: Value change events
- on_submit: Form submission events
- on_trait_change: Trait change events (deprecated)
""")

    # Demonstrate callback examples
    print_subsection_header("Callback Examples")
    
    print("""
```python
import ipywidgets as widgets
from IPython.display import display, clear_output

# Simple button callback
def on_button_click(b):
    print(f"Button clicked! Current value: {b.description}")

button = widgets.Button(description='Click me!')
button.on_click(on_button_click)
display(button)

# Slider with value change callback
def on_slider_change(change):
    print(f"Slider value changed to: {change['new']}")

slider = widgets.IntSlider(value=50, min=0, max=100, description='Value:')
slider.observe(on_slider_change, names='value')
display(slider)

# Multiple widgets with shared callback
def update_output(change):
    with output:
        clear_output()
        print(f"Slider: {slider.value}")
        print(f"Text: {text.value}")

slider = widgets.IntSlider(value=50, min=0, max=100, description='Slider:')
text = widgets.Text(value='Hello', description='Text:')
output = widgets.Output()

slider.observe(update_output, names='value')
text.observe(update_output, names='value')

display(slider, text, output)
```
""")

    # Section 5: Layout and Styling
    print_section_header("5. Layout and Styling")
    
    print("""
Widgets can be styled and arranged using layout and style properties. This 
allows you to create professional-looking interfaces.

Layout Properties:
- width: Widget width
- height: Widget height
- margin: Margin around widget
- padding: Padding inside widget
- border: Border style
- display: Display type
- flex: Flexbox properties
""")

    # Demonstrate layout examples
    print_subsection_header("Layout Examples")
    
    print("""
```python
import ipywidgets as widgets

# Basic layout
slider = widgets.IntSlider(
    value=50,
    min=0,
    max=100,
    description='Value:',
    layout=widgets.Layout(width='300px', height='50px')
)

# Container layouts
box = widgets.VBox([
    widgets.HTML(value='<h3>Widget Group</h3>'),
    widgets.IntSlider(description='Slider 1:'),
    widgets.IntSlider(description='Slider 2:'),
    widgets.Button(description='Submit')
], layout=widgets.Layout(
    width='400px',
    border='2px solid gray',
    padding='10px',
    margin='5px'
))

# Grid layout
grid = widgets.GridBox([
    widgets.Button(description=f'Button {i}') for i in range(9)
], layout=widgets.Layout(
    width='300px',
    grid_template_columns='repeat(3, 1fr)',
    grid_gap='5px'
))

# Flexbox layout
flex = widgets.HBox([
    widgets.IntSlider(description='Min:'),
    widgets.IntSlider(description='Max:'),
    widgets.Button(description='Apply')
], layout=widgets.Layout(
    justify_content='space-between',
    align_items='center'
))
```
""")

    # Section 6: Widget Containers
    print_section_header("6. Widget Containers")
    
    print("""
Container widgets allow you to group and organize other widgets:

1. Box Containers:
   - VBox: Vertical box layout
   - HBox: Horizontal box layout
   - Box: Generic box container

2. Grid Containers:
   - GridBox: Grid layout container
   - TwoByTwoLayout: 2x2 grid layout

3. Accordion and Tabs:
   - Accordion: Collapsible sections
   - Tab: Tabbed interface

4. Output Containers:
   - Output: Captures and displays output
   - HTML: Displays HTML content
   - Label: Simple text label
""")

    # Demonstrate container examples
    print_subsection_header("Container Examples")
    
    print("""
```python
import ipywidgets as widgets

# VBox (Vertical layout)
vbox = widgets.VBox([
    widgets.HTML(value='<h2>Settings</h2>'),
    widgets.IntSlider(description='Parameter 1:'),
    widgets.IntSlider(description='Parameter 2:'),
    widgets.Button(description='Save Settings')
])

# HBox (Horizontal layout)
hbox = widgets.HBox([
    widgets.IntSlider(description='Min:'),
    widgets.IntSlider(description='Max:'),
    widgets.Button(description='Apply')
])

# Accordion
accordion = widgets.Accordion([
    widgets.VBox([
        widgets.IntSlider(description='Slider 1:'),
        widgets.IntSlider(description='Slider 2:')
    ]),
    widgets.VBox([
        widgets.Text(description='Text 1:'),
        widgets.Text(description='Text 2:')
    ])
], titles=['Group 1', 'Group 2'])

# Tabs
tab = widgets.Tab([
    widgets.VBox([widgets.IntSlider(description='Tab 1:')]),
    widgets.VBox([widgets.Text(description='Tab 2:')]),
    widgets.VBox([widgets.Button(description='Tab 3:')])
])
tab.set_title(0, 'Parameters')
tab.set_title(1, 'Input')
tab.set_title(2, 'Actions')
```
""")

    # Section 7: Interactive Visualizations
    print_section_header("7. Interactive Visualizations")
    
    print("""
Widgets can be combined with plotting libraries to create interactive 
visualizations that respond to user input.
""")

    # Demonstrate interactive visualization
    print_subsection_header("Interactive Plot Example")
    
    print("""
```python
import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

def plot_function(amplitude, frequency, phase):
    x = np.linspace(0, 4*np.pi, 100)
    y = amplitude * np.sin(frequency * x + phase)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b-', linewidth=2)
    plt.title(f'Sine Wave: A={amplitude}, f={frequency}, φ={phase}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.ylim(-3, 3)
    plt.show()

# Create widgets
amplitude = widgets.FloatSlider(value=1.0, min=0.1, max=2.0, step=0.1, description='Amplitude:')
frequency = widgets.FloatSlider(value=1.0, min=0.1, max=3.0, step=0.1, description='Frequency:')
phase = widgets.FloatSlider(value=0.0, min=0, max=2*np.pi, step=0.1, description='Phase:')

# Create output widget
output = widgets.Output()

def update_plot(change):
    with output:
        clear_output(wait=True)
        plot_function(amplitude.value, frequency.value, phase.value)

# Attach callbacks
amplitude.observe(update_plot, names='value')
frequency.observe(update_plot, names='value')
phase.observe(update_plot, names='value')

# Display widgets and initial plot
display(widgets.VBox([amplitude, frequency, phase, output]))
update_plot(None)
```
""")

    # Section 8: Custom Widget Development
    print_section_header("8. Custom Widget Development")
    
    print("""
You can create custom widgets by extending existing widget classes or 
creating completely new widgets using JavaScript and Python.
""")

    # Demonstrate custom widget
    print_subsection_header("Custom Widget Example")
    
    print("""
```python
import ipywidgets as widgets
from traitlets import Unicode, Int, Bool, observe

class CustomCounter(widgets.DOMWidget):
    _view_name = Unicode('CustomCounterView').tag(sync=True)
    _view_module = Unicode('custom-counter').tag(sync=True)
    _view_module_version = Unicode('1.0.0').tag(sync=True)
    
    value = Int(0).tag(sync=True)
    label = Unicode('Counter').tag(sync=True)
    disabled = Bool(False).tag(sync=True)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def increment(self):
        self.value += 1
    
    def decrement(self):
        self.value -= 1
    
    def reset(self):
        self.value = 0

# JavaScript code for the widget
javascript_code = '''
require.undef('custom-counter');

define('custom-counter', ['@jupyter-widgets/base'], function(widgets) {
    var CustomCounterView = widgets.DOMWidgetView.extend({
        render: function() {
            this.el.innerHTML = `
                <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px;">
                    <h3>${this.model.get('label')}</h3>
                    <div style="font-size: 24px; margin: 10px 0;">${this.model.get('value')}</div>
                    <button onclick="this.increment()" ${this.model.get('disabled') ? 'disabled' : ''}>+</button>
                    <button onclick="this.decrement()" ${this.model.get('disabled') ? 'disabled' : ''}>-</button>
                    <button onclick="this.reset()" ${this.model.get('disabled') ? 'disabled' : ''}>Reset</button>
                </div>
            `;
            
            this.increment = () => this.model.set('value', this.model.get('value') + 1);
            this.decrement = () => this.model.set('value', this.model.get('value') - 1);
            this.reset = () => this.model.set('value', 0);
            
            this.model.on('change:value', this.value_changed, this);
            this.model.on('change:label', this.label_changed, this);
        },
        
        value_changed: function() {
            this.el.querySelector('div').textContent = this.model.get('value');
        },
        
        label_changed: function() {
            this.el.querySelector('h3').textContent = this.model.get('label');
        }
    });
    
    return {
        CustomCounterView: CustomCounterView
    };
});
'''

# Register the widget
from IPython.display import display, HTML
display(HTML(f'<script>{javascript_code}</script>'))

# Use the custom widget
counter = CustomCounter(label='My Counter', value=10)
display(counter)
```
""")

    # Section 9: Advanced Widget Features
    print_section_header("9. Advanced Widget Features")
    
    print("""
Advanced widget features provide additional functionality for complex applications:

1. Widget Linking:
   - Link widgets together
   - Synchronize values
   - Create dependencies

2. Widget Validation:
   - Input validation
   - Error handling
   - User feedback

3. Widget State Management:
   - Save and restore widget state
   - Serialize widget configurations
   - Share widget states

4. Performance Optimization:
   - Debouncing callbacks
   - Efficient updates
   - Memory management
""")

    # Demonstrate advanced features
    print_subsection_header("Advanced Features Examples")
    
    print("""
```python
import ipywidgets as widgets
import json

# Widget linking
slider1 = widgets.IntSlider(value=50, min=0, max=100, description='Slider 1:')
slider2 = widgets.IntSlider(value=50, min=0, max=100, description='Slider 2:')

# Link the sliders
widgets.jslink((slider1, 'value'), (slider2, 'value'))

# Widget validation
def validate_input(change):
    if change['new'] < 0:
        text.value = '0'
        text.description = 'Value (must be >= 0):'
    else:
        text.description = 'Value:'

text = widgets.IntText(value=0, description='Value:')
text.observe(validate_input, names='value')

# Save widget state
def save_state():
    state = {
        'slider1': slider1.value,
        'slider2': slider2.value,
        'text': text.value
    }
    with open('widget_state.json', 'w') as f:
        json.dump(state, f)

def load_state():
    try:
        with open('widget_state.json', 'r') as f:
            state = json.load(f)
        slider1.value = state['slider1']
        slider2.value = state['slider2']
        text.value = state['text']
    except FileNotFoundError:
        print('No saved state found')

# Debounced callback
import time
from functools import partial

def debounced_callback(func, delay=0.5):
    def wrapper(*args, **kwargs):
        if hasattr(wrapper, '_timer'):
            wrapper._timer.cancel()
        wrapper._timer = threading.Timer(delay, func, args, kwargs)
        wrapper._timer.start()
    return wrapper

@debounced_callback
def expensive_operation(change):
    print(f'Processing value: {change["new"]}')
    time.sleep(0.1)  # Simulate expensive operation

slider = widgets.IntSlider(description='Debounced:')
slider.observe(expensive_operation, names='value')
```
""")

    # Section 10: Best Practices
    print_section_header("10. Best Practices")
    
    print("""
Follow these best practices when working with Jupyter widgets:

1. Widget Organization:
   - Group related widgets together
   - Use appropriate containers
   - Provide clear labels and descriptions
   - Maintain consistent styling

2. Performance:
   - Use debouncing for expensive callbacks
   - Avoid unnecessary widget updates
   - Clean up event handlers
   - Monitor memory usage

3. User Experience:
   - Provide immediate feedback
   - Handle errors gracefully
   - Use appropriate widget types
   - Maintain responsive interfaces

4. Code Organization:
   - Separate widget creation from logic
   - Use functions for complex callbacks
   - Document custom widgets
   - Follow naming conventions

5. Accessibility:
   - Provide keyboard navigation
   - Use descriptive labels
   - Ensure color contrast
   - Support screen readers
""")

    # Section 11: Troubleshooting
    print_section_header("11. Troubleshooting")
    
    print("""
Common issues with Jupyter widgets and solutions:

1. Widgets Not Displaying:
   - Check widget installation
   - Restart Jupyter kernel
   - Update widget packages
   - Check browser console

2. Callbacks Not Working:
   - Verify callback function signature
   - Check widget property names
   - Ensure proper event binding
   - Debug with print statements

3. Layout Issues:
   - Check CSS conflicts
   - Verify layout properties
   - Test in different browsers
   - Use browser developer tools

4. Performance Problems:
   - Profile callback functions
   - Use debouncing
   - Limit widget updates
   - Monitor memory usage

5. Custom Widget Issues:
   - Check JavaScript syntax
   - Verify widget registration
   - Test in isolation
   - Review browser console errors
""")

    # Section 12: Summary and Next Steps
    print_section_header("12. Summary and Next Steps")
    
    print("""
Congratulations! You've completed the Interactive Widgets tutorial. Here's what you've learned:

Key Concepts Covered:
✅ Widget Types: Numeric, boolean, selection, string, and action widgets
✅ Properties and Attributes: Value, display, style, and event properties
✅ Event Handling: Callbacks and event-driven programming
✅ Layout and Styling: Arranging and styling widgets
✅ Widget Containers: Grouping and organizing widgets
✅ Interactive Visualizations: Combining widgets with plots
✅ Custom Widget Development: Creating custom widgets
✅ Advanced Features: Linking, validation, and state management
✅ Best Practices: Effective widget usage
✅ Troubleshooting: Common issues and solutions

Next Steps:

1. Practice with Widgets: Create interactive interfaces
2. Build Custom Widgets: Develop specialized components
3. Create Interactive Dashboards: Combine multiple widgets
4. Integrate with Data Science: Use widgets for data exploration
5. Explore Advanced Features: Master complex widget interactions
6. Contribute to Community: Share useful widgets

Additional Resources:
- Jupyter Widgets Documentation: https://ipywidgets.readthedocs.io/
- Widget Examples: https://github.com/jupyter-widgets/ipywidgets/tree/master/docs/source/examples
- Custom Widget Development: https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20Custom.html
- Widget Gallery: https://ipywidgets.readthedocs.io/en/stable/examples/Widget%20List.html

Practice Exercises:
1. Create an interactive data exploration interface
2. Build a custom widget for your specific needs
3. Develop an interactive dashboard with multiple widgets
4. Create a widget-based form with validation
5. Build an interactive visualization with real-time updates
6. Design a widget library for your team

Happy Widget-ing! 🎛️
""")

if __name__ == "__main__":
    # Run the tutorial
    main()
    
    print("\n" + "="*60)
    print(" Tutorial completed successfully!")
    print(" Try creating interactive widgets in Jupyter!")
    print("="*60) 