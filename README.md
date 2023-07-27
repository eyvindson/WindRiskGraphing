# WindRiskGraphing: Interactive Plotting Tool

## Overview

This interactive plotting tool provides visualizations a selection of wind risk scenarios used in a draft manuscript exploring the potential of using stochastic programming for wind risk planning. We explore the trade-off between maximizing Net Present Value (NPV) and ensuring the even flow of periodic income. The tool leverages Matplotlib and Pandas libraries to create interactive graphs that allow users to explore and analyze various planning outcomes.

## Requirements

To use this plotting tool, ensure you have the following Python libraries installed:

- Matplotlib
- Pandas
- Numpy
- Scipy
- Jupyter Widgets (ipywidgets)

## How to Use

1. Ensure all the required libraries are installed.
2. Download the entire package to your desktop or virtual machine, including the folders holding the optimized results (the "results" folder), the python code containing the plotting functions, and the ipynb in the base folder.repository.
3. Import the functions into your Python script or Jupyter Notebook.
4. Use the functions to generate interactive plots for your data.

## Function Usage

### Function: `show_Period_plot_interactive`

This function generates a set of six interactive plots showing the Kernel Density Estimation (KDE) of Periodic Income scenarios. The plots allow you to visualize the distribution of income for different combinations of scenarios.

#### Parameters:

- `OCCUR` (list): A list of strings representing different scenarios for the occurance of wind intensity and frequency (e.g., "LL", "HL", "MM", "ML", "LH").
- `OBJECT` (list): A list representing which objective the user wishes to present.
- `PLANNED` (list): A list of strings representing different scenarios used in the optimization for wind intensity and frequency
- `SALVAGE` (string): A string representing the assumed salvage price used.

### Function: `show_NPV_plot_interactive`

This function generates an interactive plot showing the Kernel Density Estimation (KDE) of Net Present Value (NPV) scenarios. The plot allows you to visualize the distribution of NPV for different combinations of scenarios.

#### Parameters:

- `OCCUR` (list): A list of strings representing different scenarios for the occurance of wind intensity and frequency (e.g., "LL", "HL", "MM", "ML", "LH").
- `OBJECT` (list): A list representing which objective the user wishes to present.
- `PLANNED` (list): A list of strings representing different scenarios used in the optimization for wind intensity and frequency
- `SALVAGE` (string): A string representing the assumed salvage price used.

### Graphical User Interface (GUI)

This tool comes with a graphical user interface (GUI) that allows you to interactively select the scenarios and objectives you wish to visualize. The GUI is built using Jupyter Widgets and provides the flexibility to choose specific scenarios and objectives for plotting.

To use the GUI:

1. Run the `show_GUI()` function.
2. Select the variables and scenarios you want to visualize using the provided checkboxes and radio buttons.
3. The interactive plot will be automatically generated based on your selections.

The radio buttons and the check buttons allow for changes in what is explored. The specific scenarios and management objectives are defined in the draft manuscript. 

## Example Usage

```python
# Import the functions
from my_plotting_functions import show_Period_plot_interactive, show_NPV_plot_interactive

# Example usage of show_Period_plot_interactive
show_Period_plot_interactive(OCCUR=["LL", "LH"], OBJECT=["NPV", "CVAR"], PLANNED=["LL", "LH"], SALVAGE="_5.0")

# Example usage of show_NPV_plot_interactive
show_NPV_plot_interactive(OCCUR=["LL", "LH"], OBJECT=["NPV", "CVAR"], PLANNED=["LL", "LH"], SALVAGE="_5.0")
```

#Example GUI output:
![GUI Output](output_plot.png)

## Contributors

- [Kyle Eyvindson](https://github.com/eyvindson) - Project Developer

## License

This project is licensed under the [Creative Commons Zero v1.0 Universal](LICENSE).
