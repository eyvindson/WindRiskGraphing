# WindRiskGraphing: Interactive Plotting Tool

## Overview

This interactive plotting tool provides visualizations for Net Present Value (NPV) and Periodic Income scenarios based on different combinations of scenarios and objectives. The tool leverages Matplotlib and Pandas libraries to create interactive graphs that allow users to explore and analyze various planning outcomes.

## Requirements

To use this plotting tool, ensure you have the following Python libraries installed:

- Matplotlib
- Pandas
- Numpy
- Scipy
- Jupyter Widgets (ipywidgets)

## How to Use

1. Ensure all the required libraries are installed.
2. Download the `show_Period_plot_interactive` and `show_NPV_plot_interactive` functions from this repository.
3. Import the functions into your Python script or Jupyter Notebook.
4. Use the functions to generate interactive plots for your data.

## Function Usage

### Function: `show_Period_plot_interactive`

This function generates a set of six interactive plots showing the Kernel Density Estimation (KDE) of Periodic Income scenarios. The plots allow you to visualize the distribution of income for different combinations of scenarios.

#### Parameters:

- `MMT` (list): A list of strings representing different Max-Min scenarios (e.g., "LL", "HL", "MM", "ML", "LH").
- `SUVT` (list): A list of strings representing different salvage values (e.g., "5", "10", "15", "20").
- `plotts` (list): A list of integers representing indices of variables to be plotted.
- `EXTRA` (string): A string representing extra information for the data files.

### Function: `show_NPV_plot_interactive`

This function generates an interactive plot showing the Kernel Density Estimation (KDE) of Net Present Value (NPV) scenarios. The plot allows you to visualize the distribution of NPV for different combinations of scenarios.

#### Parameters:

- `MMT` (list): A list of strings representing different Max-Min scenarios (e.g., "LL", "HL", "MM", "ML", "LH").
- `SUVT` (list): A list of strings representing different salvage values (e.g., "5", "10", "15", "20").
- `plotts` (list): A list of integers representing indices of variables to be plotted.
- `EXTRA` (string): A string representing extra information for the data files.

### Graphical User Interface (GUI)

This tool comes with a graphical user interface (GUI) that allows you to interactively select the scenarios and objectives you wish to visualize. The GUI is built using Jupyter Widgets and provides the flexibility to choose specific scenarios and objectives for plotting.

To use the GUI:

1. Run the `show_GUI()` function.
2. Select the variables and scenarios you want to visualize using the provided checkboxes and radio buttons.
3. The interactive plot will be automatically generated based on your selections.

## Example Usage

```python
# Import the functions
from my_plotting_functions import show_Period_plot_interactive, show_NPV_plot_interactive

# Example usage of show_Period_plot_interactive
show_Period_plot_interactive(MMT=["LL", "LH"], SUVT=["5", "10", "15"], plotts=[0, 1, 2], EXTRA="_example")

# Example usage of show_NPV_plot_interactive
show_NPV_plot_interactive(MMT=["LL", "LH"], SUVT=["5", "10", "15"], plotts=[0, 1, 2], EXTRA="_example")
```

## Contributors

- [Kyle Eyvindson](https://github.com/eyvindson) - Project Developer

## License

This project is licensed under the [MIT License](LICENSE).
