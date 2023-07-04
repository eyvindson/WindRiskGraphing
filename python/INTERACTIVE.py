def show_Period_plot_interactive(MM, SUVT, plotts, EXTRA):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import gaussian_kde, scoreatpercentile
    import warnings

    AREA = 287.5
    path = "./"
    files, rs = "SP_RESULTS_", "results/"
    var = ['Low_Low', 'High_Low', 'Mod_Low', 'Mod_Mid', 'Low_High']
    fig, axs = plt.subplots(6, 2, figsize=(18, 18), dpi=100)

    for SUV in SUVT:
        ls = "solid" if SUV == "_SALVAGE_NPV" else "--"
        colors = ["red", "blue", "green", "black", "orange"]

        for i, lm in enumerate(plotts):
            t11 = pd.read_csv(path + rs + files + "unsolved_EF_INCOME_" + var[lm] + SUV + EXTRA + ".csv")
            t11['EF_YEAR'] = t11['EF_YEAR'].str[0:4]
            YR = [str(2016 + 5 * i) for i in range(0, 6)]
            ax = axs[i // 2, i % 2]

            for j in range(6):
                data = t11.set_index("EF_YEAR").loc[YR[j]]["EF_AVG_NPV_unsolved_" + MM] / AREA
                data.plot.kde(alpha=0.5, ax=ax, color=colors[lm], linestyle=ls)

                xmin, xmax = ax.get_xlim()
                cvar_5_11 = scoreatpercentile(data, 5, interpolation_method='lower')

                ax.plot([cvar_5_11], [0], marker="^", color=colors[lm], markeredgecolor="black", markersize=6)

                kde = gaussian_kde(data)
                x = np.linspace(xmin, xmax, 1000)
                kde_y = kde(x)
                x0 = x[x < 200000 / AREA]
                x0_len = len(x0)
                y0 = kde_y[:x0_len]

                if x0_len > 1:
                    ax.fill_between(x=x0, y1=y0, color=colors[lm], alpha=.5)

            #ax.get_legend().remove()
            ax.set_title("Period {} Income".format(i + 1))

    plt.show()

    
def show_Period_plot_interactive(MM,SUVT,plotts,EXTRA):
    from matplotlib.lines import Line2D
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde, scoreatpercentile
    import warnings
    warnings.filterwarnings("ignore", message="Auto-removal of overlapping axes is deprecated")

    path = "./"
    files, rs =  "SP_RESULTS_", "results/"
    AREA = 287.5
    var = ['Low_Low',   'High_Low', 'Mod_Low','Mod_Mid','Low_High']
    fig, ax = plt.subplots(figsize=(18, 18), dpi=100)
    ax1 = plt.subplot2grid((5, 2), (1, 0), colspan=1)
    ax2 = plt.subplot2grid((5, 2), (1, 1), colspan=1)
    ax3 = plt.subplot2grid((5, 2), (2, 0), colspan=1)
    ax4 = plt.subplot2grid((5, 2), (2, 1), colspan=1)
    ax5 = plt.subplot2grid((5, 2), (3, 0), colspan=1)
    ax6 = plt.subplot2grid((5, 2), (3, 1), colspan=1)
    
    for SUV in SUVT:
        if SUV == "_SALVAGE_NPV":
            ls = "solid"
        else:
            ls = "--"
        colors = ["red","blue","green","black","orange"]
        for lm in plotts:
            t11 = pd.read_csv(path+rs+files+"unsolved_EF_INCOME_"+var[lm]+SUV+EXTRA+".csv")
            t11['EF_YEAR']=t11['EF_YEAR'].str[0:4]
            YR= [str(2016 + 5*i) for i in range(0,6)]
            t = [ax1,ax2,ax3,ax4,ax5,ax6]
            check = []
            t1 = pd.read_csv(path+rs+files+"unsolved_EF_"+var[lm]+SUV+EXTRA+".csv")
            for i in range(0,6):
                (t11.set_index("EF_YEAR").loc[YR[i]][["EF_AVG_NPV_unsolved_"+MM]]/AREA).plot.kde(alpha = 0.5,ax=t[i],color=colors[lm],linestyle=ls)
                xmin1, xmax1 = t[i].get_xlim()
                cvar_5_11 = scoreatpercentile(t11.set_index("EF_YEAR").loc[YR[i]][["EF_AVG_NPV_unsolved_"+MM]]/AREA, 5, interpolation_method='lower')

                t[i].plot([cvar_5_11],[0], marker = "^", color=colors[lm],markeredgecolor= "black",markersize=6)

                kde = gaussian_kde(t11.set_index("EF_YEAR").loc[YR[i]][["EF_AVG_NPV_unsolved_"+MM]]["EF_AVG_NPV_unsolved_"+MM]/AREA)
                xmin, xmax= xmin1,xmax1

                # create points between the min and max
                x = np.linspace(xmin, xmax, 1000)

                # calculate the y values from the model
                kde_y = kde(x)

                # select x values below 0
                x0 = x[x < 200000/AREA]

                # get the len, which will be used for slicing the other arrays
                x0_len = len(x0)

                # slice the arrays
                y0 = kde_y[:x0_len]
                x1 = x[x0_len:]
                y1 = kde_y[x0_len:]

                # fill the areas
                if x0_len>1:
                    t[i].fill_between(x=x0, y1=y0, color=colors[lm], alpha=.5)
                check = check+[cvar_5_11]
            titles =["1st Period Income","2nd Period Income","3rd Period Income","4th Period Income","5th Period Income","6th Period Income"]
            for i in range(0,6):
                t[i].get_legend().remove()
                t[i].set_title(titles[i])
            
                
    
    plt.show()
    

def show_NPV_plot_interactive(MM, SUVT, plotts, EXTRA):
    import matplotlib.pyplot as plt
    import pandas as pd

    AREA = 287.5
    path = "./"
    files, rs, VARI = "SP_RESULTS_", "results/", "MAX_AVG_NPV_unsolved_"
    var = ['Low_Low', 'High_Low', 'Mod_Low', 'Mod_Mid', 'Low_High']
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    for SUV in SUVT:
        ls = "solid" if SUV == "_SALVAGE_NPV" else "--"
        colors = ["red", "blue", "green", "black", "orange"]

        for lm in plotts:
            t1 = pd.read_csv(path + rs + files + "unsolved_EF_" + var[lm] + SUV + EXTRA + ".csv")
            (t1[VARI + MM] / AREA).plot.kde(ax=ax, alpha=0.5, color=colors[lm], linestyle=ls)
            ax.set_title("Net Present Value")
            custom_lines = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in plotts]
            ax.plot([t1[VARI + MM].mean() / AREA], [0], marker="x", color=colors[lm])
            ax.legend(custom_lines, [var[i] for i in plotts], bbox_to_anchor=(1.3, 1.0))

    ax.set_ylim(bottom=0)
    plt.show()


Graph = show_NPV_plot_interactive


def show_GUI():
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    var_select = widgets.RadioButtons(
        options=["LL", "HL", "MM", "ML", "LH"],
        value='LL',
        description='Outcome scenario:',
        disabled=False,
        layout=widgets.Layout(width='150px')
    )

    graph_select = widgets.RadioButtons(
        options=["Net Present Value", "Periodic Income"],
        value='Net Present Value',
        description='Which variable:',
        disabled=False,
        layout=widgets.Layout(width='150px')
    )

    salvage_select = widgets.RadioButtons(
        options=["5", "10", "15", "20"],
        value='5',
        description='Which salvage price:',
        disabled=False,
        layout=widgets.Layout(width='150px')
    )

    dictvar = {"LL": 0, "HL": 1, "MM": 2, "ML": 3, "LH": 4}

    checkboxes = [
        widgets.Checkbox(description="LL", indent = False, value=True, layout=widgets.Layout(width='150px')),
        widgets.Checkbox(description="HL", indent = False,layout=widgets.Layout(width='150px')),
        widgets.Checkbox(description="MM", indent = False,layout=widgets.Layout(width='150px')),
        widgets.Checkbox(description="ML", indent = False,layout=widgets.Layout(width='150px')),
        widgets.Checkbox(description="LH", indent = False,value=True, layout=widgets.Layout(width='150px'))
    ]

    OBJboxes = [
        widgets.Checkbox(description="NPV", indent = False,value=True, layout=widgets.Layout(width='150px')),
        widgets.Checkbox(description="CVAR", indent = False,layout=widgets.Layout(width='150px'))
    ]

    def on_change(change):
        clear_output()
        variables = [dictvar[i] for i in [checkbox.description for checkbox in checkboxes if checkbox.value]]
        obj_variables = ["_SALVAGE_" + i for i in [checkbox.description for checkbox in OBJboxes if checkbox.value]]
        display(widgets.HBox([graph_select, salvage_select, obj_checkboxes_container, var_select, checkboxes_container]))
        print(graph_select.value)
        global Graph
        if graph_select.value == "Net Present Value":
            Graph = show_NPV_plot_interactive
        else:
            Graph = show_Period_plot_interactive
        Graph(var_select.value,obj_variables,variables,"_" + salvage_select.value + ".0")
        

    var_select.observe(on_change, 'value')
    graph_select.observe(on_change, 'value')
    salvage_select.observe(on_change, 'value')
    for checkbox in checkboxes:
        checkbox.observe(on_change, 'value')
    for checkbox in OBJboxes:
        checkbox.observe(on_change, 'value')

    checkboxes_container = widgets.VBox([
        widgets.Label(value='Planned for scenarios:'),
        widgets.VBox(checkboxes, layout=widgets.Layout(justify_content ='flex-start'))
    ])

    obj_checkboxes_container = widgets.VBox([
        widgets.Label(value='Objectives:\r'),
        widgets.VBox(OBJboxes, layout=widgets.Layout(justify_content ='flex-start'))
    ])

    display(widgets.HBox([graph_select, salvage_select, obj_checkboxes_container, var_select, checkboxes_container]))
    Graph(var_select.value,
          ["_SALVAGE_" + i for i in [checkbox.description for checkbox in OBJboxes if checkbox.value]], [dictvar[i] for i in [checkbox.description for checkbox in checkboxes if checkbox.value]], "_" + salvage_select.value + ".0")
