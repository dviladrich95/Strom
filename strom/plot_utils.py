import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def plot_combined_cases(state_opt_df, state_base_df, plot_heater_output=True, plot_cooling_output=True, plot_price=True, plot_T_exterior=True, plot_wall_temp=True):
    # Determine the number of subplots based on heater output
    fig, (ax_temp, ax_cost) = plt.subplots(2, 1, figsize=(14, 8), 
                                              gridspec_kw={'height_ratios': [3, 1]}, sharex= True)
    

    # Single temperature axis
    legends_temp = []
    color = 'tab:red'
    ax_temp.set_ylabel('Temperature (°C)')
    ax_temp.plot(state_opt_df['InteriorTemperature'], color=color, linestyle='-', label='Optimal Interior Temp')
    ax_temp.plot(state_base_df['InteriorTemperature'], color=color, linestyle='--', label='Baseline Interior Temp')
    
    # Optional additional temperature plots
    if plot_wall_temp:
        color = 'tab:brown'
        ax_temp.plot(state_opt_df['WallTemperature'], color=color, linestyle='-', label='Optimal Wall Temp')
        ax_temp.plot(state_base_df['WallTemperature'], color=color, linestyle='--', label='Baseline Wall Temp')
    
    if plot_T_exterior:
        color = 'tab:pink'
        ax_temp.plot(state_opt_df['ExteriorTemperature'], color=color, linestyle='-', label='Exterior Temp')

    legends_temp = [(ax_temp.get_legend_handles_labels()[1], ax_temp, 'tab:red')]

    # Always plot cost on the first axis
    legends_cost = []
    color = 'tab:green'
    ax_cost.set_xlabel('Time (h)')
    ax_cost.set_ylabel('Cost (€)', color=color)
    ax_cost.plot(state_opt_df['Cost'].cumsum(), color=color, linestyle='-', label='Optimal Cost')
    ax_cost.plot(state_base_df['Cost'].cumsum(), color=color, linestyle='--', label='Baseline Cost')
    ax_cost.tick_params(axis='y', labelcolor=color)
    ax_cost.tick_params(axis='x', rotation=45)
    legends_cost.append((ax_cost.get_legend_handles_labels()[1], ax_cost, 'tab:green'))

    # Price Axis (if needed)
    if plot_price:
        ax_price = ax_cost.twinx()
        color = 'tab:grey'
        ax_price.plot(state_opt_df['Price'], color=color)
        ax_price.set_ylabel('Price (€/kWh)', color=color)
        ax_price.tick_params(axis='y', labelcolor=color)
        legends_cost.append((['Price'], ax_price, color))

    if plot_heater_output or plot_cooling_output:
        # Heater Output Subplot (if plot_heater_output is True)
        ax_heater = ax_cost.twinx()
        ax_heater.spines["right"].set_position(("outward", 60))
        ax_heater_label = []
        if plot_heater_output:
            color = 'tab:red'
            ax_heater.plot(state_opt_df['HeaterOutput']*100, color=color, linestyle='-', label='Optimal Heater Output')
            ax_heater.plot(state_base_df['HeaterOutput']*100, color=color, linestyle='--', label='Baseline Heater Output')
            ax_heater_label.append('Heater')

        # Heater Output Subplot (if plot_heater_output is True)
        if plot_cooling_output:
            color = 'tab:blue'
            ax_heater.plot(state_opt_df['CoolingOutput']*100, color=color, linestyle='-', label='Optimal Cooling Output')
            ax_heater.plot(state_base_df['CoolingOutput']*100, color=color, linestyle='--', label='Baseline Cooling Output')
            ax_heater_label.append('Cooling')

        ax_heater.set_ylabel('/'.join(ax_heater_label)+' Output (%)')
        legends_cost.append((ax_heater.get_legend_handles_labels()[1], ax_heater, 'tab:red'))

    # Place temp legends
    for i, (legend_text, ax, color) in enumerate(legends_temp):
        ax.legend(
            legend_text, 
            loc='lower left', 
            bbox_to_anchor=(0.25*i, 1.01), 
            ncol=len(legend_text),
            prop={'size': 10}
        )

    # Place cost legends
    for i, (legend_text, ax, color) in enumerate(legends_cost):
        ax.legend(
            legend_text, 
            loc='lower left', 
            bbox_to_anchor=(0.30*i, 1.01), 
            ncol=len(legend_text),
            prop={'size': 10}
        )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for legends

    return fig


def plot_factor_analysis(optimal_cost,baseline_cost,
                        C_walls_list, Q_heater_list, R_external_list,
                        type):
    # Create meshgrid
    X, Y, Z = np.meshgrid(C_walls_list, Q_heater_list, R_external_list)
    if type == 'Relative':
        values = 100 * (baseline_cost - optimal_cost) / baseline_cost
        title = 'Relative Cost Savings (%)'
    elif type == 'Absolute':
        values = baseline_cost - optimal_cost
        title = 'Absolute Cost Savings (€)'


    # Flatten for Plotly
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    values_flat = values.flatten()

    # Create interactive 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=X_flat, 
        y=Y_flat, 
        z=Z_flat, 
        mode='markers',
        marker=dict(
            size=5,
            color=values_flat,  # Color by cost_diff values
            colorscale='PRGn',
            colorbar=dict(title=title),
            opacity=0.8
        )
    )])

    # Set axis labels
    fig.update_layout(
        scene=dict(
            xaxis_title='Wall heat capacity (kWh/°C)',
            yaxis_title='Heating Power (kW)',
            zaxis_title='R-Value (°C/kW)',
            aspectmode='cube',  # Forces equal aspect ratio
            aspectratio=dict(x=1, y=1, z=1)  # Sets the aspect ratio to 1:1:1
        ),
        title="{} Cost Savings Analysis".format(type),
    )
    return fig