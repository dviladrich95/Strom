import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from typing import List
from matplotlib.figure import Figure

def plot_combined_cases(
    state_opt_df: pd.DataFrame,
    state_base_df: pd.DataFrame,
    plot_heater_output: bool = True,
    plot_cooling_output: bool = False,
    plot_price: bool = True,
    plot_T_exterior: bool = True,
    plot_wall_temp: bool = True
) -> Figure:
    """
    Create a combined plot comparing optimal and baseline cases for temperature and cost metrics.

    Args:
        state_opt_df: DataFrame containing optimal state data
        state_base_df: DataFrame containing baseline state data
        plot_heater_output: Whether to plot heater output
        plot_cooling_output: Whether to plot cooling output
        plot_price: Whether to plot electricity price
        plot_T_exterior: Whether to plot exterior temperature
        plot_wall_temp: Whether to plot wall temperature

    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    # Determine the number of subplots based on heater output
    fig, (ax_temp, ax_cost) = plt.subplots(2, 1, figsize=(14, 8),
                                              gridspec_kw={'height_ratios': [3, 1]}, sharex= True)

    start_time = state_opt_df.index.min()
    end_time = state_opt_df.index.max()
    ax_temp.set_xlim(start_time, end_time)
    ax_cost.set_xlim(start_time, end_time)

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

    # Price Axis with smoothing
    legends_cost = []

    if plot_price:
        ax_price = ax_cost.twinx()
        color = 'tab:grey'
        ax_price.plot(state_opt_df['Price'], color=color)
        ax_price.set_ylabel('Price (€/kWh)', color=color)
        ax_price.tick_params(axis='y', labelcolor=color)
        legends_cost.append((['Electricity Price'], ax_price, color))

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

        ax_heater.set_ylabel('/'.join(ax_heater_label)+' Output (%)', fontsize=8)
        ax_heater.set_ylim(0, 100)
        ax_heater.tick_params(axis='y', labelcolor='tab:red')
        legends_cost.append((ax_heater.get_legend_handles_labels()[1], ax_heater, 'tab:red'))

    # Place temp legend centered
    for i, (legend_text, ax, color) in enumerate(legends_temp):
        ax.legend(
            legend_text,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.01),
            ncol=len(legend_text),
            prop={'size': 10}
        )

    # Merge all cost legends into one centered legend
    all_handles, all_labels = [], []
    for legend_text, ax, color in legends_cost:
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)
    if all_handles:
        ax_cost.legend(all_handles, all_labels,
                       loc='lower center', bbox_to_anchor=(0.5, 1.01),
                       ncol=len(all_labels), prop={'size': 10})
        
    # Adjust layout
    plt.tight_layout(rect=(0, 0, 1, 0.95))  # Leave room for legends

    return fig

def plot_combined_cases_years(
    state_opt_df: pd.DataFrame,
    state_base_df: pd.DataFrame,
    plot_T_exterior: bool = True
) -> Figure:
    """
    Create a combined yearly plot comparing optimal and baseline cases with daily aggregations.

    Args:
        state_opt_df: DataFrame containing optimal state data
        state_base_df: DataFrame containing baseline state data
        plot_T_exterior: Whether to plot exterior temperature

    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    # Determine the number of subplots based on heater output
    fig, (ax_temp, ax_cost) = plt.subplots(2, 1, figsize=(14, 8), 
                                              gridspec_kw={'height_ratios': [3, 1]}, sharex= True)
    
    start_time = state_opt_df.index.min()
    end_time = state_opt_df.index.max()
    ax_temp.set_xlim(start_time, end_time)
    ax_cost.set_xlim(start_time, end_time)

    if plot_T_exterior:
        # Create daily aggregations
        daily_temp = state_opt_df['ExteriorTemperature'].resample('D').agg(['mean', 'min', 'max'])
        
        # Create a rolling average of the daily mean temperatures (7-day window)
        rolling_mean = daily_temp['mean'].rolling(window=7, center=True).mean()
        

        # First, add the comfort zone as a shaded area
        comfort_min = 18.0
        comfort_max = 24.0
        ax_temp.axhspan(comfort_min, comfort_max, alpha=0.2, color='green', label='Comfort Zone (18-24°C)')

        # Plot the smoothed line
        color = 'tab:pink'
        ax_temp.plot(rolling_mean.index, rolling_mean, 
                    color=color, linestyle='-', linewidth=2, 
                    label='Exterior Temp (7-day avg)')
        
        # Create shaded envelope for min/max temperatures
        ax_temp.fill_between(daily_temp.index, 
                            daily_temp['min'], 
                            daily_temp['max'], 
                            color=color, alpha=0.3,
                            label='Daily Min/Max Range')

        # Add legend if not already present elsewhere
        handles, labels = ax_temp.get_legend_handles_labels()
        if 'Exterior Temp' in labels or 'Exterior Temp (7-day avg)' in labels:
            ax_temp.legend(loc='best')

    # Single temperature axis
    legends_temp = []
    color = 'tab:red'
    ax_temp.set_ylabel('Temperature (°C)')
    ax_temp.plot(state_opt_df['InteriorTemperature'], color=color, linestyle='-', label='Optimal Interior Temperature')
    ax_temp.plot(state_base_df['InteriorTemperature'], color='k', linestyle='-', label='Baseline Interior Temperature')

    legends_temp = [(ax_temp.get_legend_handles_labels()[1], ax_temp, 'tab:red')]

    # Create a twin axis for cost savings


    color = 'tab:grey'
    
    # Resample price to daily values
    daily_price = state_opt_df['Price'].resample('D').agg(['mean', 'min', 'max'])
    
    # Create a rolling average (7-day window)
    rolling_price = daily_price['mean'].rolling(window=7, center=True).mean()
    
    # Plot the smoothed line
    ax_cost.plot(rolling_price.index, rolling_price,
                color=color, linestyle='-', linewidth=2,
                label='Electricity Price (7-day avg)')
    
    ax_cost.fill_between(daily_price.index, 
                        daily_price['min'], 
                        daily_price['max'], 
                        color=color, alpha=0.3,
                        label='Daily Min/Max Range')
    ax_cost.set_ylabel('Price (€/kWh)')
    ax_cost.tick_params(axis='y', labelcolor=color)

    # Cumulative cost
    ax_cost2 = ax_cost.twinx()
    ax_cost2.set_xlabel('Time (h)')
    ax_cost2.set_ylabel('Cumulative Cost (€)')
    ax_cost2.plot(state_opt_df['Cost'].cumsum(), color='tab:red', linestyle='-', label='Optimal Cumulative Cost')
    ax_cost2.plot(state_base_df['Cost'].cumsum(), color='k', linestyle='-', label='Baseline Cumulative Cost')
    ax_cost2.tick_params(axis='y', labelcolor='k')
    ax_cost2.tick_params(axis='x', rotation=45)
    print(sum(state_base_df['Cost']-state_opt_df['Cost'])/sum(state_base_df['Cost']))

    # Place temp legend
    for i, (legend_text, ax, color) in enumerate(legends_temp):
        ax.legend(
            legend_text,
            loc='lower center',
            bbox_to_anchor=(0.5, 1.01),
            ncol=len(legend_text),
            prop={'size': 10}
        )

    # Merge all cost legends into one
    all_handles, all_labels = [], []
    for ax in [ax_cost, ax_cost2]:
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)
    ax_cost.legend(all_handles, all_labels,
                   loc='lower center', bbox_to_anchor=(0.5, 1.01),
                   ncol=len(all_labels), prop={'size': 10})

    # Adjust layout
    plt.tight_layout(rect=(0, 0, 1, 0.95))  # Leave room for legends

    return fig


def plot_factor_analysis(
    optimal_cost: np.ndarray,
    baseline_cost: np.ndarray,
    C_walls_list: List[float],
    Q_heater_list: List[float],
    R_external_list: List[float],
    type: str
) -> go.Figure:
    """
    Create a 3D scatter plot analyzing the impact of different factors on cost savings.

    Args:
        optimal_cost: Array of optimal costs for different parameter combinations
        baseline_cost: Array of baseline costs for different parameter combinations
        C_walls_list: List of wall heat capacity values
        Q_heater_list: List of heating power values
        R_external_list: List of R-values (thermal resistance)
        type: Type of analysis ('Relative' or 'Absolute')

    Returns:
        plotly.graph_objects.Figure: Interactive 3D scatter plot
    """
    # Create meshgrid
    X, Y, Z = np.meshgrid(C_walls_list, Q_heater_list, R_external_list)
    if type == 'Relative':
        values = 100 * (baseline_cost - optimal_cost) / baseline_cost
        title = 'Relative Cost Savings (%)'
    elif type == 'Absolute':
        values = baseline_cost - optimal_cost
        title = 'Absolute Cost Savings (€)'
    else:
        raise ValueError("Invalid type. Expected 'Relative' or 'Absolute'.")


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


def _build_thermal_schematic(
    T_air: float,
    T_wall: float,
    T_ext: float,
    C_air: float = 0.26,
    C_wall: float = 19.1,
    T_min: float = 18.0,
    T_max: float = 24.0,
) -> dict:
    """
    Build the thermal schematic figure and return handles to all mutable artists.

    x-axis = cumulative thermal capacity (kWh/°C); each band's width equals the
    heat capacity of that component, so area of the colored fill = stored energy C×T.
    """
    x_air_l  = 0.0
    x_air_r  = C_air
    x_wall_l = C_air
    x_wall_r = C_air + C_wall
    x_ext_l  = C_air + C_wall
    x_ext_r  = C_air + C_wall + C_wall * 0.25

    T_plot_max = T_max + 3

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(x_air_l, x_ext_r)
    ax.set_ylim(0, T_plot_max)

    # Static: background bands
    ax.axvspan(x_air_l,  x_air_r,  color='lightyellow', alpha=0.8, zorder=0)
    ax.axvspan(x_wall_l, x_wall_r, color='#e8e0d8',     alpha=0.8, zorder=0)
    ax.axvspan(x_ext_l,  x_ext_r,  color='#ddeeff',     alpha=0.8, zorder=0)

    # Static: comfort band
    ax.axhspan(T_min, T_max, color='green', alpha=0.07, zorder=1,
               label=f'Comfort band ({T_min}–{T_max} °C)')

    # Static: band separators
    ax.axvline(x_air_r,  color='black', linewidth=1.5, zorder=4)
    ax.axvline(x_wall_r, color='black', linewidth=1.5, zorder=4)

    # Mutable: exterior fill, line, label
    rect_ext = mpatches.Rectangle(
        (x_ext_l, 0), x_ext_r - x_ext_l, T_ext,
        facecolor='#ff69b4', alpha=0.35, zorder=2)
    ax.add_patch(rect_ext)
    line_ext, = ax.plot([x_ext_l, x_ext_r], [T_ext, T_ext],
                        color='#cc3377', linestyle='--', linewidth=1.5, zorder=3)
    text_ext = ax.text(x_ext_r * 0.998, T_ext + 0.4, f'{T_ext:.1f} °C',
                       color='#cc3377', ha='right', va='bottom', fontsize=10)

    # Static: band header labels and capacity annotations
    y_label = T_plot_max * 0.97
    y_cap   = T_plot_max * 0.87
    ax.text((x_air_l  + x_air_r)  / 2, y_label, 'Air',      ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text((x_wall_l + x_wall_r) / 2, y_label, 'Wall',     ha='center', va='top', fontsize=13, fontweight='bold')
    ax.text((x_ext_l  + x_ext_r)  / 2, y_label, 'Exterior', ha='center', va='top', fontsize=10, fontweight='bold')
    ax.text((x_air_l  + x_air_r)  / 2, y_cap, f'$C_{{air}}={C_air}$ kWh/°C',
            ha='center', va='top', fontsize=8, color='grey')
    ax.text((x_wall_l + x_wall_r) / 2, y_cap, f'$C_{{wall}}={C_wall}$ kWh/°C',
            ha='center', va='top', fontsize=8, color='grey')
    ax.text((x_ext_l  + x_ext_r)  / 2, y_cap, r'$C_{ext}\to\infty$',
            ha='center', va='top', fontsize=8, color='grey')

    # Static: → ∞ arrow
    ax.annotate('', xy=(x_ext_r + 0.05, T_plot_max / 2),
                xytext=(x_ext_r - 0.3, T_plot_max / 2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                annotation_clip=False)

    # Mutable: energy fill rectangles
    rect_air = mpatches.Rectangle(
        (x_air_l, 0), x_air_r - x_air_l, T_air,
        facecolor='red', alpha=0.35, zorder=2)
    rect_wall = mpatches.Rectangle(
        (x_wall_l, 0), x_wall_r - x_wall_l, T_wall,
        facecolor='sienna', alpha=0.22, zorder=2)
    ax.add_patch(rect_air)
    ax.add_patch(rect_wall)

    # Mutable: temperature dashed lines
    line_air,  = ax.plot([x_air_l,  x_air_r],  [T_air,  T_air],
                         color='red',    linestyle='--', linewidth=1.5, zorder=3)
    line_wall, = ax.plot([x_wall_l, x_wall_r], [T_wall, T_wall],
                         color='sienna', linestyle='--', linewidth=1.5, zorder=3)

    # Mutable: temperature labels
    text_air  = ax.text(x_air_r  * 0.98,  T_air  + 0.4, f'{T_air:.1f} °C',
                        color='red',    ha='right', va='bottom', fontsize=10)
    text_wall = ax.text(x_wall_r * 0.997, T_wall + 0.4, f'{T_wall:.1f} °C',
                        color='sienna', ha='right', va='bottom', fontsize=10)

    # Mutable: time title
    title_text = ax.text(0.5, 1.02, 't = 0 min', transform=ax.transAxes,
                         ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Temperature (°C)', fontsize=11)
    ax.set_xlabel('Cumulative thermal capacity (kWh/°C)', fontsize=11)
    ax.set_xticks([x_air_l, x_air_r, x_wall_r])
    ax.set_xticklabels(['0', f'{C_air}', f'{C_air + C_wall:.2f}'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()

    return dict(
        fig=fig, ax=ax,
        rect_air=rect_air,   rect_wall=rect_wall,   rect_ext=rect_ext,
        line_air=line_air,   line_wall=line_wall,   line_ext=line_ext,
        text_air=text_air,   text_wall=text_wall,   text_ext=text_ext,
        title_text=title_text,
        x_air_l=x_air_l,   x_air_r=x_air_r,
        x_wall_l=x_wall_l, x_wall_r=x_wall_r,
        x_ext_l=x_ext_l,   x_ext_r=x_ext_r,
    )


def plot_thermal_schematic(
    T_air: float = 22.0,
    T_wall: float = 18.0,
    T_ext: float = 5.0,
    C_air: float = 0.26,
    C_wall: float = 19.1,
    T_min: float = 18.0,
    T_max: float = 24.0,
) -> Figure:
    h = _build_thermal_schematic(T_air, T_wall, T_ext, C_air, C_wall, T_min, T_max)
    return h['fig']


def animate_thermal_schematic(
    T_air_0: float = 22.0,
    T_wall_0: float = 18.0,
    T_ext: float = 5.0,
    C_air: float = 0.26,
    C_wall: float = 19.1,
    R_int: float = 0.42,
    R_ext: float = 8.86,
    Q_heater: float = 2.0,
    T_min: float = 18.0,
    T_max: float = 24.0,
    dt_min: float = 5.0,
    duration_h: float = 2.0,
    save_path: str | None = None,
) -> 'matplotlib.animation.FuncAnimation':
    """
    Animate the thermal schematic over a heating period using forward Euler.

    Parameters match the 2R2C model: R_int/R_ext in °C/kW, C_air/C_wall in kWh/°C,
    Q_heater in kW, dt_min in minutes, duration_h in hours.
    """
    from matplotlib.animation import FuncAnimation

    # Forward Euler trajectory
    dt = dt_min / 60.0
    n_steps = int(round(duration_h / dt))
    states = [(T_air_0, T_wall_0)]
    T_air, T_wall = T_air_0, T_wall_0
    for _ in range(n_steps):
        dT_air  = (T_wall - T_air)  / (R_int * C_air)  + Q_heater / C_air
        dT_wall = (T_air  - T_wall) / (R_int * C_wall) + (T_ext - T_wall) / (R_ext * C_wall)
        T_air  += dt * dT_air
        T_wall += dt * dT_wall
        states.append((T_air, T_wall))

    h = _build_thermal_schematic(T_air_0, T_wall_0, T_ext, C_air, C_wall, T_min, T_max)
    fig        = h['fig']
    rect_air   = h['rect_air']
    rect_wall  = h['rect_wall']
    line_air   = h['line_air']
    line_wall  = h['line_wall']
    text_air   = h['text_air']
    text_wall  = h['text_wall']
    title_text = h['title_text']
    x_air_l, x_air_r   = h['x_air_l'], h['x_air_r']
    x_wall_l, x_wall_r = h['x_wall_l'], h['x_wall_r']

    def update(frame):
        Ta, Tw = states[frame]
        rect_air.set_height(Ta)
        rect_wall.set_height(Tw)
        line_air.set_ydata([Ta, Ta])
        line_wall.set_ydata([Tw, Tw])
        text_air.set_position((x_air_r * 0.98, Ta + 0.4))
        text_air.set_text(f'{Ta:.1f} °C')
        text_wall.set_position((x_wall_r * 0.997, Tw + 0.4))
        text_wall.set_text(f'{Tw:.1f} °C')
        title_text.set_text(f't = {frame * dt_min:.0f} min')
        return rect_air, rect_wall, line_air, line_wall, text_air, text_wall, title_text

    anim = FuncAnimation(fig, update, frames=len(states), interval=150, blit=False)

    if save_path is not None:
        writer = 'ffmpeg' if str(save_path).endswith('.mp4') else 'pillow'
        anim.save(save_path, writer=writer, fps=8)
        print(f'Saved animation to {save_path}')

    return anim


def animate_thermal_schematic_from_data(
    state_df: pd.DataFrame,
    C_air: float = 0.26,
    C_wall: float = 19.1,
    T_min: float = 18.0,
    T_max: float = 24.0,
    stride: int = 6,
    save_path: str | None = None,
) -> 'matplotlib.animation.FuncAnimation':
    """
    Animate the thermal schematic using recorded simulation data.

    stride=6 at 5-min resolution gives one frame per 30 minutes.
    state_df must have columns: InteriorTemperature, WallTemperature,
    ExteriorTemperature, HeaterOutput, Price.
    """
    from matplotlib.animation import FuncAnimation

    df = state_df.iloc[::stride].reset_index()
    T_air_0  = df['InteriorTemperature'].iloc[0]
    T_wall_0 = df['WallTemperature'].iloc[0]
    T_ext_0  = df['ExteriorTemperature'].iloc[0]

    h = _build_thermal_schematic(T_air_0, T_wall_0, T_ext_0, C_air, C_wall, T_min, T_max)
    fig        = h['fig']
    rect_air   = h['rect_air'];   rect_wall  = h['rect_wall'];   rect_ext  = h['rect_ext']
    line_air   = h['line_air'];   line_wall  = h['line_wall'];   line_ext  = h['line_ext']
    text_air   = h['text_air'];   text_wall  = h['text_wall'];   text_ext  = h['text_ext']
    title_text = h['title_text']
    x_air_l, x_air_r   = h['x_air_l'], h['x_air_r']
    x_wall_l, x_wall_r = h['x_wall_l'], h['x_wall_r']
    x_ext_l,  x_ext_r  = h['x_ext_l'],  h['x_ext_r']

    def update(i):
        row = df.iloc[i]
        Ta  = row['InteriorTemperature']
        Tw  = row['WallTemperature']
        Te  = row['ExteriorTemperature']
        ts  = row['Timestamp']
        heat = row['HeaterOutput'] * 100
        price = row['Price']

        rect_air.set_height(Ta)
        rect_wall.set_height(Tw)
        rect_ext.set_height(Te)

        line_air.set_ydata([Ta, Ta])
        line_wall.set_ydata([Tw, Tw])
        line_ext.set_ydata([Te, Te])

        text_air.set_position((x_air_r * 0.98, Ta + 0.4))
        text_air.set_text(f'{Ta:.1f} °C')
        text_wall.set_position((x_wall_r * 0.997, Tw + 0.4))
        text_wall.set_text(f'{Tw:.1f} °C')
        text_ext.set_position((x_ext_r * 0.998, Te + 0.4))
        text_ext.set_text(f'{Te:.1f} °C')

        title_text.set_text(
            f"{ts.strftime('%a %d %b  %H:%M')}   |   "
            f"Heater: {heat:3.0f}%   |   "
            f"Price: {price:.2f} €/kWh"
        )
        return (rect_air, rect_wall, rect_ext,
                line_air, line_wall, line_ext,
                text_air, text_wall, text_ext, title_text)

    anim = FuncAnimation(fig, update, frames=len(df), interval=150, blit=False)

    if save_path is not None:
        writer = 'ffmpeg' if str(save_path).endswith('.mp4') else 'pillow'
        anim.save(save_path, writer=writer, fps=8)
        print(f'Saved animation to {save_path}')

    return anim