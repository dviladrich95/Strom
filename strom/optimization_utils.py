import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import requests
from datetime import datetime

import os
import xml.etree.ElementTree as ET
from entsoe import EntsoePandasClient

from strom import data_utils
from strom import api_utils

def get_temp_price_df():
    temp_df = data_utils.get_temp_series()
    prices_df = data_utils.get_price_series()
    temp_price_df = data_utils.join_data(temp_df, prices_df)
    return temp_price_df

# parameters estimated from https://protonsforbreakfast.wordpress.com/2022/12/19/estimating-the-heat-capacity-of-my-house/
# C_ air = 0.15*C_walls
# define an object heating_parameters

class House:
    def __init__(self, C_air=0.56, C_walls=3.5, R_internal=1.0,
                R_external=6.06, Q_heater=2.0, min_temperature=18.0, 
                max_temperature=24.0, init_indoor_temp = 18.5,
                init_wall_temp = 20.0, freq='h'):
        
        self.C_air = C_air
        self.C_walls = C_walls
        self.R_internal = R_internal
        self.R_external= R_external
        self.Q_heater=Q_heater
        self.freq=freq
        self.min_temperature=min_temperature
        self.max_temperature=max_temperature
        self.init_indoor_temp = init_indoor_temp
        self.init_wall_temp = init_wall_temp


def find_heating_decision(temp_price_df, house, heating_mode):
    """
    Determines the optimal heating decision for a given day based on outdoor temperature and electricity price,
    using explicit Euler integration for thermal dynamics.
    """
    state_df = temp_price_df.copy()  # Make a copy of the dataframe
    if house.freq == 'min':
        state_df = state_df.resample('min').interpolate(method='cubic')
        dt = 1.0/60
    elif house.freq == 'h':
        dt = 1.0

    time_steps = len(state_df)
    outdoor_temperature = state_df["Temperature"]
    

    heater_state = cp.Variable(time_steps)
    constraints = [heater_state >= 0, heater_state <= 1]
    
    # Temperature variables
    indoor_temperature = cp.Variable(time_steps)
    wall_temperature = cp.Variable(time_steps)
    
    # Initial conditions assuming slightly warmer start
    constraints.append(indoor_temperature[0] == house.init_indoor_temp)  
    constraints.append(wall_temperature[0] == house.init_wall_temp)
    
    # Thermal dynamics constraints
    for t in range(time_steps - 1):
        heat_loss_air = (indoor_temperature[t] - wall_temperature[t]) / house.R_internal
        heat_loss_wall = (wall_temperature[t] - outdoor_temperature.iloc[t]) / house.R_external
        
        constraints.append(
            indoor_temperature[t + 1] == indoor_temperature[t] + dt * (house.Q_heater * heater_state[t] - heat_loss_air) / house.C_air
        )
        constraints.append(
            wall_temperature[t + 1] == wall_temperature[t] + dt * (heat_loss_air - heat_loss_wall) / house.C_walls
        )
    
    # Minimum and maximum temperature constraint
    constraints.append(indoor_temperature >= house.min_temperature)
    constraints.append(indoor_temperature <= house.max_temperature)
    
    # Objective function
    if heating_mode == "optimal":
        obj = cp.sum(cp.multiply(state_df["Price"], heater_state * house.Q_heater)) + 0.1*cp.norm(cp.diff(indoor_temperature), 1)
    elif heating_mode == "baseline":
        obj = cp.sum(cp.maximum(house.min_temperature-indoor_temperature, 0))
    objective = cp.Minimize(obj)
    
    # Solve optimization
    problem = cp.Problem(objective, constraints)
    problem.solve()

    #check that an optimal solution was found
    if problem.status == cp.OPTIMAL:
        #add the decision to the dataframe
        state_df['Decision'] = heater_state.value
        state_df['Indoor Temperature'] = indoor_temperature.value
        state_df['Cost'] = state_df['Price'] * state_df['Decision']
    else:
        print("No optimal solution found with parameters: C_air={}, C_walls={}, R_internal={}, R_external={}, Q_heater={}, dt={}, min_temperature={}."
                        .format(house.C_air, house.C_walls, house.R_internal, house.R_external, house.Q_heater, dt, house.min_temperature))
        # fill with NaN arrays
        state_df['Decision'] = np.full(time_steps, np.nan)
        state_df['Indoor Temperature'] = np.full(time_steps, np.nan)
        state_df['Cost'] = np.full(time_steps, np.nan)


    
    return state_df


def compare_decision_costs(temp_price_df,house):
        
    """
    units will use kW and kWh
    """

    optimal_state_df  = find_heating_decision(temp_price_df, house, "optimal")
    #baseline_state_df = find_heating_decision(temp_price_df, house, "hybrid")
    baseline_state_df = find_heating_decision(temp_price_df, house, "baseline")

    return optimal_state_df, baseline_state_df

def get_state_df(temp_price_df, decision, house):
    
    # time steps is the length of the time index resulting from starting at the first time and going to the last time in steps of dt
    state_df = temp_price_df.copy()  # Make a copy of the dataframe
    state_df['Decision'] = decision
    if house.freq == 'min':
        state_df = state_df.resample('min').interpolate(method='cubic')
        dt = 1.0/60
    elif house.freq == 'h':
        dt = 1.0
    
    time_steps = len(state_df)

    indoor_temperature = np.zeros(time_steps)
    wall_temperature = np.zeros(time_steps)

    indoor_temperature[0] = house.init_indoor_temp
    wall_temperature[0] = house.init_wall_temp

    for t in range(time_steps - 1):
        heat_loss_air = (indoor_temperature[t] - wall_temperature[t]) / house.R_internal
        heat_loss_wall = (wall_temperature[t] - state_df['Temperature'][t]) / house.R_external
        
        indoor_temperature[t + 1] = indoor_temperature[t] + dt * (house.Q_heater * state_df['Decision'][t] - heat_loss_air) / house.C_air
        wall_temperature[t + 1] = wall_temperature[t] + dt * (heat_loss_air - heat_loss_wall) / house.C_walls

    # Calculate the cost of the baseline decision
    state_df['Cost'] = state_df['Price'] * state_df['Decision']
    state_df['Indoor Temperature'] = indoor_temperature

    return state_df

def plot_state(state_df, case_label, plot_price=True):
    """
    Plots the costs, temperatures, and heater state for a single case (e.g., Baseline or Optimal).
    Args:
        compare_df (pd.DataFrame): DataFrame containing the costs, temperatures, and heater state.
        case_label (str): Label for the case being plotted (e.g., 'Baseline' or 'Optimal').
    Returns:
        fig, ax1, ax2, ax3: Matplotlib figure and axes objects.
    """
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel(f'{case_label} Cost (€)', color=color)
    ax1.plot(state_df['Cost'].cumsum(), color=color, linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', rotation=45)  # Rotate x-axis tick labels

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel(f'{case_label} Indoor Temperature', color=color)
    ax2.plot(state_df['Indoor Temperature'], color=color, linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax1.twinx()
    color = 'tab:green'
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(state_df['Decision'], color=color, linestyle='-')
    ax3.set_ylabel(f'{case_label} Heater State', color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    if plot_price:
        ax4 = ax1.twinx()
        color = 'tab:grey'
        ax4.spines['right'].set_position(('outward', 120))
        ax4.plot(state_df['Price'], color=color, linestyle='--')
        ax4.set_ylabel('Price (€/kWh)', color=color)
        ax4.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    return fig 

def plot_combined_cases(state_opt_df,state_base_df):
    #save the plots in the plots folder
    os.chdir(api_utils.find_root_dir())

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Cost (€)', color=color)
    ax1.plot(state_opt_df['Cost'].cumsum(), color=color, linestyle='-')
    ax1.plot(state_base_df['Cost'].cumsum(), color=color, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', rotation=45)  # Rotate x-axis tick labels

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Indoor Temperature (°C)', color=color)  # we already handled the x-label with ax1
    ax2.plot(state_opt_df['Indoor Temperature'], color=color, linestyle='-')
    ax2.plot(state_base_df['Indoor Temperature'], color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.tick_params(axis='x', rotation=45)  # Rotate x-axis tick labels


    ax3 = ax1.twinx()
    color = 'tab:green'
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(state_opt_df['Decision'], color=color, linestyle='-')
    ax3.plot(state_base_df['Decision'], color=color, linestyle='--')
    ax3.set_ylabel('Heater State', color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    # add a third line in yellow with the electricity price
    ax4 = ax1.twinx()
    # make the color a pale blue
    color = 'tab:grey'

    ax4.spines['right'].set_position(('outward', 120))
    ax4.plot(state_opt_df['Price'], color=color)
    ax4.set_ylabel('Price (€/kWh)', color=color)
    ax4.tick_params(axis='y', labelcolor=color)

    # Add legends
    # Add legends outside the plot area
    ax1.legend(['Optimal Cost', 'Baseline Cost'], loc='upper left', bbox_to_anchor=(-0.2, 1.2))
    ax2.legend(['Optimal Temperature', 'Baseline Temperature'], loc='upper left', bbox_to_anchor=(0.35, 1.2))
    ax3.legend(['Optimal Heater State', 'Baseline Heater State'], loc='upper left', bbox_to_anchor=(0.5, 1.2))
    ax4.legend(['Price'], loc='upper left', bbox_to_anchor=(0.75, 1.2))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

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