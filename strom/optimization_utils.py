import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import requests
import pandas as pd
import os
import xml.etree.ElementTree as ET
from entsoe import EntsoePandasClient

from .api_utils import find_root_dir, get_api_key, get_weather_data, get_prices
from .data_utils import get_temp_price_df

def join_data(temp_df, prices_df):
    """
    Merge temperature and price dataframes on the 'Timestamp' column and extract temperature and prices as numpy arrays.
    Parameters:
    temp_df (pd.DataFrame): DataFrame containing temperature data with a 'Timestamp' column.
    prices_df (pd.DataFrame): DataFrame containing price data with a 'Timestamp' column.
    Returns:
    pd.DataFrame: Merged DataFrame containing both temperature and price data.
    """
    
    # Reindex temp_df to match the timestamps of prices_df
    temp_df_reindexed = temp_df.reindex(prices_df.index.union(temp_df.index)).interpolate(method='time')

    # Fill any remaining NaN values after interpolation
    temp_df_reindexed = temp_df_reindexed.fillna(method='bfill').fillna(method='ffill')

    # Reindex again to match exactly the prices_df index
    temp_df_reindexed = temp_df_reindexed.reindex(prices_df.index)
    temp_price_df = pd.merge(temp_df_reindexed, prices_df, left_index=True, right_index=True, how='inner')

    return temp_price_df  # Returning the merged dataframe

def find_heating_decision(temp_price_df, type = "optimal", decision = 'relaxed',
                            heat_loss = 0.1,  # Heat loss rate per degree difference per hour
                            heating_power = 2,  # Heating rate (degrees per hour)
                            min_temperature = 18,  # Minimum temperature constraint (°C)
                          ):
    """
    Determines the optimal heating decision for a given day based on outdoor temperature and electricity price.
    Parameters:
    temp_price_df (pd.DataFrame): A DataFrame containing two columns:
        - "Temperature (°C)": Outdoor temperature for each hour of the day.
        - "Price": Electricity price for each hour of the day.
    Returns:
    array: The optimal state of the heater (on/off) throughout the day.
    """

    # Parameters
    time_steps = 24  # 24 hours in a day

    # Simulate outdoor temperature (cool at night, warm in the day)
    outdoor_temperature = temp_price_df["Temperature (°C)"]

    initial_temperature = min_temperature  # Initial temperature (°C)

    # Constraints
    constraints = []

    if decision == 'relaxed':
        # Decision variables
        heater_state = cp.Variable(time_steps)

        # Heater state constraint (continuous between 0 and 1)
        constraints.append(heater_state >= 0)
        constraints.append(heater_state <= 1)

    elif decision == 'discrete':
        # Decision variables
        heater_state = cp.Variable(time_steps, boolean=True)

    indoor_temperature = cp.Variable(time_steps)

    # Objective: Minimize monetary cost (optimal) or temperature deviation from 20°C (baseline)
    if type == "optimal":
        cost = cp.sum(cp.multiply(temp_price_df["Price"], heater_state * heating_power))
    elif type == "baseline":
        cost = cp.sum(cp.abs(indoor_temperature - min_temperature))
    objective = cp.Minimize(cost)

    # Initial temperature constraint
    constraints.append(indoor_temperature[0] == initial_temperature)

    # Minimum temperature constraint
    constraints += [indoor_temperature >= min_temperature]

    # Thermal dynamics constraints
    for t in range(1, time_steps):
        heat_loss_effect = heat_loss * (indoor_temperature[t - 1] - outdoor_temperature[t - 1])
        constraints.append(
            indoor_temperature[t] == indoor_temperature[t - 1]
            + heater_state[t] * heating_power
            - heat_loss_effect
        )

    # Problem definition
    problem = cp.Problem(objective, constraints)

    # Solve the problem
    problem.solve()
    decision = heater_state.value
    indoor_temp = indoor_temperature.value
    return decision, indoor_temp

def compare_decision_costs(temp_price_df,
                            heat_loss = 0.1,  # Heat loss rate per degree difference per hour
                            heating_power = 2,  # Heating rate (degrees per hour)
                            min_temperature = 18,  # Minimum temperature constraint (°C)
                           ):
    """
    Compares the costs of the optimal and baseline heating decisions for a given day.
    Parameters:
    temp_price_df (pd.DataFrame): A DataFrame containing two columns:
        - "Temperature (°C)": Outdoor temperature for each hour of the day.
        - "Price": Electricity price for each hour of the day.
    Returns:
    tuple: A tuple containing the costs of the optimal and baseline heating decisions.
    """
    # Get the optimal heating decision
    optimal_decision, optimal_indoor_temperature  = find_heating_decision(temp_price_df, type = "optimal",
                                                                        heat_loss = heat_loss,
                                                                        heating_power = heating_power,
                                                                        min_temperature = min_temperature)
    
    baseline_decision, baseline_indoor_temperature = find_heating_decision(temp_price_df, type = "baseline",
                                                                        heat_loss = heat_loss,
                                                                        heating_power = heating_power,
                                                                        min_temperature = min_temperature)

    # Calculate the cost of the optimal decision
    optimal_cost= temp_price_df["Price"] * optimal_decision

    # Calculate the cost of the baseline decision
    baseline_cost = temp_price_df["Price"] * baseline_decision

    # merge all the 4 data into one dataframe
    compare_df = pd.DataFrame({
        'Optimal Cost': optimal_cost,
        'Baseline Cost': baseline_cost,
        'Optimal Indoor Temperature': optimal_indoor_temperature,
        'Baseline Indoor Temperature': baseline_indoor_temperature,
        'Price': temp_price_df['Price']
    })

    return compare_df



def plot_costs_and_temps(compare_df):
    """
    Plots the costs of the optimal and baseline decisions and temperatures with two different axes.
    Args:
        compare_df (pd.DataFrame): DataFrame containing the optimal and baseline costs and temperatures.
    """

    #save the plots in the plots folder
    os.chdir(find_root_dir())

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Cost (€)', color=color)
    ax1.plot(compare_df['Optimal Cost'].cumsum(), color=color, linestyle='-')
    ax1.plot(compare_df['Baseline Cost'].cumsum(), color=color, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', rotation=45)  # Rotate x-axis tick labels

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Indoor Temperature (°C)', color=color)  # we already handled the x-label with ax1
    ax2.plot(compare_df['Optimal Indoor Temperature'], color=color, linestyle='-')
    ax2.plot(compare_df['Baseline Indoor Temperature'], color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.tick_params(axis='x', rotation=45)  # Rotate x-axis tick labels

    # add a third line in yellow with the electricity price
    ax3 = ax1.twinx()
    # make the color a pale blue
    color = 'tab:grey'

    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(compare_df['Price'], color=color)
    ax3.set_ylabel('Price (€/kWh)', color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.subplots_adjust(top=0.85) 

    # Add legends
    # Add legends outside the plot area
    ax1.legend(['Optimal Cost', 'Baseline Cost'], loc='upper left', bbox_to_anchor=(0.25, 1.2))
    ax2.legend(['Optimal Temperature', 'Baseline Temperature'], loc='upper left', bbox_to_anchor=(0.65, 1.2))
    ax3.legend(['Price'], loc='upper left', bbox_to_anchor=(0.0, 1.2))

    # Save the plot
    plt.savefig('./plots/compare_costs_temps.png', bbox_inches='tight')

    plt.show()