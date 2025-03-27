import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

import requests
from datetime import datetime

import os
import xml.etree.ElementTree as ET

from entsoe import EntsoePandasClient

def find_root_dir(target_folder="Strom"):
    """
    Recursively searches for the specified target folder in the current directory
    and its parent directories, returning the absolute path to the target folder
    if found.
    Args:
        target_folder (str): The name of the folder to search for. Defaults to "Strom".
    Returns:
        str: The absolute path to the target folder if found.
    Raises:
        FileNotFoundError: If the target folder is not found in the current directory
                           or any of its parent directories.
    """
    current_dir = os.getcwd()
    
    while True:
        if target_folder in os.listdir(current_dir):
            return os.path.abspath(os.path.join(current_dir, target_folder))
        
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached the filesystem root
            raise FileNotFoundError(f"'{target_folder}' folder not found in any parent directories.")
        
        current_dir = parent_dir

def get_api_key(key_path):
    """
    Reads an API key from a file.

    Args:
        key_path (str): The path to the file containing the API key.

    Returns:
        str: The API key read from the file.
    """
    with open(key_path, 'r') as file:
        api_key = file.read().strip()  # Read the file
    return api_key

def regularize_df(df):

    df_resamp = df.resample('1h').asfreq()
    # make a new dataframe without any columns
    df_resamp = df_resamp.drop(columns = df_resamp.columns)

    #merge dataframes
    merged_df = pd.merge(df_resamp, df, left_index=True, right_index=True, how='outer')
    #interpolate the missing values
    merged_df = merged_df.interpolate(method='cubic')

    #extrapolate the missing values
    merged_df = merged_df.interpolate(method='cubic', limit_direction='both')
    # take only the rows with indices present in the resampled dataframe
    merged_df = merged_df[merged_df.index.isin(df_resamp.index)]

    #remove rows with Nan
    merged_df = merged_df.dropna()
    return merged_df

def get_temp_series():
    """
    Fetches weather data for Barcelona from the OpenWeatherMap API, processes it, and returns a DataFrame with hourly temperature data.
    The function performs the following steps:
    1. Reads the OpenWeatherMap API key from a configuration file.
    2. Makes an API call to fetch the weather forecast data for Barcelona.
    3. Parses the JSON response to extract timestamps and temperatures.
    4. Converts the timestamps to the 'Europe/Madrid' timezone.
    5. Converts temperatures from Kelvin to Celsius.
    6. Interpolates missing data points to generate an hourly temperature DataFrame.
    7. Ensures the DataFrame covers exactly a 24-hour range starting from the current hour.
    Returns:
        pd.DataFrame: A DataFrame with two columns:
            - 'Timestamp': Timestamps in the 'Europe/Madrid' timezone.
            - 'Temperature': Temperatures in Celsius.
    """

    # open weather map API key text file
    os.chdir(find_root_dir())
    with open('./config/weather_api_key.txt') as f:
        API_KEY = f.read().strip()

    call_str = "https://api.openweathermap.org/data/2.5/forecast?q=Barcelona&appid=" + API_KEY

    # Make the API call
    response = requests.get(call_str)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        data = response.json()  # Parse the JSON response

        # Prepare lists for timestamps and temperatures
        timestamps = [
            pd.Timestamp(entry['dt'], unit='s', tz='UTC').tz_convert('Europe/Madrid')  # Convert to Madrid timezone
            for entry in data['list']
        ]
        temperatures = [entry['main']['temp'] for entry in data['list']]

        # Create a DataFrame
        temp_series = pd.Series(
            data=np.array(temperatures) - 273.15,  # Convert Kelvin to Celsius
            index=pd.to_datetime(timestamps, utc=True).tz_convert('Europe/Madrid'),
            name='Temperature')

    else:
        print(f"Error: {response.status_code}")

    return temp_series

def get_price_series():


    price_api_key = get_api_key('./config/price_api_key.txt')  # Please see readme to see how to create your config folder with the API key

    # Replace with your API key
    client = EntsoePandasClient(api_key=price_api_key)

    # Define the current timestamp (now) and timezone
    start = pd.Timestamp.now(tz='Europe/Madrid').round('min')  # Current time in Madrid timezone, rounded to the nearest minute
    end = start + pd.Timedelta(hours=24)  # 24 hours after the current time

    # Country code for Spain
    country_code = 'ES'  # Spain

    # Querying the day-ahead prices for Spain
    price_series = client.query_day_ahead_prices(country_code, start=start, end=end)

    # Change series name to Prices
    price_series.name = 'Price'

    # divide the prices by 1000 to convert the price from (€/MWh) to (€/kWh)
    price_series = price_series / 1000.0

    return price_series

def join_data(temp_series, price_series):
    """
    Merge temperature and price dataframes on the 'Timestamp' column and extract temperature and prices as numpy arrays.
    Parameters:
    temp_df (pd.DataFrame): DataFrame containing temperature data with a 'Timestamp' column.
    prices_df (pd.DataFrame): DataFrame containing price data with a 'Timestamp' column.
    Returns:
    pd.DataFrame: Merged DataFrame containing both temperature and price data.
    """
    
    # make one dataframe from the two series
    temp_df = temp_series.to_frame()
    prices_df = price_series.to_frame()

    # Merge the two dataframes on the 'Timestamp' column
    temp_price_df = pd.merge(temp_df, prices_df, left_index=True, right_index=True,how='outer')

    # regularize the dataframe
    temp_price_df = regularize_df(temp_price_df)

    return temp_price_df  # Returning the merged dataframe

def get_temp_price_df():
    temp_df = get_temp_series()
    prices_df = get_price_series()
    temp_price_df = join_data(temp_df, prices_df)
    return temp_price_df

def find_heating_decision(temp_price_df, type="optimal", decision='relaxed',
                          C_air=120000, C_walls=750000,
                          R_internal=0.01, R_external=0.05,
                          Q_heater=2000, freq='h',
                          min_temperature=18, max_temperature=24):
    """
    Determines the optimal heating decision for a given day based on outdoor temperature and electricity price,
    using explicit Euler integration for thermal dynamics.
    """
    state_df = temp_price_df.copy()  # Make a copy of the dataframe
    if freq == 'min':
        state_df.resample('min').interpolate(method='cubic')
        dt = 60
    elif freq == 'h':
        dt = 3600

    time_steps = len(state_df)
    outdoor_temperature = state_df["Temperature"]
    
    # Decision variables
    if decision == 'relaxed':
        heater_state = cp.Variable(time_steps)
        constraints = [heater_state >= 0, heater_state <= 1]
    elif decision == 'discrete':
        heater_state = cp.Variable(time_steps, boolean=True)
        constraints = []
    
    # Temperature variables
    indoor_temperature = cp.Variable(time_steps)
    wall_temperature = cp.Variable(time_steps)
    
    # Initial conditions assuming slightly warmer start
    constraints.append(indoor_temperature[0] == min_temperature + 0.5)  
    constraints.append(wall_temperature[0] == min_temperature + 0.5)
    
    # Thermal dynamics constraints
    for t in range(time_steps - 1):
        heat_loss_air = (indoor_temperature[t] - wall_temperature[t]) / R_internal
        heat_loss_wall = (wall_temperature[t] - outdoor_temperature.iloc[t]) / R_external
        
        constraints.append(
            indoor_temperature[t + 1] == indoor_temperature[t] + dt * (Q_heater * heater_state[t] - heat_loss_air) / C_air
        )
        constraints.append(
            wall_temperature[t + 1] == wall_temperature[t] + dt * (heat_loss_air - heat_loss_wall) / C_walls
        )
    
    # Minimum and maximum temperature constraint
    constraints.append(indoor_temperature >= min_temperature)
    constraints.append(indoor_temperature <= max_temperature)
    
    # Objective function
    if type == "optimal":
        cost = cp.sum(cp.multiply(state_df["Price"], heater_state * Q_heater*3600/1000))
    elif type == "baseline":
        #cost = cp.sum((cp.abs(min_temperature-indoor_temperature) + (min_temperature-indoor_temperature)) / 2)
        cost = cp.sum(cp.maximum(min_temperature-indoor_temperature, 0))
        #cost = cp.sum((cp.abs(min_temperature-indoor_temperature) + (min_temperature-indoor_temperature)) / 2)
    objective = cp.Minimize(cost)
    
    # Solve optimization
    problem = cp.Problem(objective, constraints)
    problem.solve()

    #check that an optimal solution was found
    if problem.status != cp.OPTIMAL:
        raise Exception("No optimal solution found with parameters: C_air={}, C_walls={}, R_internal={}, R_external={}, Q_heater={}, dt={}, min_temperature={}.".format(C_air, C_walls, R_internal, R_external, Q_heater, dt, min_temperature))

    #add the decision to the dataframe
    state_df['Decision'] = heater_state.value
    state_df['Indoor Temperature'] = indoor_temperature.value
    state_df['Cost'] = state_df['Price'] * state_df['Decision']
    
    return state_df


def compare_decision_costs(temp_price_df,
                            C_air=120000, C_walls=750000,
                            R_internal=0.01, R_external=0.05,
                            Q_heater=2000, freq='h',
                            min_temperature=18):
        
    """
    units will use kW and kWh
    """

    # Get the optimal heating decision
    optimal_state_df  = find_heating_decision(temp_price_df, type="optimal",
                                                                            C_air=C_air, C_walls=C_walls,
                                                                            R_internal=R_internal, R_external=R_external,
                                                                            Q_heater=Q_heater, freq=freq,
                                                                            min_temperature=min_temperature)
    
    baseline_state_df = find_heating_decision(temp_price_df, type="baseline",
                                                                            C_air=C_air, C_walls=C_walls,
                                                                            R_internal=R_internal, R_external=R_external,
                                                                            Q_heater=Q_heater, freq=freq,
                                                                            min_temperature=min_temperature)

    return optimal_state_df, baseline_state_df

def get_state_df(temp_price_df, decision,
                            C_air=120000, C_walls=750000,
                            R_internal=0.01, R_external=0.05,
                            Q_heater=2000,  freq='h',
                            min_temperature=18):
    
    # time steps is the length of the time index resulting from starting at the first time and going to the last time in steps of dt
    state_df = temp_price_df.copy()  # Make a copy of the dataframe
    state_df['Decision'] = decision
    if freq == 'min':
        state_df = state_df.resample('min').interpolate(method='cubic')
        dt = 60
    elif freq == 'h':
        dt = 3600
    
    time_steps = len(state_df)

    indoor_temperature = np.zeros(time_steps)
    wall_temperature = np.zeros(time_steps)

    indoor_temperature[0] = min_temperature + 0.5
    wall_temperature[0] = min_temperature + 0.5

    for t in range(time_steps - 1):
        heat_loss_air = (indoor_temperature[t] - wall_temperature[t]) / R_internal
        heat_loss_wall = (wall_temperature[t] - state_df['Temperature'][t]) / R_external
        
        indoor_temperature[t + 1] = indoor_temperature[t] + dt * (Q_heater * state_df['Decision'][t] - heat_loss_air) / C_air
        wall_temperature[t + 1] = wall_temperature[t] + dt * (heat_loss_air - heat_loss_wall) / C_walls

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
    return fig, ax1, ax2, ax3


def plot_combined_cases(fig1, fig2, state_df1, state_df2):
    """
    Combines the plots for both Baseline and Optimal cases into a single plot using the generic plot output.
    Args:
        fig1: Matplotlib figure object for the first case (e.g., Baseline).
        fig2: Matplotlib figure object for the second case (e.g., Optimal).
    Returns:
        fig: Matplotlib figure object with combined plots.
    """
    # Create a new figure
    fig, ax1 = plt.subplots()

    # Extract data from fig1 and fig2
    for ax in fig1.axes:
        for line in ax.get_lines():
            ax1.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), linestyle='-')

    for ax in fig2.axes:
        for line in ax.get_lines():
            ax1.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), linestyle='--')

    # Add legend
    ax1.legend()

    # Set labels and title
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Values')
    ax1.set_title('Combined Baseline and Optimal Cases')

    return fig