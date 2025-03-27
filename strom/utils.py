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
        state_df.resample('min').interpolate(method='cubic')
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
        obj = cp.sum(cp.multiply(state_df["Price"], heater_state * house.Q_heater))
    elif heating_mode == "baseline":
        #cost = cp.sum((cp.abs(min_temperature-indoor_temperature) + (min_temperature-indoor_temperature)) / 2)
        obj = cp.sum(cp.maximum(house.min_temperature-indoor_temperature, 0))
        #cost = cp.sum((cp.abs(min_temperature-indoor_temperature) + (min_temperature-indoor_temperature)) / 2)
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