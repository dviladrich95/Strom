import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

import requests
from datetime import datetime
import pandas as pd
import os
import xml.etree.ElementTree as ET

from entsoe import EntsoePandasClient
import pandas as pd

def find_root_dir(target_folder="Strom"):
    """
    Traverse up the directory tree to find the target folder and return its absolute path.
    """
    current_dir = os.getcwd()
    
    while True:
        if target_folder in os.listdir(current_dir):
            return os.path.abspath(os.path.join(current_dir, target_folder))
        
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached the filesystem root
            raise FileNotFoundError(f"'{target_folder}' folder not found in any parent directories.")
        
        current_dir = parent_dir

def call_api():
    price = 1
    return price

def get_api_key(key_path):
    with open(key_path, 'r') as file:
        api_key = file.read().strip()  # Read the file
    return api_key

def get_weather_data():
    time_steps = 24  # 24 hours in a day

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
        temp_df = pd.DataFrame({
            'Timestamp': timestamps,
            'Temperature (K)': temperatures
        })

        # Convert temperature from Kelvin to Celsius
        temp_df['Temperature (°C)'] = temp_df['Temperature (K)'] - 273.15
        # Remove the temperature in Kelvin
        temp_df = temp_df.drop(columns=['Temperature (K)'])
    else:
        print(f"Error: {response.status_code}")

    # This is a 3-hour forecast, so we have 8 data points per day, interpolate the missing data points
    temp_df = temp_df.set_index('Timestamp').resample('h').interpolate().reset_index()

    # Generate exactly 24-hour range
    start_time = pd.Timestamp.now(tz='Europe/Madrid').floor('h')
    end_time = start_time + pd.Timedelta(hours=time_steps)
    full_range = pd.date_range(start=start_time + pd.Timedelta(hours=1), end=end_time, freq='h', tz='Europe/Madrid')
    temp_df = temp_df.set_index('Timestamp').reindex(full_range).interpolate().reset_index()
    temp_df.columns = ['Timestamp', 'Temperature (°C)']  # Rename columns

    return temp_df

def get_prices():
    apikey = get_api_key('./config/apiKey.txt')  # Please see readme to see how to create your config folder with the API key

    # Replace with your API key
    client = EntsoePandasClient(api_key=apikey)

    # Define the current timestamp (now) and timezone
    start = pd.Timestamp.now(tz='Europe/Madrid')  # Current time in Madrid timezone
    end = start + pd.Timedelta(hours=24)  # 24 hours after the current time

    # Country code for Spain
    country_code = 'ES'  # Spain

    # Querying the day-ahead prices for Spain
    prices_series = client.query_day_ahead_prices(country_code, start=start, end=end)

    # Convert the Series to a DataFrame and rename columns
    prices_df = prices_series.to_frame(name='Price').reset_index()
    prices_df.rename(columns={'index': 'Timestamp'}, inplace=True)

    return prices_df

def join_data(temp_df, prices_df):
    
    # Merge the dataframes on the 'Timestamp' column
    temp_price_df = pd.merge(temp_df, prices_df, on='Timestamp')

    # Extract the temperature and prices as numpy arrays
    temperature = temp_price_df['Temperature (°C)'].values
    prices = temp_price_df['Price'].values

    return temp_price_df  # Returning the merged dataframe

def find_optimal_heating_decision(temp_price_df):

    # Parameters
    time_steps = 24  # 24 hours in a day

    # Simulate outdoor temperature (cool at night, warm in the day)
    outdoor_temperature = temp_price_df["Temperature (°C)"]

    # Thermal properties
    heat_loss = 0.1  # Heat loss rate per degree difference per hour
    heating_power = 2  # Heating rate (degrees per hour)
    min_temperature = 18  # Minimum temperature constraint (°C)
    initial_temperature = 20  # Initial temperature (°C)

    # Decision variables
    heater_state = cp.Variable(time_steps, boolean=True)
    indoor_temperature = cp.Variable(time_steps)

    # Objective: Minimize cost
    cost = cp.sum(cp.multiply(temp_price_df["Price"], heater_state * heating_power))
    objective = cp.Minimize(cost)

    # Constraints
    constraints = []

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
    
    decision = heater_state.value[0]
    return decision

def automation_kasa(decision):
    status = []
    return status