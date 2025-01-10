import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

import requests
from datetime import datetime
import pandas as pd

def call_api():
    price = 1
    return price

def get_api_key(key_path):
    with open(key_path, 'r') as file:
        api_key = file.read().strip()  # Read the file
    return api_key

def get_weather_data():
    time_steps = 24  # 24 hours in a day

    #open weather map api key text file at the path config/weather_api_key.txt
    with open('../config/weather_api_key.txt') as f:
        API_KEY = f.read().strip()

    call_str = "https://api.openweathermap.org/data/2.5/forecast?q=Barcelona&appid="+API_KEY

    # Make the API call
    response = requests.get(call_str)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        data = response.json()  # Parse the JSON response

        # Prepare lists for timestamps and temperatures
        timestamps = [datetime.utcfromtimestamp(entry['dt']) for entry in data['list']]
        temperatures = [entry['main']['temp'] for entry in data['list']]

        # Create a DataFrame
        df = pd.DataFrame({
            'Timestamp': timestamps,
            'Temperature (K)': temperatures
        })

        # Convert temperature from Kelvin to Celsius
        df['Temperature (°C)'] = df['Temperature (K)'] - 273.15
        # remove the temperature in Kelvin
        df = df.drop(columns=['Temperature (K)'])
    else:
        print(f"Error: {response.status_code}")

    # this is a 3 hour forecast, so we have 8 data points per day, interpolate the missing data points
    df = df.set_index('Timestamp').resample('h').interpolate().reset_index()

    # trim the dataframe to only include the first 24 hours
    df = df.head(time_steps)
    return df

def data_analysis(prices):

    # Parameters
    time_steps = 24  # 24 hours in a day
    hours = np.arange(time_steps)

    # Simulate outdoor temperature (cool at night, warm in the day)
    outdoor_temperature = 10 + -5 * np.cos(2 * np.pi * hours / 24)

    # Thermal properties
    heat_loss = 0.1  # Heat loss rate per degree difference per hour
    heating_power = 2  # Heating rate (degrees per hour)
    min_temperature = 18  # Minimum temperature constraint (°C)
    initial_temperature = 20  # Initial temperature (°C)

    # Decision variables
    heater_state = cp.Variable(time_steps, boolean=True)
    indoor_temperature = cp.Variable(time_steps)

    # Objective: Minimize cost
    cost = cp.sum(cp.multiply(prices, heater_state * heating_power))
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
    decision = heater_state.value

    return decision

def automation_kasa(decision):
    status = []
    return status