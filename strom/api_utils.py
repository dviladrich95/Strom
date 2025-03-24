import os
from pathlib import Path
import requests
from datetime import datetime
import pandas as pd
from entsoe import EntsoePandasClient
from bs4 import XMLParsedAsHTMLWarning
import warnings

# Filter XML parser warning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

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

def get_weather_data():
    """
    Fetches weather data for Barcelona from the OpenWeatherMap API, processes it, and returns a DataFrame with hourly temperature data.
    The function performs the following steps:
    1. Reads the OpenWeatherMap API key from an environment variable or a configuration file.
    2. Makes an API call to fetch the weather forecast data for Barcelona.
    3. Parses the JSON response to extract timestamps and temperatures.
    4. Converts the timestamps to the 'Europe/Madrid' timezone.
    5. Converts temperatures from Kelvin to Celsius.
    6. Interpolates missing data points to generate an hourly temperature DataFrame.
    7. Ensures the DataFrame covers exactly a 24-hour range starting from the current hour.
    Returns:
        pd.DataFrame: A DataFrame with two columns:
            - 'Timestamp': Timestamps in the 'Europe/Madrid' timezone.
            - 'Temperature (°C)': Temperatures in Celsius.
    """
    time_steps = 24  # 24 hours in a day

    # Get the API key from environment variable or config file
    API_KEY = os.getenv('WEATHER_API_KEY')
    if not API_KEY:
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
    temp_df = temp_df.set_index('Timestamp').reindex(full_range).interpolate()
    temp_df.columns = ['Temperature (°C)']  # Rename columns

    # if there are nan values, fill them with the previous value
    if temp_df.isnull().values.any():
        print(f"Warning: There are NaN values in the DataFrame, filling in {temp_df.isnull().sum()} NaN values.")
        temp_df.bfill(inplace=True)
    return temp_df

def get_prices():
    """
    Fetches the day-ahead electricity prices for Spain using the EntsoePandasClient.
    This function retrieves the API key from an environment variable or a specified configuration file, initializes the 
    EntsoePandasClient with the API key, and queries the day-ahead electricity prices for Spain 
    for the next 24 hours starting from the current time in the 'Europe/Madrid' timezone. The 
    resulting prices are returned as a pandas DataFrame with timestamps and corresponding prices.
    Returns:
        pandas.DataFrame: A DataFrame containing the timestamps and corresponding day-ahead 
        electricity prices for Spain, in kWh.
    Note:
        Ensure that the API key is correctly placed in the './config/price_api_key.txt' file as 
        specified in the readme.
    """
    # Get the API key from environment variable or config file
    price_api_key = os.getenv('PRICE_API_KEY')
    if not price_api_key:
        price_api_key = get_api_key('./config/price_api_key.txt')  # Please see readme to see how to create your config folder with the API key

    # Replace with your API key
    client = EntsoePandasClient(api_key=price_api_key)

    # Define the current timestamp (now) and timezone
    start = pd.Timestamp.now(tz='Europe/Madrid')  # Current time in Madrid timezone
    end = start + pd.Timedelta(hours=24)  # 24 hours after the current time

    #make a dataframe with one column for the timestamp and one for the price
    timestamp_index = pd.date_range(start=start, end=end, freq='h', tz='Europe/Madrid')

    # Country code for Spain
    country_code = 'ES'  # Spain

    # Querying the day-ahead prices for Spain
    prices_series = client.query_day_ahead_prices(country_code, start=start, end=end)

    # Convert the Series to a DataFrame and reindex to get exactly 24 hours
    prices_df = prices_series.to_frame(name='Price')
    prices_df = prices_df.reindex(timestamp_index, method='nearest').head(24)

    # divide the prices by 1000 to get the price in €/kWh
    prices_df['Price'] = prices_df['Price'] / 1000

    return prices_df