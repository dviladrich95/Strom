from datetime import datetime
from pathlib import Path
from typing import Optional

import os
import warnings
import requests
import pandas as pd
from entsoe import EntsoePandasClient
from bs4 import XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

EXAMPLE_CITIES = [
    "Barcelona, ES", "Madrid, ES", "Berlin, DE", 
    "Paris, FR", "London, GB", "Rome, IT"
]

def find_root_dir(target_folder: str = "Strom") -> str:
    current_dir = os.getcwd()
    parent_dir = current_dir

    while parent_dir == current_dir or current_dir != '/':
        if target_folder in os.listdir(current_dir):
            return os.path.abspath(os.path.join(current_dir, target_folder))
        current_dir = os.path.dirname(current_dir)
    
    raise FileNotFoundError(f"'{target_folder}' folder not found in directory tree")

def read_api_key(key_path: str) -> str:
    return Path(key_path).read_text().strip()

def get_api_key(key_path: str) -> str:
    """Alias for read_api_key for backward compatibility"""
    return read_api_key(key_path)

def get_weather_api_key() -> str:
    api_key = os.getenv('WEATHER_API_KEY')
    if api_key:
        return api_key
        
    os.chdir(find_root_dir())
    return read_api_key('./config/weather_api_key.txt')

def get_weather_data(time_range: Optional[pd.DatetimeIndex] = None, city: str = "Barcelona, ES") -> pd.DataFrame:
    """Get weather for specified city. Examples: Barcelona, ES | Madrid, ES | Berlin, DE"""
    api_key = get_weather_api_key()
    
    try:
        response = requests.get(
            f"https://api.openweathermap.org/data/2.5/forecast",
            params={"q": city, "appid": api_key}
        )
        response.raise_for_status()
    except requests.RequestException as e:
        available_cities = "\n".join(f"- {city}" for city in EXAMPLE_CITIES)
        raise ValueError(
            f"Failed to fetch weather data for '{city}'. Examples:\n{available_cities}"
        ) from e
    
    data = response.json()
    weather_data = [(pd.Timestamp(entry['dt'], unit='s', tz='UTC').tz_convert('Europe/Madrid'),
                     entry['main']['temp'] - 273.15) for entry in data['list']]
    
    df = pd.DataFrame(weather_data, columns=['Timestamp', 'Exterior Temperature'])
    return interpolate_hourly_data(df, 24)

def get_historical_weather_data(time_range: Optional[pd.DatetimeIndex] = None, city: str = "Barcelona, ES") -> pd.DataFrame:
    """Get historical weather data for a specified city. Examples: Barcelona, ES | Madrid, ES | Berlin, DE"""
    api_key = get_weather_api_key()
    
    # Use geocoding API to get the latitude and longitude of the city
    geocode_url = f"https://api.openweathermap.org/data/2.5/weather"
    geocode_response = requests.get(
        geocode_url, params={"q": city, "appid": api_key}
    )
    geocode_response.raise_for_status()
    geocode_data = geocode_response.json()
    
    lat = geocode_data['coord']['lat']
    lon = geocode_data['coord']['lon']
    
    # Define the time range
    if time_range is None:
        # Default to last 24 hours if no time range is provided
        end_time = datetime.utcnow()
        start_time = end_time - pd.Timedelta(hours=24)
    else:
        start_time, end_time = time_range[0], time_range[-1]
    
    start_timestamp = int(start_time.timestamp())
    end_timestamp = int(end_time.timestamp())
    
    # Call the historical weather data API
    historical_url = f"https://api.openweathermap.org/data/2.5/onecall/timemachine"
    weather_data = []
    
    for timestamp in range(start_timestamp, end_timestamp, 3600):  # Loop through hourly timestamps
        response = requests.get(
            historical_url,
            params={"lat": lat, "lon": lon, "dt": timestamp, "appid": api_key}
        )
        response.raise_for_status()
        data = response.json()
        weather_data.append((pd.Timestamp(data['current']['dt'], unit='s', tz='UTC').tz_convert('Europe/Madrid'),
                             data['current']['temp'] - 273.15))  # Convert from Kelvin to Celsius
    
    df = pd.DataFrame(weather_data, columns=['Timestamp', 'Exterior Temperature'])
    
    # If necessary, interpolate missing hourly data
    return df

def interpolate_hourly_data(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    df = df.set_index('Timestamp').resample('h').interpolate()
    
    start_time = pd.Timestamp.now(tz='Europe/Madrid').floor('h')
    time_range = pd.date_range(start=start_time + pd.Timedelta(hours=1),
                              end=start_time + pd.Timedelta(hours=hours),
                              freq='h', tz='Europe/Madrid')
    
    df = df.reindex(time_range).interpolate()
    return df.bfill() if df.isnull().values.any() else df

def get_spain_electricity_prices(time_range: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
    price_api_key = os.getenv('PRICE_API_KEY') or read_api_key('./config/price_api_key.txt')
    client = EntsoePandasClient(api_key=price_api_key)
    
    if time_range is None:
        start = pd.Timestamp.now(tz='Europe/Madrid')
        end = start + pd.Timedelta(hours=24)
        time_range = pd.date_range(start=start, end=end, freq='h', tz='Europe/Madrid')
    else:
        start = time_range[0]
        end = time_range[-1]
    
    prices = client.query_day_ahead_prices('ES', start=start, end=end)
    return (prices.to_frame(name='Price')
            .reindex(time_range, method='nearest')
            .assign(Price=lambda x: x['Price'] / 1000)) # convert price from EUR/MWh to EUR/kWh

def get_prices(time_range: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame: #TODO: expand to other countries
    return get_spain_electricity_prices(time_range = time_range)