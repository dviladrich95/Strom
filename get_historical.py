# Import necessary modules
from strom import optimization_utils
from strom.api_utils import read_api_key as get_api_key, get_weather_data, get_price_series
from strom.data_utils import get_temp_price_df, join_data

import pandas as pd

# Load the CSV file
temp_df = pd.read_csv('data/Temp_Barcelona_Nov.csv')
temp_df.rename(columns={'temp': 'Exterior Temperature'}, inplace=True)

# Convert 'datetimeEpoch' to datetime
temp_df['Timestamp'] = pd.to_datetime(temp_df['datetimeEpoch'], unit='s')

# Localize the 'Timestamp' to a specific timezone, e.g., Europe/Madrid
temp_df['Timestamp'] = temp_df['Timestamp'].dt.tz_localize('Europe/Madrid')

# Set 'Timestamp' as the index
temp_df.set_index('Timestamp', inplace=True)

# Convert to Series
temperature_series = temp_df['Exterior Temperature']
temperature_series.name = 'Exterior Temperature'

# Create the time range based on the index
time_range = temperature_series.index

# Get price data for the specified time range
price_df = get_price_series(time_range=time_range)

# Get current price data
price_now_df = get_price_series()

# Display the price data (this could be for debugging purposes)
# print(price_df)

# Join temperature and price data
price_temp_df = join_data(temp_df, price_df)

# Output the joined data (again, for debugging purposes)
print(price_temp_df)
