# Import necessary modules
from strom.api_utils import get_price_series
from strom.data_utils import join_data

import pandas as pd

temp_df = pd.read_csv('data/Temp_Barcelona_Nov.csv')
temp_df.rename(columns={'temp': 'Exterior Temperature'}, inplace=True)
temp_df['Timestamp'] = pd.to_datetime(temp_df['datetimeEpoch'], unit='s').dt.tz_localize('Europe/Madrid')
temp_df.set_index('Timestamp', inplace=True)
temperature_series = temp_df['Exterior Temperature']
time_range = temperature_series.index
price_df = get_price_series(time_range=time_range)
price_now_df = get_price_series()
price_temp_df = join_data(temp_df, price_df)
price_temp_df.to_csv('data/Price_Temp_Barcelona_Nov.csv')
