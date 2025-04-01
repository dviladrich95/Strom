
from strom.api_utils import get_price_series
from strom.data_utils import join_data, regularize_df, get_temp_price_from_temp
from strom.optimization_utils import House, compare_output_costs
from strom.plot_utils import plot_combined_cases

import pandas as pd
import matplotlib.pyplot as plt

temp_df = pd.read_csv('data/Temp_Barcelona_Mar23_Mar25.csv')
temp_price_df = get_temp_price_from_temp(temp_df)

temp_price_df.to_csv('data/Temp_Price_Barcelona_Mar23_Mar25.csv')
temp_price_df_new = pd.read_csv('data/Temp_Price_Barcelona_Mar23_Mar25.csv', index_col='Timestamp', parse_dates=['Timestamp'])

print(temp_price_df_new.index.dtype)
