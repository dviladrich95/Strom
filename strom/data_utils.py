import pandas as pd
from .api_utils import get_weather_data, get_prices

def join_data(temp_df, prices_df):
    """
    Merge temperature and price dataframes on the 'Timestamp' column and extract temperature and prices as numpy arrays.
    Parameters:
    temp_df (pd.DataFrame): DataFrame containing temperature data with a 'Timestamp' column.
    prices_df (pd.DataFrame): DataFrame containing price data with a 'Timestamp' column.
    Returns:
    pd.DataFrame: Merged DataFrame containing both temperature and price data.
    """
    
    temp_df_reindexed = temp_df.reindex(prices_df.index.union(temp_df.index)).interpolate(method='time')
    temp_df_reindexed = temp_df_reindexed.bfill().ffill()
    temp_df_reindexed = temp_df_reindexed.reindex(prices_df.index)
    temp_price_df = pd.merge(temp_df_reindexed, prices_df, left_index=True, right_index=True, how='inner')
    return temp_price_df

def get_temp_price_df():
    temp_df = get_weather_data()
    prices_df = get_prices()
    temp_price_df = join_data(temp_df, prices_df)
    return temp_price_df
