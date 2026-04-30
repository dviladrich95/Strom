import pandas as pd
from .api_utils import get_weather_data, get_price_series


def remove_temperature_spikes(
    series: pd.Series, threshold: float = 3.0
) -> pd.Series:
    """Replace isolated salt-and-pepper spikes with the mean of their neighbours.

    A spike is a single sample where the jump from the previous value AND the
    jump to the next value both exceed ``threshold`` in magnitude AND point in
    opposite directions (a +X then -X pattern, or vice versa). This is more
    conservative than a deviation-from-mean test — fast but continuous
    temperature changes (a cold front passing) are not flagged. Endpoints are
    left unchanged.
    """
    delta_prev = series - series.shift(1)
    delta_next = series.shift(-1) - series
    is_spike = (
        (delta_prev.abs() > threshold)
        & (delta_next.abs() > threshold)
        & (delta_prev * delta_next < 0)
    )
    neighbor_mean = (series.shift(1) + series.shift(-1)) / 2
    return series.where(~is_spike, other=neighbor_mean)

def join_data(temp_series, price_series):
    """
    Merge temperature and price dataframes on the 'Timestamp' column and extract temperature and prices as numpy arrays.
    Parameters:
    temp_df (pd.DataFrame): DataFrame containing temperature data with a 'Timestamp' column.
    prices_df (pd.DataFrame): DataFrame containing price data with a 'Timestamp' column.
    Returns:
    pd.DataFrame: Merged DataFrame containing both temperature and price data.
    """
    temp_price_df = pd.concat([temp_series, price_series], axis=1)
    temp_price_df.sort_index(inplace=True)
    temp_price_df = temp_price_df.interpolate(method='cubic').bfill().ffill()
    return temp_price_df

def regularize_df(df, freq = '1h'):
    df_resamp = df.resample(freq).asfreq()
    merged_df = df_resamp.interpolate(method='cubic', limit_direction='both').bfill().ffill()
    return merged_df

def get_temp_price_df():
    temp_series = get_weather_data()
    prices_series = get_price_series()
    temp_price_df = join_data(temp_series, prices_series)
    temp_price_df = regularize_df(temp_price_df)
    return temp_price_df

def get_temp_price_from_temp(temp_df):
    temp_df.rename(columns={'temp': 'ExteriorTemperature'}, inplace=True)
    temp_df['Timestamp'] = pd.to_datetime(temp_df['datetimeEpoch'], unit='s').dt.tz_localize('Europe/Madrid', ambiguous='NaT', nonexistent='shift_forward')
    temp_df['Timestamp'] = temp_df['Timestamp'].dt.tz_convert('UTC') # UTC needed to avoid mistakes when loading from csv
    temp_df.set_index('Timestamp', inplace=True)
    temp_df = temp_df.groupby(temp_df.index).mean().resample('h').interpolate('time')
    temp_series = temp_df['ExteriorTemperature']
    price_series = get_price_series()
    temp_price_df = join_data(temp_series, price_series)
    return temp_price_df