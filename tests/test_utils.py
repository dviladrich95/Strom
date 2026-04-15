import pytest
import numpy as np
import pandas as pd

from strom import optimization_utils
from strom.optimization_utils import House, smooth_temperature, calculate_baseline_target, find_heating_output
from strom.api_utils import read_api_key as get_api_key, get_weather_data, get_price_series
from strom.data_utils import get_temp_price_df, join_data, regularize_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_temp_price_df(n_hours=24, base_temp=10.0, base_price=0.05):
    """Synthetic hourly temp+price DataFrame for offline tests."""
    index = pd.date_range(
        start=pd.Timestamp.now(tz='Europe/Madrid').floor('h'),
        periods=n_hours,
        freq='h',
    )
    temp = pd.Series(base_temp + np.sin(np.linspace(0, 2 * np.pi, n_hours)), index=index, name='ExteriorTemperature')
    price = pd.Series(base_price + 0.01 * np.cos(np.linspace(0, 2 * np.pi, n_hours)), index=index, name='Price')
    return pd.concat([temp, price], axis=1)


# ---------------------------------------------------------------------------
# api_utils — offline
# ---------------------------------------------------------------------------

def test_get_api_key():
    test_key_path = './tests/test_price_api_key.txt'
    api_key = get_api_key(test_key_path)
    assert api_key == 'test123'


# ---------------------------------------------------------------------------
# api_utils — live (integration)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_get_weather_data():
    temp_series = get_weather_data(city="Oslo")
    assert len(temp_series) > 0
    assert not temp_series.isnull().any()
    assert temp_series.name == 'ExteriorTemperature'
    assert temp_series.index.tz is not None


@pytest.mark.integration
def test_get_weather_data_different_cities():
    oslo_series = get_weather_data(city="Oslo")
    bergen_series = get_weather_data(city="Bergen")

    assert len(oslo_series) == len(bergen_series)
    assert not oslo_series.equals(bergen_series)


@pytest.mark.integration
def test_get_price_data():
    price_series = get_price_series()
    assert len(price_series) > 0
    assert not price_series.isnull().any()
    assert price_series.name == 'Price'
    assert (price_series >= -1.0).all()  # day-ahead prices can be negative (e.g. solar surplus)
    assert price_series.index.tz is not None


@pytest.mark.integration
def test_join_data():
    temp_series = get_weather_data(city="Oslo")
    price_series = get_price_series()

    df = join_data(temp_series, price_series)
    assert df.shape[1] == 2
    assert 'ExteriorTemperature' in df.columns
    assert 'Price' in df.columns
    assert df.isnull().values.any() == False


@pytest.mark.integration
def test_get_temp_price_df():
    temp_price_df = get_temp_price_df()
    assert temp_price_df.shape[1] == 2
    assert 'ExteriorTemperature' in temp_price_df.columns
    assert 'Price' in temp_price_df.columns
    assert temp_price_df.isnull().values.any() == False
    assert temp_price_df.index.to_series().diff().dropna().eq(pd.Timedelta(hours=1)).all()


# ---------------------------------------------------------------------------
# data_utils — offline
# ---------------------------------------------------------------------------

def test_regularize_df():
    index = pd.date_range(start='2024-01-01', periods=10, freq='3h', tz='Europe/Madrid')
    df = pd.DataFrame({'ExteriorTemperature': np.linspace(5, 15, 10), 'Price': np.linspace(0.05, 0.15, 10)}, index=index)
    result = regularize_df(df, freq='1h')
    assert result.index.freq == pd.tseries.frequencies.to_offset('1h')
    assert not result.isnull().values.any()


# ---------------------------------------------------------------------------
# optimization_utils — House
# ---------------------------------------------------------------------------

def test_house_defaults():
    house = House()
    assert house.C_air == 0.56
    assert house.C_wall == 3.5
    assert house.R_interior == 1.0
    assert house.R_exterior == 6.06
    assert house.Q_heater == 2.0
    assert house.Q_cooling == 0.0
    assert house.T_min == 18.0
    assert house.T_max == 24.0
    assert house.freq == '1h'


# ---------------------------------------------------------------------------
# optimization_utils — pure functions
# ---------------------------------------------------------------------------

def test_smooth_temperature_shape():
    data = pd.Series(np.random.randn(48) + 10)
    result = smooth_temperature(data, window_hours=6, dt=1.0)
    assert result.shape == data.shape


def test_smooth_temperature_bounded():
    data = pd.Series(np.linspace(0, 20, 48))
    result = smooth_temperature(data, window_hours=4, dt=1.0)
    assert result.min() >= data.min() - 1e-9
    assert result.max() <= data.max() + 1e-9


def test_calculate_baseline_target_clipped():
    ext_temp = pd.Series(np.linspace(-5, 30, 48))
    T_min, T_max = 18.0, 24.0
    result = calculate_baseline_target(ext_temp, T_min, T_max, resolution_hours=1.0)
    assert result.shape == ext_temp.shape
    assert result.min() >= T_min - 1e-9
    assert result.max() <= T_max + 1e-9


# ---------------------------------------------------------------------------
# optimization_utils — find_heating_output
# ---------------------------------------------------------------------------

def test_find_heating_output_optimal():
    df = make_temp_price_df(n_hours=24)
    house = House()
    result = find_heating_output(df, house, 'optimal')
    assert 'HeaterOutput' in result.columns
    assert 'CoolingOutput' in result.columns
    assert 'InteriorTemperature' in result.columns
    assert not result['HeaterOutput'].isnull().any()
    assert (result['HeaterOutput'] >= 0).all()
    assert (result['HeaterOutput'] <= 1).all()
    assert (result['InteriorTemperature'] >= house.T_min - 1e-6).all()
    assert (result['InteriorTemperature'] <= house.T_max + 1e-6).all()


def test_find_heating_output_baseline():
    df = make_temp_price_df(n_hours=24)
    house = House()
    result = find_heating_output(df, house, 'baseline')
    assert 'HeaterOutput' in result.columns
    assert not result['HeaterOutput'].isnull().any()


def test_find_heating_output_invalid_mode():
    df = make_temp_price_df(n_hours=24)
    house = House()
    with pytest.raises(ValueError):
        find_heating_output(df, house, 'invalid_mode')


# ---------------------------------------------------------------------------
# optimization_utils — compare_output_costs (integration via live API)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_compare_output_costs():
    temp_price_df = get_temp_price_df()
    house = optimization_utils.House(P_base=0.0, Q_cooling=2.0)
    optimal_state_df, baseline_state_df = optimization_utils.compare_output_costs(temp_price_df, house)
    assert baseline_state_df.isnull().values.any() == False
    assert optimal_state_df.isnull().values.any() == False
    assert optimal_state_df['Cost'].sum() <= baseline_state_df['Cost'].sum()
