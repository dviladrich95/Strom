from strom import optimization_utils
from strom.api_utils import read_api_key as get_api_key, get_weather_data, get_prices
from strom.data_utils import get_temp_price_df, join_data


def test_get_api_key():
    test_key_path = './tests/test_price_api_key.txt'
    api_key = get_api_key(test_key_path)
    assert api_key == 'test123'

def test_get_weather_data():
    df = get_weather_data(city="Oslo")
    assert df.shape[1] == 1
    assert df.shape[0] == 24
    #check that all values are non nan
    assert df['Temperature (°C)'].isnull().sum() == 0

def test_get_weather_data_different_cities():
    oslo_df = get_weather_data(city="Oslo")
    bergen_df = get_weather_data(city="Bergen")
    
    assert oslo_df.shape == bergen_df.shape
    assert not oslo_df['Temperature (°C)'].equals(bergen_df['Temperature (°C)'])
    assert oslo_df.shape[0] == 24
    assert oslo_df.shape[1] == 1

def test_get_prices():
    prices_df = get_prices()
    assert prices_df.shape[0] == 24

def test_join_data():
    weather_df = get_weather_data(city="Oslo")
    prices_df = get_prices()

    assert weather_df.shape[0] == 24
    assert prices_df.shape[0] == 24

    df = join_data(weather_df, prices_df)
    assert df.shape[0] == 24
    assert df.shape[1] == 2
    assert 'Temperature (°C)' in df.columns
    assert 'Price' in df.columns

def test_get_temp_price_df():
    temp_price_df = get_temp_price_df()
    assert temp_price_df.shape[0] == 24
    assert temp_price_df.shape[1] == 2
    assert 'Temperature (°C)' in temp_price_df.columns
    assert 'Price' in temp_price_df.columns

def test_compare_decision_costs():
    temp_price_df = get_temp_price_df()
    optimization_utils.compare_decision_costs(temp_price_df)
