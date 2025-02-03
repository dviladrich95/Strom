from strom import utils

def test_get_api_key():
    test_key_path = './tests/test_price_api_key.txt'
    api_key = utils.get_api_key(test_key_path)
    assert api_key == 'test123'

def test_get_weather_data():
    df = utils.get_weather_data()
    assert df.shape[1] == 2
    assert df.shape[0] == 24
    #check that all values are non nan
    assert df['Temperature (°C)'].isnull().sum() == 0

def test_get_prices():
    prices_df = utils.get_prices()
    assert prices_df.shape[0] == 24

def test_join_data():
    weather_df = utils.get_weather_data()
    prices_df = utils.get_prices()

    assert weather_df.shape[0] == 24
    assert prices_df.shape[0] == 24

    df = utils.join_data(weather_df, prices_df)
    assert df.shape[0] == 24
    assert df.shape[1] == 2
    assert 'Temperature (°C)' in df.columns
    assert 'Price' in df.columns

def test_get_temp_price_df():
    temp_price_df = utils.get_temp_price_df()
    assert temp_price_df.shape[0] == 24
    assert temp_price_df.shape[1] == 2
    assert 'Temperature (°C)' in temp_price_df.columns
    assert 'Price' in temp_price_df.columns

def test_compare_decision_costs():
    temp_price_df = utils.get_temp_price_df()
    utils.compare_decision_costs(temp_price_df)
