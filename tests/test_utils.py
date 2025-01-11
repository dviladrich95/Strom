from src.strom import utils

def test_get_api_key():
    test_key_path = './tests/test_apiKey.txt'
    api_key = utils.get_api_key(test_key_path)
    assert api_key == 'test123'

def test_get_weather_data():
    df = utils.get_weather_data()
    assert df.shape[1] == 2
    assert df.shape[0] == 24

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
    assert df.shape[1] == 3
    assert 'Temperature (°C)' in df.columns
    assert 'Price' in df.columns
    assert 'Timestamp' in df.columns