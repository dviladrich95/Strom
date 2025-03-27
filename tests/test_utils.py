from strom import utils
import numpy as np

def test_get_api_key():
    test_key_path = './tests/test_price_api_key.txt'
    api_key = utils.get_api_key(test_key_path)
    assert api_key == 'test123'

def test_get_weather_data():
    temp_series = utils.get_temp_series()
    # no nan values
    assert not temp_series.isnull().values.any()
    #check that all values are non nan
    assert not temp_series.isnull().values.any()
    #check that the name of the series is 'Temperature'
    assert temp_series.name == 'Temperature'

def test_get_prices():
    price_series = utils.get_price_series()
    # no nan values
    assert not price_series.isnull().values.any()
    #check that the name of the series is 'Price'
    assert price_series.name == 'Price'



def test_get_temp_price_df():
    temp_price_df = utils.get_temp_price_df()
    assert temp_price_df.shape[0] <= 24
    assert temp_price_df.shape[1] == 2
    assert 'Temperature' in temp_price_df.columns
    assert 'Price' in temp_price_df.columns
    #check that there are no nan values
    assert temp_price_df.isnull().values.any() == False
    #check that the period is 1 hour for each row
    assert (temp_price_df.index[1] - temp_price_df.index[0]).seconds == 3600

def test_get_state_df():
    temp_price_df = utils.get_temp_price_df()
    time_steps = len(temp_price_df)
    # make a decision function that is a sine wave array
    decision = np.sin(np.linspace(0, 5*2*np.pi, time_steps))
    #make the sine wave into a square wave between 0 and 1
    decision = np.sign(decision)
    decision = (decision + 1) / 2


    state = utils.get_state_df(temp_price_df, decision)
    assert state.shape[0] == time_steps

def test_compare_decision_costs():
    temp_price_df = utils.get_temp_price_df()
    utils.compare_decision_costs(temp_price_df)

