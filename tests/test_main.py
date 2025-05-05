# test_smart_plug_controller.py
import pytest
import asyncio
from unittest.mock import patch
import pandas as pd

# Import your main module (assuming it's named smart_plug_controller.py)
from strom.cli import main
from strom.cli import setup_env_config

# Create your mock device for testing
class MockSmartPlug:
    def __init__(self):
        self.is_on = None
        
    async def update(self):
        return True
        
    async def turn_on(self):
        self.is_on = True
        return True
        
    async def turn_off(self):
        self.is_on = False
        return True
        
    async def async_close(self):
        return True

# Test that heating turns on when needed
@pytest.mark.asyncio




async def test_heating_on():

    email, password, device_ip, house = setup_env_config()
    # Create a mock device
    mock_device = MockSmartPlug()
    
    # Create mock heating result (ON)
    mock_result = pd.DataFrame({'HeaterOutput': [1]})
    
    with patch('kasa.Discover.discover_single', return_value=mock_device), \
         patch('strom.cli.find_heating_output', return_value=mock_result):
        
        await main(email, password, device_ip, house)
        print("Device state after main:", mock_device.is_on)
        assert mock_device.is_on == True

# Test that heating turns off when not needed
@pytest.mark.asyncio
async def test_heating_off():

    email, password, device_ip, house = setup_env_config()
    # Create a mock device
    mock_device = MockSmartPlug()
    
    # Create mock heating result (OFF)
    mock_result = pd.DataFrame({'HeaterOutput': [0]})

    
    with patch('kasa.Discover.discover_single', return_value=mock_device), \
         patch('strom.cli.find_heating_output', return_value=mock_result):
        
        await main(email, password, device_ip, house)
        assert mock_device.is_on == False