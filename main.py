import asyncio
from kasa import Discover
from dotenv import load_dotenv
import os
import json
from strom.data_utils import get_temp_price_df
from strom.api_utils import find_root_dir
from strom.optimization_utils import find_heating_output, House

# Load the environment variables from the .env file
load_dotenv(dotenv_path="../../config/tapologin.env")

email = os.getenv("EMAIL")  # Get email from the environment variable
password = os.getenv("PASSWORD")  # Get password from the environment variable
device_ip = os.getenv("DEVICEIP")

# Load house config parameters
try:
    with open('./config/house_config.json', 'r') as f:
        house_params = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    # Raise an error if the file is not found or is not valid JSON
    raise ValueError("House config file not found or invalid JSON.")

house = House(**house_params)

if not device_ip:
    os.chdir(find_root_dir())
    with open('./config/device_IP.txt') as f:
        device_ip = f.read().strip()

async def main():
    try:
        # Discover the devices
        if not device_ip:
            raise ValueError("DEVICEIP environment variable is not set or is invalid.")
        dev = await Discover.discover_single(device_ip, username=email, password=password)
        temp_price_df = get_temp_price_df()
        # Prompt the user for input (0 for off, 1 for on)
        user_input = bool(find_heating_output(temp_price_df, house, 'optimal')['HeaterOutput'][0])
        # Check if the device was discovered successfully
        if dev is None:
            raise ValueError("Device could not be discovered. Please check the DEVICEIP, email, and password.")
        # Check user input and turn the switch on or off accordingly
        print(user_input)
        if user_input:
            await dev.turn_on()
            print("Device turned on.")
        else:
            await dev.turn_off()
            print("Device turned off.")
       # else:
        #    print("Invalid input. Please enter 0 or 1.")

        # Update the device state after action
        await dev.update()
        print(f"Device state: {'ON' if dev.is_on else 'OFF'}")

        # Close the device connection manually
        await dev.async_close()
        print("Device connection closed.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
