import asyncio
from kasa import Discover
from dotenv import load_dotenv
import os

# Load the environment variables from the .env file
load_dotenv(dotenv_path="/Users/huber/PycharmProjects/Strom/src/strom/tapologin.env")

device_ip = "192.168.1.16"
email = os.getenv("EMAIL")  # Get email from the environment variable
password = os.getenv("PASSWORD")  # Get password from the environment variable


async def main():
    try:
        # Discover the device
        dev = await Discover.discover_single(device_ip, username=email, password=password)

        # Prompt the user for input (0 for off, 1 for on)
        user_input = input("Enter 1 to turn the light on, 0 to turn it off: ")

        # Check user input and turn the light on or off accordingly
        if user_input == "1":
            await dev.turn_on()
            print("Device turned on.")
        elif user_input == "0":
            await dev.turn_off()
            print("Device turned off.")
        else:
            print("Invalid input. Please enter 0 or 1.")

        # Update the device state after action
        await dev.update()
        print(f"Device state: {'ON' if dev.is_on else 'OFF'}")

        # Close the device connection manually
        await dev.close()
        print("Device connection closed.")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
