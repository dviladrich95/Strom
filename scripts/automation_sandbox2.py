import asyncio
from kasa import Discover
from datetime import datetime
from dotenv import load_dotenv
import os
#the lost sandbox 2 i never pushed
# Load .env file for credentials
load_dotenv(dotenv_path="/Users/huber/PycharmProjects/Strom/config/tapologin.env")
email = os.getenv("EMAIL")
password = os.getenv("PASSWORD")


device_ip = "192.168.1.16"

# The 24-hour schedule as a list of 1s and 0s
hourly_schedule = [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
#                  0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23

async def control_light():
    try:
        # Discover the device
        dev = await Discover.discover_single(device_ip, username=email, password=password)

        while True:
            # Get the current hour
            current_hour = datetime.now().hour
            # Determine action based on the schedule
            action = hourly_schedule[current_hour]

            if action == 1:
                await dev.turn_on()
                print(f"[{datetime.now()}] Hour {current_hour}: Device turned ON.")
            else:
                await dev.turn_off()
                print(f"[{datetime.now()}] Hour {current_hour}: Device turned OFF.")

            await dev.update()
            print(f"Device state: {'ON' if dev.is_on else 'OFF'}")

            # Sleep until the next hour
            await asyncio.sleep(500)  # 3600 seconds = 1 hour

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(control_light())
