import asyncio
from strom.cli import setup_env_config, main

if __name__ == "__main__":
    email, password, device_ip, house = setup_env_config()
    asyncio.run(main(email, password, device_ip, house))
