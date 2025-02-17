# Strom Project

## Overview
The goal of this project is to analyze and interact with API data about energy and energy prices.
One of the possible interactions is to trigger a smart plug to turn off in periods with high energy demand/prices.

## Installation
1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Create a _config_ folder in the root project directory. This folder is your personal api keys will be saved
5. Place your electricity price and weather API keys in a "price_api_key.txt" "weather_api_Key.txt" file that you create in the _config_ folder.
6. Place your tapo account credentials in a "tapologin.env" file in the _config_ folder.

## Usage

[Technical documentation](documentation.md)

To run the main script manually:
```sh
python main.py  # python3 main.py for Mac users
```

Alternatively create [a cron job](https://www.freecodecamp.org/news/cron-jobs-in-linux/) or similar, that activates the main script hourly.


## Future Considerations
- Cron job installer
- Standalone executable