# Strom Project

## Overview
Strom is a free, open-source script that brings smart heating to your home. It uses weather forecasts and electricity price data to fine-tune energy use, finding a cost-effective heating schedule through convex optimization. With a smart plug, Strom quietly takes care of the details, automatically adjusting your heating to save energy. It’s a simple, clever way to make your home more efficient and eco-friendly.

[Read the docs here](https://janbalanya.com/strom-docs/)

## Requirements
Requires Python version 3.12.8

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

[Technical documentation](https://janbalanya.com/strom-docs/)

To run the main script manually:
```sh
python main.py  # python3 main.py for Mac users
```

Alternatively create [a cron job](https://www.freecodecamp.org/news/cron-jobs-in-linux/) or similar, that activates the main script hourly.


## Future Considerations
- Cron job installer
- Standalone executable