# Strom Project

[![Unit Tests](https://github.com/Bloodwing1/Strom/actions/workflows/strom-tests.yml/badge.svg)](https://github.com/Bloodwing1/Strom/actions/workflows/strom-tests.yml)

## Overview

Strom is a free, open-source script that brings smart heating to your home. It uses weather forecasts and electricity price data to fine-tune energy use, finding a cost-effective heating schedule through convex optimization. With a smart plug, Strom quietly takes care of the details, automatically adjusting your heating to save energy. It’s a simple, clever way to make your home more efficient and eco-friendly.

[Read the docs here](https://janbalanya.com/strom-docs/)

## Requirements

Requires **Python 3.12.8**

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/Bloodwing1/Strom.git
    cd Strom
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use ``
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

4. Create a _config_ folder in the root project directory. This folder is your personal api keys will be saved
5. Place your electricity price and weather API keys in a "price_api_key.txt" "weather_api_Key.txt" file that you create in the _config_ folder.
6. Place your tapo account credentials in a "tapologin.env" file in the _config_ folder. The content of this .env file should look like this:

    ```env
    EMAIL=myemail@hotmail.com
    PASSWORD=myPassword12
    ```

6. You can modify house heating parameters in the "house_config.json" file in the _config_ folder.

```json
{
    "C_air": 0.26,
    "C_wall": 19.1,
    "R_interior": 0.42,
    "R_exterior": 8.86,
    "Q_heater": 2.0,
    "Q_cooling": 0.0,
    "T_min": 18.0,
    "T_max": 24.0,
    "T_interior_init": 18.5,
    "T_wall_init": 18.5,
    "P_base": 0.01,
    "freq": "1h"
}
```

 Otherwise the main script will generate and empty json file that will fill the house with the default values we used.


## Usage

[Technical documentation](https://janbalanya.com/strom-docs/)

To run the main script manually:

```sh
python main.py  # python3 main.py for Mac users
```

Alternatively create [a cron job](https://www.freecodecamp.org/news/cron-jobs-in-linux/) or similar, that activates the main script hourly.

## Documentation

The project uses Sphinx for documentation. While the repository is private, you can build and view the documentation locally.

### Local Build Instructions

1.  **Install documentation dependencies**:
    ```sh
    pip install -r docs/requirements.txt
    ```

2.  **Build the documentation**:
    ```sh
    cd docs
    sphinx-build -b html . _build/html
    ```

3.  **View the documentation**:
    Open `docs/_build/html/index.html` in your web browser.

Once the repository is made public, it is pre-configured for automatic deployment on [Read the Docs](https://readthedocs.org).

## Future Considerations

- Cron job installer
- Standalone executable
