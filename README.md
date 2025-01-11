# Strom Project

## Overview
The goal of this project is to analyze and interact with API data about energy and energy prices.
One of the possible interactions is to trigger a smart plug to turn off in periods with high energy demand/prices.
Currently there is a bit of manual setup needed, detailed below.

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

4. Create a _config_ folder in the root project directory. This folder is where we save secret keys.
5. Place your API key in a "apiKey.txt" file that you create in the _config_ folder
6. Optionally: Place your tapo account credentials in a text file in the _config_ folder

## Usage
To run the main script manually:
```sh
python main.py #python3 main.py for Mac users
```
Alternatively create a cron job or similar, that activates the main script hourly.


## Future Considerations
- Cron job installer
