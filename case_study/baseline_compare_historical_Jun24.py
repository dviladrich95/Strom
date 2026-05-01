import json

from strom.optimization_utils import House, compare_output_costs
from strom.plot_utils import plot_combined_cases
from strom.data_utils import remove_temperature_spikes
from case_study.results_utils import compute_and_save_results

import pandas as pd
import matplotlib.pyplot as plt

with open("config/house_config.json") as f:
    house_cfg = json.load(f)
house = House(**house_cfg, freq="5min")

temp_price_df = pd.read_csv(
    "data/Temp_Price_Barcelona_Mar23_Mar25.csv",
    index_col="Timestamp",
    parse_dates=["Timestamp"],
)
assert isinstance(temp_price_df.index, pd.DatetimeIndex)

temp_price_df["ExteriorTemperature"] = remove_temperature_spikes(
    temp_price_df["ExteriorTemperature"]
)
temp_price_df = temp_price_df[
    (temp_price_df.index.year == 2024) & (temp_price_df.index.month == 6)
]

optimal_state_df, baseline_state_df, thermostat_state_df = compare_output_costs(temp_price_df, house)

fig = plot_combined_cases(optimal_state_df, baseline_state_df)
fig.savefig("./plots/compare_costs_temps_Barcelona_Jun24.png")

compute_and_save_results(
    optimal_state_df,
    baseline_state_df,
    thermostat_state_df,
    "./results/results_Barcelona_Jun24.txt",
    house,
    label="Barcelona — June 2024",
)

plt.close()
