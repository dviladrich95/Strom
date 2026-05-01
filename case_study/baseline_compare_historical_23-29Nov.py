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

temp_price_df = pd.read_csv('data/Temp_Price_Barcelona_Nov.csv', index_col="Timestamp", parse_dates=["Timestamp"])
temp_price_df['ExteriorTemperature'] = remove_temperature_spikes(temp_price_df['ExteriorTemperature'])

# 7-day window: 23rd through 29th November (inclusive)
i_init = 22 * len(temp_price_df) // 30
i_end  = 29 * len(temp_price_df) // 30
temp_price_df = temp_price_df[i_init:i_end]

optimal_state_df, baseline_state_df, thermostat_state_df = compare_output_costs(temp_price_df, house)

fig = plot_combined_cases(optimal_state_df, baseline_state_df)
fig.savefig("./plots/compare_costs_temps_Barcelona_23-29_Nov.png")

compute_and_save_results(
    optimal_state_df,
    baseline_state_df,
    thermostat_state_df,
    "./results/results_Barcelona_23-29_Nov.txt",
    house,
    label="Barcelona — 23rd to 29th November 2024",
)

plt.close()
