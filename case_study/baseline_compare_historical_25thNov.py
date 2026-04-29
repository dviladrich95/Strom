from strom.optimization_utils import House, compare_output_costs
from strom.plot_utils import plot_combined_cases
from strom.data_utils import remove_temperature_spikes
from case_study.results_utils import compute_and_save_results

import pandas as pd
import matplotlib.pyplot as plt

house = House(
    C_air = 0.26,
    C_wall = 19.1,
    R_interior = 0.42,
    R_exterior = 8.86,
    Q_heater = 2.0,
    Q_cooling = 2.0,
    T_min = 18.0,
    T_max = 24.0,
    T_interior_init = 18.5,
    T_wall_init = 18.5,
    P_base = 0.01,
    freq = '5min'
    )

temp_price_df = pd.read_csv('data/Temp_Price_Barcelona_Nov.csv', index_col="Timestamp", parse_dates=["Timestamp"])
temp_price_df['ExteriorTemperature'] = remove_temperature_spikes(temp_price_df['ExteriorTemperature'])

# remove the first half of the data
i_init = 23*len(temp_price_df) //30
i_end = 25*len(temp_price_df) //30
temp_price_df = temp_price_df[i_init:i_end]

optimal_state_df, baseline_state_df = compare_output_costs(temp_price_df,house)

fig = plot_combined_cases(optimal_state_df, baseline_state_df)
# Save as png
fig.savefig("./plots/compare_costs_temps_Barcelona_25th_Nov.png")

compute_and_save_results(
    optimal_state_df,
    baseline_state_df,
    "./results/results_Barcelona_25th_Nov.txt",
    house,
    label="Barcelona — 25th November 2024",
)

plt.close()