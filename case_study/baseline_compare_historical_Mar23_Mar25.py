from strom.optimization_utils import House
from strom.plot_utils import plot_combined_cases_years
from case_study.results_utils import compute_and_save_results

import pandas as pd
import matplotlib.pyplot as plt

house = House(
    C_air=0.26,
    C_wall=19.1,
    R_interior=0.42,
    R_exterior=8.86,
    Q_heater=2.0,
    Q_cooling=2.0,
    T_min=18.0,
    T_max=24.0,
    T_interior_init=18.5,
    T_wall_init=18.5,
    P_base=0.01,
    freq="1h",
)

optimal_state_df = pd.read_csv('data/optimal_state2.csv', index_col='Timestamp', parse_dates=['Timestamp'])
baseline_state_df = pd.read_csv('data/baseline_state2.csv', index_col='Timestamp', parse_dates=['Timestamp'])

#df_half = 1- len(optimal_state_df) // 100
#optimal_state_df = optimal_state_df[df_half:]
#baseline_state_df = baseline_state_df[df_half:]

fig = plot_combined_cases_years(optimal_state_df, baseline_state_df)
# Save as png
fig.savefig("./plots/compare_costs_temps_Barcelona_Mar23_Mar25.png")

compute_and_save_results(
    optimal_state_df,
    baseline_state_df,
    "./results/results_Barcelona_Mar23_Mar25.txt",
    house,
    label="Barcelona — March 2023 to March 2025",
)

plt.close()