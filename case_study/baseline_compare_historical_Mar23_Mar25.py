from strom.plot_utils import plot_combined_cases_years

import pandas as pd
import matplotlib.pyplot as plt

optimal_state_df = pd.read_csv('data/optimal_state2.csv', index_col='Timestamp', parse_dates=['Timestamp'])
baseline_state_df = pd.read_csv('data/baseline_state2.csv', index_col='Timestamp', parse_dates=['Timestamp'])

#df_half = 1- len(optimal_state_df) // 100
#optimal_state_df = optimal_state_df[df_half:]
#baseline_state_df = baseline_state_df[df_half:]

fig = plot_combined_cases_years(optimal_state_df, baseline_state_df)
# Save as png
fig.savefig("./plots/compare_costs_temps_Barcelona_Mar23_Mar25.png")

#show the plot
plt.show()