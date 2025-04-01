from strom.optimization_utils import House, compare_output_costs
from strom.plot_utils import plot_combined_cases

import pandas as pd
import matplotlib.pyplot as plt

optimal_state_df = pd.read_csv('data/optimal_state.csv', index_col='Timestamp', parse_dates=['Timestamp'])

baseline_state_df = pd.read_csv('data/baseline_state.csv', index_col='Timestamp', parse_dates=['Timestamp'])

fig = plot_combined_cases(optimal_state_df, baseline_state_df)
# Save as png
fig.savefig("./plots/compare_costs_temps_Barcelona_Mar23_Mar25.png")

#show the plot
plt.show()