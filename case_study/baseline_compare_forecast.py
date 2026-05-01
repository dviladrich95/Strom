import json

from strom.data_utils import get_temp_price_df
from strom.optimization_utils import House, compare_output_costs
from strom.plot_utils import plot_combined_cases
import matplotlib.pyplot as plt

temp_price_df = get_temp_price_df()

with open("config/house_config.json") as f:
    house_cfg = json.load(f)
house = House(**house_cfg, freq="5min")

optimal_state_df, baseline_state_df, thermostat_state_df = compare_output_costs(temp_price_df, house)

fig = plot_combined_cases(optimal_state_df, baseline_state_df)
# Save as png
fig.savefig("./plots/compare_costs_temps_Barcelona.png")

#show the plot
plt.show()