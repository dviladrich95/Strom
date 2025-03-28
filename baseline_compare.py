
from strom.data_utils import get_temp_price_df
from strom.optimization_utils import House, compare_decision_costs, plot_combined_cases, plot_factor_analysis
import numpy as np
import matplotlib.pyplot as plt

temp_price_df = get_temp_price_df()

house = House(
    C_air = 0.56,
    C_walls = 3.5,
    R_internal = 1.0,
    R_external = 6.06,
    Q_heater = 2.0,
    min_temperature = 18.0,
    max_temperature = 24.0,
    init_indoor_temp = 18.5,
    init_wall_temp = 18.5,
    freq = 'min')

optimal_state_df, baseline_state_df = compare_decision_costs(temp_price_df,house)

fig = plot_combined_cases(optimal_state_df, baseline_state_df)
# Save as png
fig.savefig("./plots/compare_costs_temps.png")

#show the plot
plt.show()