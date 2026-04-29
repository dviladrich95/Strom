
from strom.data_utils import get_temp_price_df
from strom.optimization_utils import House, compare_output_costs
from strom.plot_utils import plot_combined_cases
import matplotlib.pyplot as plt

temp_price_df = get_temp_price_df()

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
    freq = '15min'
    )

optimal_state_df, baseline_state_df = compare_output_costs(temp_price_df,house)

fig = plot_combined_cases(optimal_state_df, baseline_state_df)
# Save as png
fig.savefig("./plots/compare_costs_temps_Barcelona.png")

#show the plot
plt.show()