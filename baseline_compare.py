
from strom.data_utils import get_temp_price_df
from strom.optimization_utils import House, compare_decision_costs, plot_state, plot_factor_analysis
import numpy as np
import matplotlib.pyplot as plt

temp_price_df = get_temp_price_df()
house = House(freq='min')
optimal_state_df, baseline_state_df = compare_decision_costs(temp_price_df,house)

fig_base = plot_state(baseline_state_df, 'Baseline')
fig_opt = plot_state(optimal_state_df, 'Optimal')

# Initialize parameters
n = 5
min_temperature = 18.0
C_air=0.56
R_internal=1.0
C_walls_list = np.linspace(2.5, 4.0, n)
R_external_list = np.linspace(5.0, 10.0, n)
Q_heater_list = np.linspace(2.0, 10.0, n)

# Initialize the cost arrays
optimal_cost = np.zeros((n, n, n))
baseline_cost = np.zeros((n, n, n))

# Get temperature and price data once
temp_price_df = get_temp_price_df()

# Iterate over all possible parameter values
for i, C_walls in enumerate(C_walls_list):
    for j, Q_heater in enumerate(Q_heater_list):
        for k, R_external in enumerate(R_external_list):
            house = House(C_air=C_air, 
                                C_walls=C_walls, 
                                R_internal=R_internal, 
                                R_external=R_external, 
                                Q_heater=Q_heater, 
                                min_temperature=min_temperature)
            
            optimal_state_df, baseline_state_df = compare_decision_costs(temp_price_df, house) 
            # Update the cost arrays
            optimal_cost[i, j, k] = optimal_state_df['Cost'].sum(min_count=1)
            baseline_cost[i, j, k] = baseline_state_df['Cost'].sum(min_count=1)
            print(optimal_cost[i, j, k])

print("Arrays populated successfully")
fig = plot_factor_analysis(optimal_cost,baseline_cost,
                            C_walls_list, Q_heater_list, R_external_list,
                            'Relative')
# Save as HTML
fig.write_html("./plots/cost_savings_3d_rel.html")
fig = plot_factor_analysis(optimal_cost,baseline_cost,
                            C_walls_list, Q_heater_list, R_external_list,
                            'Absolute')
# Save as HTML
fig.write_html("./plots/cost_savings_3d_abs.html")


