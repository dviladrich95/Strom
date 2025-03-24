from strom import optimization_utils

temp_price_df = optimization_utils.get_temp_price_df()
compare_df = optimization_utils.compare_decision_costs(temp_price_df)

optimization_utils.plot_costs_and_temps(compare_df)