from strom import utils

temp_price_df = utils.get_temp_price_df()
compare_df = utils.compare_decision_costs(temp_price_df)

utils.plot_costs_and_temps(compare_df)