# Case Study: Smart Heating Optimization

The idea for the Strom project came from realizing the huge energy price fluctuations in Spain. Since heating has a bit of flexibility in when it's turned on, there is an opportunity to pick the optimal hours and lower the energy bill.

![Energy Price Fluctuation](../plots/screenshot-energy-2025.png)
*Typical energy price fluctuations throughout a day in Spain.*

## Proof of Concept Hackathon
Over a weekend in January 2025, we worked on the main code for Strom. We got the code to work and interact correctly with the smart plug. Despite there being a lot of room for improvement, we celebrated this first "minimum viable setup" achievement.

## Case Study Parameters

We modeled a house as a coupled, two-component system: the indoor air inside the house and its insulated wall. The electricity price typically has a minimum cost (`P_base`) that does not reach zero, even if the overall day-ahead price does.

| Parameter              | Value | Units  | Description                                     |
|------------------------|-------|--------|-------------------------------------------------|
| `C_air`                | 0.56  | kWh/°C | Heat capacity of indoor air                    |
| `C_wall`               | 3.5   | kWh/°C | Heat capacity of the insulated wall            |
| `R_interior`           | 1.0   | °C/kW  | Thermal resistance between air and wall        |
| `R_exterior`           | 6.06  | °C/kW  | Thermal resistance between wall and outside    |
| `T_min`                | 18.0  | °C     | Minimum allowed indoor temperature             |
| `T_max`                | 24.0  | °C     | Maximum allowed indoor temperature             |
| `T_interior_init`      | 18.5  | °C     | Initial indoor temperature                     |
| `T_wall_init`          | 20.0  | °C     | Initial wall temperature                       |
| `Q_heater`             | 2.0   | kW     | Power of the heating unit                      |
| `P_base`               | 0.01  | €/kWh  | Estimated base price from the provider         |

## Historical Analysis

### 25th of November

![Historical Comparison - Nov 25th 2024](../plots/compare_costs_temps_Barcelona_25th_Nov.png)
*Historical Comparison between our forecast-aware optimal cost policy and the constant thermostat temperature policy on the 25th of November 2024.*

The cost-aware strategy heats the interior during the central portion of the day and at night, taking advantage of the daily "duck curve" energy oversupply. This approach yields significant cost savings of at least 10%.

### November 2024

During November 2024, we observed spikes in indoor temperature. The largest spike occurred on the 25th of November, a day of exceedingly low energy prices for a prolonged period of time.

![Monthly Comparison - Nov 2024](../plots/compare_costs_temps_Barcelona_Nov.png)
*Comparison between our forecast-aware optimal cost policy and the constant thermostat temperature policy during the second half of November 2024.*

### Two-Year Analysis (March 2023 to March 2025)

For this two-year period, we added a cooling option. The cumulative difference over the two-year period was 66€, a **17% reduction** relative to the total cost of the base policy.

![Long-term Comparison - 2023-2025](../plots/compare_costs_temps_Barcelona_Mar23_Mar25.png)
*Comparison between our forecast-aware optimal cost policy and the constant thermostat temperature policy from March 2023 to March 2025.*

## Upcoming Improvements

### Modeling
The model still lacks some additions to make it more realistic. We did not incorporate efficiency parameters. A factor analysis would be interesting to map out the potential savings from this method depending on location and thermal parameters.

### Dedicated Hardware
To reliably send updated instructions to the smart plug, we plan to containerize our implementation to make it executable on low-end hardware for home assistants.
