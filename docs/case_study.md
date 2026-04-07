# Case Study: Smart Heating Optimization

## 1. The Business Problem

A friend heard about the dynamic electricity tariffs spreading across Europe: prices that swing hour by hour, sometimes dropping to almost nothing at midday. His temporary electric stove is the most expensive appliance he owns, and the idea of running it at the right hour and paying a fraction of the usual rate was genuinely exciting. The problem is that doing it manually means checking an app every hour. He wanted it automated.

![Energy Price Fluctuation](../plots/screenshot-energy-2025.png)
*Typical intra-day electricity price fluctuations in Spain.*

The Spanish energy market is one of Europe's with the largest green electricity production, and thus is the first to see these large intra-day price fluctuations, known as the "duck curve." The spread between cheap and expensive hours is widening as solar penetration grows, and dynamic pricing plans and smart appliances are now exposing these fluctuations directly to consumers, making it possible to act on them.

Load shifting means consuming energy at lower-cost times without changing the total amount consumed. Among flexible loads, heating and cooling are particularly attractive: they are large (often the dominant household load), and comfort tolerates a window of a few degrees and several hours, which is exactly the flexibility we need.

The objective is to minimize the electricity bill while maintaining the indoor temperature within a defined comfort band. To do this systematically, we leverage a physical model of the building to formulate a linear optimization problem: one that can be solved in milliseconds, guarantees a global optimum, and whose solution is fully interpretable. Because every parameter in the model has a direct physical meaning (a heat capacity, a thermal resistance, a power rating), we can always trace back why the optimizer chose to heat at a given hour, which makes the system explainable by construction. This also makes it auditable: a building manager or end-user can inspect the solution and follow the reasoning, unlike a black-box ML model where the decision is opaque.

## 2. Intuitive Explanation of the Dynamics

**Why storage matters.** What good is even more perishable than fish? Electricity. The instant a light bulb switches on, the grid must inject exactly that much extra power to stay stable. If no one uses it, it is gone forever. This is why every corner of the European grid is being searched for storage: chemical batteries in EVs, hot water tanks, and yes, the thermal mass of a building that stays warm for hours after the heater shuts off. Storage is ultimately about keeping options open. The most valuable storage is sometimes just the freedom to run the laundry at a different time of day.

**The first reframe.** When someone says "our customers pay too much for electricity," the instinct is to treat it as a pricing problem. The more useful frame is: *this is a thermal storage problem*. A house stores heat the way a battery stores charge. Once we see it that way, the path from the business problem to a formal model becomes much clearer.

**Intuitive heat battery anatomy.** Picture two chambers connected by a narrow pipe. The indoor air chamber is thin and fills quickly: a small amount of heat raises its temperature fast. The wall chamber is wide and fills slowly: it absorbs a lot of energy before its temperature budges. A second, leaky pipe connects the wall to the outside, and how narrow that pipe is determines how well-insulated the house is. Rescaling each chamber's width by its thermal mass rather than its physical size gives the "temperature perspective": the thermal resistance is represented by the width of the pipe, and the thermal capacity by the volume of the chamber. These are exactly the constants we will use in the model.

Three concepts govern this thermal battery:

- *Heat Capacity:* How much energy is needed to raise the temperature of a component by 1°C.
- *Thermal Resistance:* How fast heat leaks between two components (or to the outside).
- *Comfort Band:* The acceptable temperature range (e.g., 18°C to 24°C), which is the hard constraint the optimizer must never violate.

## 3. The Physical Model

**The second reframe.** From the "thermal battery" intuition, the components of a physical model follow directly. Thermal masses (air, wall) map to capacitors. Insulation between them maps to resistors. Heater power is the controllable input. The model parameters C_air, C_wall, R_interior, and R_exterior are exactly the chamber volumes and pipe widths from the intuitive picture above, now given precise units. This is a 2R2C (Two-Resistor, Two-Capacitor) equivalent circuit, a standard modeling pattern in building energy simulation.

The discrete-time dynamics are:

$$T_\text{air}[t+1] = T_\text{air}[t] + \frac{\Delta t}{C_\text{air}} \left( \frac{T_\text{wall}[t] - T_\text{air}[t]}{R_\text{interior}} + Q[t] \right)$$

$$T_\text{wall}[t+1] = T_\text{wall}[t] + \frac{\Delta t}{C_\text{wall}} \left( \frac{T_\text{air}[t] - T_\text{wall}[t]}{R_\text{interior}} + \frac{T_\text{ext}[t] - T_\text{wall}[t]}{R_\text{exterior}} \right)$$

The parameters used for this case study are:

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

## 4. Modeling Alternatives and Selection Justification

A spectrum of modeling approaches exists, each with different trade-offs between accuracy and tractability.

**1st-Order Model (1R1C).** A single thermal mass and a single resistance. Very easy to implement, but it treats the building as a uniform lump. It cannot capture the delayed thermal response of massive walls: the fact that a wall heated in the morning is still radiating warmth hours later. This leads to over-optimistic estimates of heat retention and poor schedule quality.

**White-box / Finite Element Models (e.g., EnergyPlus).** Detailed physics-based simulations that model every room, surface, and airflow path. Highly accurate, but computationally heavy, very hard to parameterize for an arbitrary home, and completely unsuitable for running as a real-time optimizer every hour.

**Our choice: 2-Component Lumped Model (2R2C).** The smallest model that captures the essential dynamic of slow wall inertia versus fast air response, while remaining linear. Linearity is what allows us to embed the model directly inside a linear optimization problem and solve it to global optimality in milliseconds.

## 5. Optimization Method and Alternatives

Given a model of how the house responds to heating, the optimizer finds the sequence of heater commands over a 24-hour horizon that minimizes total energy cost while keeping temperatures inside the comfort band.

**Rule-Based Control (Thermostat).** The standard approach: turn the heater on when temperature drops below a setpoint, off when it exceeds it. Simple and robust, but fundamentally myopic. It cannot look ahead to see that electricity will be half the price in two hours and pre-heat accordingly. It has no concept of price at all.

**Reinforcement Learning (RL).** An agent learns a control policy by interacting with the environment. Can in principle handle non-linear dynamics and complex reward structures, but requires large amounts of training data, offers no hard guarantees that temperature constraints will be satisfied, and is difficult to interpret or certify for deployment.

**Our choice: Linear Optimization (Model Predictive Control).** Because our physical model is linear and our cost function is linear in heater power, the entire problem is linear. Linear solvers guarantee the global optimum. Temperature comfort bounds are hard constraints, not soft penalties: they are always satisfied by construction. The problem is re-solved every hour with fresh price and weather forecasts, which is the "receding horizon" aspect of Model Predictive Control.

A linear formulation also gives us slack variables for free. They reveal exactly where the optimizer had headroom and where it hit a system limit, such as the heater running at full capacity or the indoor temperature pressing against the comfort boundary. This is valuable both for debugging the system and for explaining scheduling decisions to end-users.

## 6. Tools and Tech Stack

**Python + CVXPY (short horizons).** Python is the natural home for this work, ubiquitous in data science and engineering, with CVXPY as the standard library for formulating convex optimization problems. For daily runs and real-time execution, ecosystem and iteration speed matter more than raw performance.

**Julia + JuMP.jl (long horizons).** For multi-year historical sweeps, solver performance becomes the bottleneck. Julia is designed for high-performance numerical computing, and JuMP.jl is a powerful optimization modeling language that proved its worth when solving large-scale historical instances that would have been impractical in Python.

Additional components:

- `pandas` and `numpy` for time-series handling of price and weather data
- ENTSO-E API for day-ahead electricity price forecasts
- OpenWeatherMap API for exterior temperature forecasts
- `python-kasa` for async control of the TP-Link smart plug

## 7. Proof of Concept

Over a weekend in January 2025, we built the first working version of Strom: API fetching, the optimizer, and smart plug control all wired together. Despite significant room for improvement, it was a meaningful milestone: a system that autonomously decides when to heat a home based on forecast prices.

## 8. Historical Analysis

To evaluate the approach, we ran the optimizer retrospectively over historical price and weather data and compared it against a baseline constant-temperature policy: a thermostat that simply maintains 21°C at all times.

### 25th of November 2024

![Historical Comparison - Nov 25th 2024](../plots/compare_costs_temps_Barcelona_25th_Nov.png)
*Comparison between the optimal cost policy and the constant thermostat policy on the 25th of November 2024.*

The cost-aware strategy concentrates heating during the central portion of the day and at night, exploiting the daily duck-curve oversupply. This yields cost savings of at least 10% on this day alone.

### November 2024

![Monthly Comparison - Nov 2024](../plots/compare_costs_temps_Barcelona_Nov.png)
*Comparison over the second half of November 2024.*

The pattern holds across the month. The largest temperature spike occurred on the 25th: a day of exceptionally low prices for an extended period, which the optimizer correctly identified and exploited.

### Two-Year Analysis (March 2023 to March 2025)

For this analysis we extended the model to include a cooling option and ran it across two full years of data.

![Long-term Comparison - 2023-2025](../plots/compare_costs_temps_Barcelona_Mar23_Mar25.png)
*Comparison from March 2023 to March 2025.*

The cumulative saving over two years was **66 €**, a **17% reduction** relative to the total cost of the baseline policy.

## 9. Upcoming Improvements

**Modeling.** The current model omits efficiency parameters (COP for heat pumps, for example). A factor analysis mapping potential savings as a function of location, insulation quality, and device efficiency would help understand where this approach is most valuable.

**Dedicated Hardware.** To run reliably 24/7, we plan to containerize Strom for deployment on low-end home-assistant hardware (e.g., a Raspberry Pi), removing the dependency on a personal laptop.
