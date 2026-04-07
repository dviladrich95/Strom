# Case Study: Smart Heating Optimization (Skeleton)

## 1. The Business Problem
- **Stakeholder:** A friend heard about the dynamic electricity tariffs spreading across Europe: prices that swing hour by hour, sometimes dropping to almost nothing at midday. His temporary electric stove is the most expensive appliance he owns, and the idea of running it at the right hour and paying a fraction of the usual rate was genuinely exciting. The problem is that doing it manually means checking an app every hour. He wanted it automated.
- **Context:** The Spanish energy market is one of Europe's with the largest green electricity production, and thus is the first to see large intra-day price fluctuations (the "duck curve"). The spread between cheap and expensive hours is widening as solar penetration grows. Dynamic pricing plans and smart appliances are now exposing these fluctuations directly to consumers, making it possible to act on them.
- **Opportunity:** Load shifting means consuming energy at lower-cost times without changing the total amount consumed. Among flexible loads, heating and cooling are particularly attractive: they are large (often the dominant household load), and comfort tolerates a window of a few degrees and several hours, which is exactly the flexibility we need.
- **Objective:** Minimize the electricity bill while maintaining the indoor temperature within a defined comfort band. To do this systematically, we leverage a physical model of the building to formulate a linear optimization problem: one that can be solved in milliseconds, guarantees a global optimum, and whose solution is fully interpretable. Because every parameter in the model has a direct physical meaning (a heat capacity, a thermal resistance, a power rating), we can always trace back why the optimizer chose to heat at a given hour, which makes the system explainable by construction.
- **Value Proposition:** Cost savings for end-consumers and demand-side load shifting for the grid. Crucially, grounding the approach in physics also makes it auditable: a building manager or end-user can inspect the solution and follow the reasoning, unlike a black-box ML model where the decision is opaque.

## 2. Intuitive Explanation of the Dynamics
- **Why storage matters:** What good is even more perishable than fish? Electricity. The instant a light bulb switches on, the grid must inject exactly that much extra power to stay stable. If no one uses it, it is gone forever. This is why every corner of the European grid is being searched for storage: chemical batteries in EVs, hot water tanks, and yes, the thermal mass of a building that stays warm for hours after the heater shuts off. Storage is ultimately about keeping options open. The most valuable storage is sometimes just the freedom to run the laundry at a different time of day.
- **The first reframe:** When someone says "our customers pay too much for electricity," the instinct is to treat it as a pricing problem. The more useful frame is: *this is a thermal storage problem*. A house stores heat the way a battery stores charge. Once we see it that way, the path from the business problem to a formal model becomes much clearer.
- **Intuitive heat battery anatomy:** Picture two chambers connected by a narrow pipe. The indoor air chamber is thin and fills quickly: a small amount of heat raises its temperature fast. The wall chamber is wide and fills slowly: it absorbs a lot of energy before its temperature budges. A second, leaky pipe connects the wall to the outside, and how narrow that pipe is determines how well-insulated the house is. Rescaling each chamber's width by its thermal mass rather than its physical size gives the "temperature perspective": the thermal resistance is represented by the width of the pipe, and the thermal capacity by the volume of the chamber. These are exactly the constants we will use in the model.
- **Thermal Inertia:** The walls and air store heat, allowing us to pre-heat the house when energy is cheap and let it slowly cool down when energy is expensive.
- **Key Terms:**
    - *Heat Capacity:* How much energy is needed to raise the temperature of a component by 1°C.
    - *Thermal Resistance:* How fast heat leaks between two components (or to the outside).
    - *Comfort Band:* The acceptable temperature range (e.g., 18°C to 24°C), which is the constraint the optimizer must never violate.

## 3. The Physical Model
- **The second reframe:** From the "thermal battery" intuition, the components of a physical model follow directly. Thermal masses (air, wall) map to capacitors. Insulation between them maps to resistors. Heater power is the controllable input. The model parameters C_air, C_wall, R_interior, and R_exterior are exactly the chamber volumes and pipe widths from the intuitive picture above, now given precise units. This is a 2R2C (Two-Resistor, Two-Capacitor) equivalent circuit, a standard modeling pattern in building energy simulation.
- **Components:**
    - Two thermal masses (capacitors): Indoor Air and the Insulated Wall.
    - Two thermal resistances: Between air and wall, and between the wall and the exterior environment.
- **System State:** The temperatures of the air and the wall over time.
- **Inputs:** Exterior temperature (disturbance) and the power delivered by the heating/cooling unit (control variable).

## 4. Modeling Alternatives & Selection Justification
- **Alternative 1: 1st-Order Model (1R1C)**
    - *Pros:* Extremely simple.
    - *Cons:* Too simplistic to capture the delayed thermal response of massive walls; often over-optimistic about heat retention.
- **Alternative 2: White-box / Finite Element Models (e.g., EnergyPlus)**
    - *Pros:* Highly accurate and detailed spatial representation.
    - *Cons:* Computationally heavy, hard to parameterize for an average home, and not suited for fast online optimization.
- **Our Choice: 2-Component Lumped Model (2R2C)**
    - *Why:* The smallest model that captures the essential dynamic of slow wall inertia versus fast air response, while remaining linear. Linearity is what allows us to embed the model directly inside a linear optimization problem.

## 5. Optimization Method & Alternatives
- **The Challenge:** Finding the optimal sequence of heater/cooler commands over a 24-hour horizon, subject to linear system dynamics and temperature constraints.
- **Alternative 1: Rule-Based Control (Thermostat)**
    - *Pros:* Simple, hardware out-of-the-box.
    - *Cons:* Myopic. Cannot plan for future price drops or pre-heat the house. Has no concept of price.
- **Alternative 2: Reinforcement Learning (RL)**
    - *Pros:* Can handle complex, non-linear dynamics without explicit modeling.
    - *Cons:* Requires massive amounts of training data, lacks hard guarantees on constraint satisfaction (comfort), and is computationally expensive to train.
- **Our Choice: Linear Optimization (Model Predictive Control)**
    - *Why:* Our physical model is linear and the cost function is linear in heater power, so the full problem is linear. Linear solvers guarantee the global optimum. Temperature constraints are hard constraints, not soft penalties: they are satisfied by construction. The problem re-solves every hour with fresh forecasts (receding horizon). A linear formulation also gives us slack variables for free: they reveal exactly where the optimizer had headroom and where it hit a system limit, such as the heater running at full capacity or the indoor temperature pressing against the comfort boundary. This is valuable both for debugging the system and for explaining scheduling decisions to end-users.

## 6. Tools and Tech Stack
- **Python + CVXPY (short horizons):** Python is the natural home for this work, ubiquitous in data science and engineering, with CVXPY as the standard library for formulating convex problems. For daily runs and real-time execution, the ecosystem and iteration speed matter more than raw performance.
- **Julia + JuMP.jl (long horizons):** For multi-year historical sweeps, solver performance becomes the bottleneck. Julia is designed for high-performance numerical computing, and JuMP.jl is a powerful optimization modeling language that proved its worth when solving large-scale historical instances that would have been impractical in Python.
- **Data Handling:** `pandas` and `numpy` for time-series manipulation of prices and temperatures.
- **APIs:** ENTSO-E for day-ahead price forecasts, OpenWeatherMap for exterior temperature forecasts.
- **Execution:** `python-kasa` for async control of the TP-Link smart plug.
