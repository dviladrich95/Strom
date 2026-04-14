## Sketch

# Case Study: Smart Heating Optimization — Sketch

**Story:** friend, electric stove, dynamic tariffs, wants automation

---

## 1. Business Problem
- **Stakeholder:** individual homeowner on variable tariff
- **USP:** comfort guarantees (T_min) + low bill
- **Context:** Spain → duck curve → widening price spread → dynamic plans expose fluctuations to consumers
- **Opportunity:** load shifting; heating/cooling = large load + comfort slack (few °C, few hours)
- **Objective:** minimize bill s.t. comfort band; physical model → linear opt → ms solve, global optimum, interpretable, explainable (e.g. "heat battery full at hour X")
- **Value prop:** lower bills; grid flexibility; auditable (unlike black-box ML)

---

## 2. Intuitive Dynamics
- **Reframe:** pricing problem → thermal storage problem
- **Why storage:** electricity most perishable good; thermal mass = cheapest storage
- **Battery anatomy:** two chambers (air: thin/fast, wall: wide/slow) + leaky pipe to outside → chamber volume = C, pipe width = R
- **Key terms:** C (energy/°C), R (leakage rate), comfort band (hard constraint)

---

## 3. Physical Model
- **2R2C circuit:** masses→capacitors, insulation→resistors, heater→current source
- **State:** T_air, T_wall over time
- **Inputs:** T_ext (disturbance), Q_heater (control)

---

## 4. Modeling Alternatives
| Model | Pro | Con |
|---|---|---|
| 1R1C | simple | misses wall dynamics |
| EnergyPlus | accurate | heavy, can't parameterize, not real-time |
| **2R2C** | wall inertia + linear | lumped approximation |

---

## 5. Optimization Alternatives
| Method | Pro | Con |
|---|---|---|
| Thermostat | simple | myopic, price-blind |
| RL | no explicit model needed | no hard guarantees; data-hungry; bad for infra (note: less critical than infra but hurts retention) |
| **Linear MPC** | global opt; hard constraints; receding horizon; standard solvers (CVXPY); slack vars → headroom/limits visible | needs linear model |

---

## 6. Tools
| Tool | Role |
|---|---|
| Python + CVXPY | daily runs, real-time; linear/convex formulation |
| Julia + JuMP.jl | multi-year backtests; ~10x faster than Python solvers |
| pandas/numpy | time-series |
| ENTSO-E API | price forecasts |
| OpenWeatherMap | T_ext forecasts |
| python-kasa | smart plug control |

---

## 7. Results
*(to fill)*

## 8. Conclusion
- cost savings + comfort guarantees summary

## 9. Future Work
- coordinator-level optimization
- stochastic elements
- V2G energy management (see separate study)


## Draft

# Case Study: Smart Heating Optimization

## 0. The story

A friend wanted to automate the use of his electric stove around dynamic electricity tariffs — prices that swing hour by hour — without having to check an app manually every hour.

## 1. The Business Problem

- **Stakeholder:** Individual user 


USP conditions: 
safety gurantees in comfort temperature, and a low electricity bill.


- **Context:** Spain has one of Europe's highest shares of renewable generation, which drives large intra-day price swings (the "duck curve"). As solar penetration grows, the spread between cheap and expensive hours keeps widening, and dynamic pricing plans now expose these fluctuations directly to consumers.
- **Opportunity:** *Load shifting* — consuming energy at lower-cost times without changing total consumption. Heating and cooling are especially attractive targets: they are large loads, and comfort tolerates a few degrees and several hours of flexibility, which is exactly the slack we need.
- **Objective:** Minimize the electricity bill while keeping indoor temperature within a defined comfort band. [add and objective about gurantees: we want to guarantee that the temperature will not drop below the minimum temperature, and that the user will never pay exorbitant prices] A physical model of the building lets us formulate this as a convex optimization problem: solved in milliseconds, globally optimal, and fully interpretable. [add an example, like seeing that the heat battery reached it full capacity in this time frame]
- **Value Proposition:** Lower bills for consumers; demand-side flexibility for the grid.

---

## 2. Intuitive Explanation of the Dynamics

The key reframe: *this is a thermal storage problem*. A house stores heat the way a battery stores charge — and once we see it that way, the path to a formal model becomes clear.

- **Why thermal storage?** Electricity is the most perishable good there is: the instant demand rises, the grid must respond, and any surplus vanishes immediately. Thermal mass is one of the cheapest forms of storage available — pre-heat the house when power is cheap, then let it coast through expensive hours.

- **Building as a battery — the intuition:** Picture two chambers connected by a narrow pipe. The *air chamber* is thin: a small amount of heat raises its temperature quickly. The *wall chamber* is wide: it absorbs a lot of energy before its temperature moves much. A second, leaky pipe connects the wall to the outdoors; how narrow that pipe is reflects how well-insulated the building is. The chamber volumes map to *thermal capacity* (energy per degree), and the pipe widths map to *thermal resistance* (how fast heat leaks). These are exactly the constants used in the model.

- **Key Terms:**
  - *Heat Capacity (C):* Energy required to raise a component's temperature by 1 °C.
  - *Thermal Resistance (R):* Rate at which heat leaks between two components (or to the outside).
  - *Comfort Band:* The acceptable temperature range (e.g., 18 °C – 24 °C) — the hard constraint the optimizer must never violate.

---

## 3. The Physical Model

The battery intuition maps directly to an electrical circuit analogy: thermal masses → capacitors, insulation → resistors, heater power → controlled current source. This yields a **2R2C model** (Two-Resistor, Two-Capacitor), a standard pattern in building energy simulation.

- **Components:**
  - Two thermal masses: Indoor Air and the Insulated Wall.
  - Two thermal resistances: Air ↔ Wall, and Wall ↔ Exterior.
- **System State:** Air and wall temperatures over time.
- **Inputs:** Exterior temperature (disturbance) and heater/cooler power (control variable).

---

## 4. Modeling Alternatives & Selection Justification

| Model | Pros | Cons |
|---|---|---|
| **1R1C (1st-order)** | Extremely simple | Too simplistic; misses slow wall dynamics; over-optimistic about heat retention |
| **White-box / FEM (e.g. EnergyPlus)** | Highly accurate, detailed spatial resolution | Computationally heavy, hard to parameterize for a typical home, not suited for fast online optimization |
| **2R2C (our choice)** | Captures slow wall vs. fast air dynamics; remains linear | Lumped approximation — not suitable for highly asymmetric or unusual buildings |

**Why 2R2C:** The smallest model that captures the essential wall-inertia dynamic while staying *linear*. Linearity is what allows us to embed the model directly inside a convex optimization problem.

---

## 5. Optimization Method & Alternatives

**The challenge:** find the optimal heater command sequence over a 24-hour horizon, subject to linear system dynamics and hard temperature constraints.

[add comment about that ML without guarantees is not as critical as infrastructure, but still quite important for user retention]

| Method | Pros | Cons |
|---|---|---|
| **Rule-based (thermostat)** | Simple, off-the-shelf | Myopic; no concept of price; cannot pre-heat |
| **Reinforcement Learning** | Handles non-linear dynamics without explicit modeling | Requires extensive training data; no hard constraint guarantees on comfort |
| **Convex Optimization / MPC (our choice)** | [change to linear optimization] Global optimum guaranteed; hard constraints satisfied by construction; re-solves hourly with fresh forecasts | Requires a linear model — non-linear dynamics would break convexity |

**Why convex optimization:** With a linear physical model and a linear cost function (power × price), the full problem is convex. 
[comment that we can use more powerful standard solvers like CVXPY]

Solvers exploit this structure to find the global optimum in known timeframes. Temperature bounds are hard constraints, not soft penalties — comfort is guaranteed, not just encouraged.

---

## 6. Tools & Tech Stack

| Tool | Role |
|---|---|
| **Python + CVXPY** | Optimization formulation for daily runs and real-time execution [CVXPY specifically for linear and convex optimization problems] |
| **Julia + JuMP.jl** | High-performance solver for large-scale historical backtests where Python becomes the bottleneck [python solvers run 10x slower ] |

[separate julia vs python as two separate options]


| Tool | Role |
|---|---|
| **pandas / numpy** | Time-series manipulation of prices and temperatures |
| **ENTSO-E API** | Day-ahead electricity price forecasts |
| **OpenWeatherMap API** | Exterior temperature forecasts |
| **python-kasa** | Async control of the TP-Link smart plug |


## Results 

## 7. Conclusion

[add a summary of the results, highlighting the cost savings and comfort guarantees]

## 8. Future Work

[optimization on the coordinator level, adding stochastic elements, see my study on V2G energy management]

[]