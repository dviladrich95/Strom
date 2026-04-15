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

<!----------------------------------------------------------------------------->
<!-- SKETCH — structural overview only; iterate here, do not develop content -->
<!----------------------------------------------------------------------------->

## Sketch

**Story:** friend [Jan Balanya](https://janbalanya.com/), electric stove, dynamic tariffs, wants automation — a full-stack project: API data collection → physical modeling → convex optimization → IoT device control; same principles companies are now scaling to industrial demand response across Europe

---

### 1. Business Problem
- **Stakeholder:** individual homeowner on variable tariff
- **USP:** comfort guarantees (T_min, price ceiling) + low bill
- **Context:** Spain → duck curve → widening price spread → dynamic plans expose fluctuations to consumers
- **Opportunity:** load shifting; heating/cooling = large load + comfort slack (few °C, few hours)
- **Objective:** minimize bill s.t. comfort band + price ceiling; physical model → linear opt → ms solve, global optimum, interpretable (e.g. "heat battery full at hour X")
- **Value prop:** lower bills; grid flexibility; auditable (unlike black-box ML)

---

### 2. Intuitive Dynamics
- **Reframe:** pricing problem → thermal storage problem
- **Why storage:** electricity most perishable good; thermal mass = cheapest storage
- **Battery anatomy:** two chambers (air: thin/fast, wall: wide/slow) + leaky pipe to outside → chamber volume = C, pipe width = R
- **Key terms:** $C$ (energy/°C), $R$ (leakage rate), comfort band (hard constraint)
- **Cooling:** same model, $Q_{cool}$ with negative sign; passive alternatives (shutters → linear; windows → bilinear → Future Work)

---

### 3. Physical Model
- **2R2C circuit:** masses→capacitors, insulation→resistors, heater→current source
- **State:** T_air, T_wall over time
- **Inputs:** T_ext (disturbance), Q_heater (control)
- **Equations:** continuous-time ODEs at air–wall and wall–exterior interfaces → discretized state-space for optimization

---

### 4. Modeling Alternatives
| Model | Pro | Con |
|---|---|---|
| 1R1C | simple | misses wall dynamics |
| EnergyPlus | accurate | heavy, not real-time |
| **2R2C** | wall inertia + linear | lumped approximation |

- **Expanded view:** 2-equation system in calculus form → matrix form (Ẋ = AX + Bu); analogies to hydraulic networks and RC circuits in EE

---

### 5. Optimization Alternatives
| Method | Pro | Con |
|---|---|---|
| Thermostat | simple | myopic, price-blind |
| RL | no explicit model needed | no hard guarantees; data-hungry; hurts user retention |
| **Linear MPC** | global opt; hard constraints; receding horizon; CVXPY; interpretable | needs linear model |

- **Model in code:** discretized matrix form as in `find_heating_output`; state-space propagation as equality constraint; price vector as linear cost coefficient

---

### 6. Tools
| Tool | Role |
|---|---|
| Python + CVXPY | daily runs, real-time |
| Julia + JuMP.jl | large-scale backtests (~10× faster) |
| pandas/numpy | time-series |
| ENTSO-E API | price forecasts |
| OpenWeatherMap | T_ext forecasts |
| python-kasa | smart plug control |

---

### 7. Results
- **Baseline:** 24h-smoothed exterior temp clipped to comfort band
- **Single day (Nov 25):** pre-heats at price floor; ~10% savings
- **Charging event:** 18°C → ~21°C; $T_{min}$ never breached
- **Two-year (+ cooling):** 17% reduction / 66€; gap widest in spring + autumn (most scheduling slack)
- **Temperature volatility:** ~4°C avg swings; gradual so tolerable on average, but outlier spikes need capping for consumer trust

### 8. Conclusion
- full-stack applied math: physics → LP → real-time IoT control
- hard comfort guarantees by construction, not soft penalties; globally optimal; auditable
- quantified savings: >10% single day, 17% over two years

### 9. Future Work
- coordinator-level optimization across devices / households
- stochastic elements (price and weather uncertainty)
- V2G energy management (see separate study)
- temperature regularization term to smooth jumps (~4°C avg; low priority but worth testing with users)
- passive thermal control: shutters → new linear solar-gain input (stays LP); windows → $R_{exterior}$ as decision variable → bilinear programming → SDP relaxations

<!----------------------------------------------------------------------------->
<!-- SKELETON — develop this into the final case study                       -->
<!----------------------------------------------------------------------------->

---
---

## 0. The Story

[Jan Balanya](https://janbalanya.com/) wanted to automate the use of his electric stove around dynamic electricity tariffs — prices that swing hour by hour — without having to check an app manually every hour.

What started as a weekend project is a full-stack applied mathematics exercise: it spans API data collection from energy and weather services, physical modeling of the building's thermal dynamics, convex optimization to produce the heating schedule, and direct IoT control of a smart plug to execute it. Each layer connects to the next, and each has a real engineering justification.

The same principles are what companies are now scaling to industrial-level demand response across Europe — coordinating fleets of devices, industrial loads, and storage assets to shift consumption away from expensive hours. Strom is a single-household proof of concept for exactly that idea.

---

## 1. The Business Problem

- **Stakeholder:** Individual user on a variable tariff.

USP conditions: safety guarantees on comfort temperature, and a low electricity bill.

- **Context:** Spain has one of Europe's highest shares of renewable generation, which drives large intra-day price swings (the "duck curve"). As solar penetration grows, the spread between cheap and expensive hours keeps widening, and dynamic pricing plans now expose these fluctuations directly to consumers.
- **Opportunity:** *Load shifting* — consuming energy at lower-cost times without changing total consumption. Heating and cooling are especially attractive targets: they are large loads, and comfort tolerates a few degrees and several hours of flexibility, which is exactly the slack we need.
- **Objective:** Minimize the electricity bill while keeping indoor temperature within a defined comfort band, guaranteeing that the temperature never drops below T_min and that the user is never exposed to exorbitant price spikes. A physical model of the building lets us formulate this as a convex optimization problem: solved in milliseconds, globally optimal, and fully interpretable — for example, the optimizer can tell you "pre-heat fully between 02:00 and 05:00 when the price hits its floor, then coast through the morning peak."
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

Active cooling follows the same model in reverse: $Q_{cool}$ enters the air equation with a negative sign, and the same price-aware scheduling logic applies — pre-cool before the afternoon peak and coast through it. This is equally relevant for Spanish households managing summer heat.

---

## 3. The Physical Model

The battery intuition maps directly to an electrical circuit analogy: thermal masses → capacitors, insulation → resistors, heater power → controlled current source. This yields a **2R2C model** (Two-Resistor, Two-Capacitor), a standard pattern in building energy simulation.

- **Components:**
  - Two thermal masses: Indoor Air and the Insulated Wall.
  - Two thermal resistances: Air ↔ Wall, and Wall ↔ Exterior.
- **System State:** Air and wall temperatures over time.
- **Inputs:** Exterior temperature (disturbance) and heater/cooler power (control variable).
- **Governing equations:** Heat flow at each interface follows Newton's law of cooling. At the air–wall interface: C_air · dT_air/dt = (T_wall − T_air)/R_interior + Q_heater. At the wall–exterior interface: C_wall · dT_wall/dt = (T_air − T_wall)/R_interior + (T_ext − T_wall)/R_exterior. These two coupled first-order ODEs are then discretized (Euler forward) into a state-space form suitable for embedding in the optimization.

---

## 4. Modeling Alternatives & Selection Justification

| Model | Pros | Cons |
|---|---|---|
| **1R1C (1st-order)** | Extremely simple | Too simplistic; misses slow wall dynamics; over-optimistic about heat retention |
| **White-box / FEM (e.g. EnergyPlus)** | Highly accurate, detailed spatial resolution | Computationally heavy, hard to parameterize for a typical home, not suited for fast online optimization |
| **2R2C (our choice)** | Captures slow wall vs. fast air dynamics; remains linear | Lumped approximation — not suitable for highly asymmetric or unusual buildings |

**Why 2R2C:** The smallest model that captures the essential wall-inertia dynamic while staying *linear*. Linearity is what allows us to embed the model directly inside a convex optimization problem.

The two ODEs above can be written compactly in matrix form:

```
dX/dt = A·X + B·u + d
```

where X = [T_air, T_wall]ᵀ, u = Q_heater, and d encodes the T_ext disturbance. This is the same state-space structure familiar from hydraulic network analysis (where A encodes conductance between pressure nodes) and RC circuit analysis in electrical engineering (where the same equation governs node voltages). The analogy is not coincidental — heat, charge, and fluid obey the same conservation laws at each node.

---

## 5. Optimization Method & Alternatives

**The challenge:** find the optimal heater command sequence over a 24-hour horizon, subject to linear system dynamics and hard temperature constraints.

Note on RL: the lack of hard comfort guarantees is less critical than infrastructure stability, but it matters significantly for user retention — a single cold night is enough for a user to lose confidence in the system permanently.

| Method | Pros | Cons |
|---|---|---|
| **Rule-based (thermostat)** | Simple, off-the-shelf | Myopic; no concept of price; cannot pre-heat |
| **Reinforcement Learning** | Handles non-linear dynamics without explicit modeling | Requires extensive training data; no hard constraint guarantees on comfort |
| **Linear Optimization / MPC (our choice)** | Global optimum guaranteed; hard constraints satisfied by construction; re-solves hourly with fresh forecasts | Requires a linear model — non-linear dynamics would break convexity |

**Why linear optimization:** With a linear physical model and a linear cost function (power × price), the full problem is linear. This lets us use powerful standard solvers — CVXPY in particular, which provides a clean Python interface for formulating linear and convex problems and dispatches to highly optimized backends (CLARABEL, OSQP, etc.).

Solvers exploit this structure to find the global optimum in known, bounded time. Temperature bounds enter as hard constraints, not soft penalties — comfort is guaranteed, not just encouraged.

The discretized model as implemented in `find_heating_output` expresses the state-space propagation X_{t+1} = A·X_t + B·u_t + d_t as equality constraints in the optimization problem. The heater output u_t is the decision variable at each timestep; the day-ahead price vector enters as the linear cost coefficient. The result is a single LP solved once per hour over the full 24-step horizon.

---

## 6. Tools & Tech Stack

**Optimization runtimes:**

| Tool | Role |
|---|---|
| **Python + CVXPY** | Primary runtime for daily runs and real-time execution; CVXPY natively handles linear and convex problem structures |
| **Julia + JuMP.jl** | High-performance option for large-scale historical backtests; Python solvers run ~10× slower at this scale |

**Data & infrastructure:**

| Tool | Role |
|---|---|
| **pandas / numpy** | Time-series manipulation of prices and temperatures |
| **ENTSO-E API** | Day-ahead electricity price forecasts |
| **OpenWeatherMap API** | Exterior temperature forecasts |
| **python-kasa** | Async control of the TP-Link smart plug |

---

## 7. Results

All comparisons are against a **baseline policy**: a 24-hour rolling mean of exterior temperature, clipped to $[T_{min}, T_{max}]$, tracked by a small optimizer with a negligible cost term to prevent simultaneous heating and cooling.

### Single Day — 25 November 2024

![Historical Comparison - Nov 25th 2024](../plots/compare_costs_temps_Barcelona_25th_Nov.png)
*Optimal vs. baseline policy on 25 November 2024.*

- The optimizer pre-heats during the duck-curve price floor at the center of the day and at night, then coasts through the expensive morning peak.
- Cost savings: **~10%** vs. the baseline policy on this day.
- The 25th was a day of exceptionally low prices for a prolonged window — the largest single charging event in the backtest period.

### November 2024

![Monthly Comparison - Nov 2024](../plots/compare_costs_temps_Barcelona_Nov.png)
*Optimal vs. baseline policy across the second half of November 2024.*

- Temperature spikes are visible throughout the month wherever the price floor is deep enough to warrant pre-heating.
- The largest spike (Nov 25) drove $T_{interior}$ from $T_{min} = 18°C$ to approximately $21°C$ — well within the comfort band and illustrating the heat battery being charged to near-full.
- $T_{min}$ was never breached; comfort constraints held throughout.

### Two-Year Backtest — March 2023 to March 2025

![Long-term Comparison - 2023-2025](../plots/compare_costs_temps_Barcelona_Mar23_Mar25.png)
*Optimal vs. baseline policy over two years, with cooling included.*

- Cooling was added for this analysis ($Q_{cool}$ entering the air equation with a negative sign; same LP framework).
- Cumulative savings: **66€** over two years.
- Relative reduction: **17%** vs. total cost of the baseline policy.
- The 7-day rolling average cost lines separate most visibly during spring and autumn. The likely reason: when exterior temperature sits naturally inside $[T_{min}, T_{max}]$, the optimizer has the full comfort band as scheduling slack. In deep winter the heater is largely occupied just compensating heat losses, leaving less room to shift load toward cheap hours.
- The optimized temperature profile shows spikes toward $T_{min}$ in winter and toward $T_{max}$ in summer — pre-charging the heat battery at cheap hours. This is positive in one sense: the interior temperature sometimes reaches closer to the center of the comfort band while saving money. But the same behaviour means higher temperature volatility for occupants (~4°C average swings), which could feel uncomfortable depending on the user — think repeatedly switching between sweater and no sweater. Since swings happen gradually over hours rather than minutes, they are less disruptive than the absolute magnitude suggests. However, outlier events with larger swings do occur and are a more pressing concern: a sudden large jump in interior temperature would erode consumer trust quickly, regardless of the average case. Capping these outliers — via a regularization term or a hard rate constraint on $\Delta T$ per timestep — is a natural next step before any wider deployment.

---

## 8. Conclusion

*(to fill — summarize cost savings achieved, comfort guarantees maintained throughout, and how the price-ceiling objective protects users from spike exposure)*

---

## 9. Future Work

- Coordinator-level optimization across multiple devices or households
- Adding stochastic elements (price and weather uncertainty)
- V2G energy management (see separate study)
- A regularization term penalizing large jumps in interior temperature (e.g. $\sum_t |T_{t+1} - T_t|$) or a hard rate constraint on $\Delta T$ per timestep could smooth the profile without significantly increasing cost. Average ~4°C swings are gradual enough to be tolerable, but outlier events with larger swings need to be capped before wider deployment — a single jarring temperature spike is enough to break consumer trust.
- Passive thermal control: roller shutters modulating solar gain introduce a new linear input term and stay within the current LP framework. Opening windows to ventilate is more interesting — it makes $R_{exterior}$ a decision variable, introducing a product $\alpha \cdot T_{wall}$ of two optimization variables. This is a bilinear term that breaks convexity, placing the extended problem firmly in the terrain of bilinear programming, addressable via SDP relaxations.
