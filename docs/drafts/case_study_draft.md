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
- **USP:** comfort guarantees ($T_{min}$) + low bill; price floor $P_{base}$ keeps consumer price realistic vs. wholesale ENTSO-E data
- **Context:** Spain → duck curve → widening price spread → dynamic plans expose fluctuations to consumers
- **Opportunity:** load shifting; heating/cooling = large load + comfort slack (few °C, few hours)
- **Objective:** minimize bill s.t. comfort band + price ceiling; physical model → linear opt → ms solve, global optimum, interpretable (e.g. "heat battery full at hour X")
- **Value prop:** lower bills; grid flexibility; auditable (unlike black-box ML)

---

### 2. Intuitive Dynamics
- **Reframe:** pricing problem → thermal storage problem
- **Why storage:** electricity most perishable good; thermal mass = cheapest storage
- **Battery anatomy:** two chambers (air: thin/fast, wall: wide/slow) + leaky pipe to outside → $C$ = chamber volume, $R$ = pipe width
- **Newton cooling:** $C \frac{dT}{dt} = \frac{T_{neighbour}-T}{R} + Q$ — one equation per lump, one per interface
- **Cooling:** same model, $Q_{cool}$ with negative sign; passive alternatives (shutters → linear; windows → bilinear → Future Work)

---

### 3. Modeling Alternatives & Model Selection
| Model | Granularity | Pro | Con |
|---|---|---|---|
| 1R1C | air only | simple | misses wall dynamics; over-optimistic |
| **2R2C** | air + wall | wall inertia + linear | lumped approximation |
| EnergyPlus | full spatial | accurate | heavy; not real-time |

- **Selection:** 2R2C = minimal model capturing two time constants (fast air, slow wall); 1R1C loses heat-battery effect; FEM breaks linearity

---

### 4. Physical Model
- **2R2C circuit:** masses→capacitors, insulation→resistors, heater→current source
- **State:** $T_{air}$, $T_{wall}$; inputs: $T_{ext}$ (disturbance), $Q_{heater}$ (control)
- **Equations:** $\dot{\mathbf{T}} = A\mathbf{T} + B\mathbf{u}_t + \mathbf{d}_t$; $A$ encodes inter-node conductance; $B$ diagonal (inverse capacities); $\mathbf{u}_t = [Q_{heater} u_t,\ 0]^\top$, $u_t \in [0,1]$ heater fraction; $\mathbf{d}_t \neq 0$ — $T_{ext}$ forcing in wall row
- **Parameters:** $C_{air}=0.56$, $C_{wall}=3.5$, $R_{int}=1.0$, $R_{ext}=6.06$; estimated via [Protons for Breakfast method](https://protonsforbreakfast.wordpress.com/2022/12/19/estimating-the-heat-capacity-of-my-house/)

---

### 5. Optimization Alternatives
| Method | Pro | Con |
|---|---|---|
| Thermostat | simple | myopic, price-blind |
| RL | no explicit model needed | no hard guarantees; data-hungry; hurts user retention |
| **Linear MPC** | global opt; hard constraints; receding horizon; CVXPY; interpretable | needs linear model |

- **Discretization:** Euler step $\mathbf{T}_{t+1} = \mathbf{T}_t + \Delta t(A\mathbf{T}_t + B\mathbf{u}_t + \mathbf{d}_t)$ → linear equality constraints
- **LP in code:** [`find_heating_output`](../../strom/optimization_utils.py) (line 96); $u_t \in [0,1]$ decision variable; price vector as cost coefficient

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
- **Price data realism:** The ENTSO-E API provides day-ahead prices at the megawatt industrial scale, where prices can reach zero or go negative during oversupply. A real consumer tariff always has a minimum — network charges, taxes, and provider margins don't disappear. A constant price floor $P_{base}$ is added to all ENTSO-E prices before optimization, bridging the gap between wholesale market data and the price a household actually pays.
- **Value Proposition:** Lower bills for consumers; demand-side flexibility for the grid.

---

## 2. Intuitive Explanation of the Dynamics

The key reframe: *this is a thermal storage problem*. A house stores heat the way a battery stores charge — and once we see it that way, the path to a formal model becomes clear.

- **Why thermal storage?** Electricity is the most perishable good there is: the instant demand rises, the grid must respond, and any surplus vanishes immediately. Thermal mass is one of the cheapest forms of storage available — pre-heat the house when power is cheap, then let it coast through expensive hours.

- **Building as a battery — the intuition:** Picture two chambers connected by a narrow pipe. The *air chamber* is thin: a small amount of heat raises its temperature quickly. The *wall chamber* is wide: it absorbs a lot of energy before its temperature moves much. A second, leaky pipe connects the wall to the outdoors; how narrow that pipe is reflects how well-insulated the building is. The chamber volumes map to *thermal capacity* (energy per degree), and the pipe widths map to *thermal resistance* (how fast heat leaks). These are exactly the constants used in the model.

- **Key Terms:**
  - *Heat Capacity ($C$):* Energy required to raise a component's temperature by 1 °C.
  - *Thermal Resistance ($R$):* Rate at which heat leaks between two components (or to the outside).
  - *Comfort Band:* The acceptable temperature range (e.g., 18 °C – 24 °C) — the hard constraint the optimizer must never violate.

- **Newton's law of cooling** governs how each lump exchanges heat with its neighbours:
  $$C \frac{dT}{dt} = \frac{T_{neighbour} - T}{R} + Q$$
  A lump heats up when its neighbour is warmer, cools down when it is colder, and any externally injected power $Q$ shifts the balance. Each interface in the building adds one such equation to the system.

Active cooling follows the same model in reverse: $Q_{cool}$ enters the air equation with a negative sign, and the same price-aware scheduling logic applies — pre-cool before the afternoon peak and coast through it. This is equally relevant for Spanish households managing summer heat.

---

## 3. Modeling Alternatives & Model Selection

A building can be modeled at many levels of granularity — from a single lumped temperature to full finite-element spatial resolution. The right choice depends on what dynamics matter for the application.

| Model | Granularity | Pros | Cons |
|---|---|---|---|
| **1R1C (1 lump)** | Air only | Extremely simple | Misses wall dynamics; over-optimistic about heat retention |
| **2R2C (2 lumps — our choice)** | Air + wall | Captures slow wall vs. fast air dynamics; remains linear | Lumped approximation |
| **White-box / FEM (e.g. EnergyPlus)** | Full spatial | Highly accurate | Computationally heavy; hard to parameterize; not suited for real-time optimization |

**Why 2R2C:** A house has two thermally relevant lumps — the indoor air, which responds quickly, and the insulated wall mass, which responds slowly. This difference in time constants is exactly what creates the heat-battery effect: the wall stores energy over hours, not minutes. A 1R1C model collapses both into one and loses this dynamic entirely. A full FEM model adds resolution that isn't needed here and breaks the linearity required to embed the model inside a convex optimization. The 2R2C is the minimal model that captures the essential physics.

---

## 4. The Physical Model

With the 2R2C model selected, the building maps to an electrical circuit: thermal masses → capacitors, insulation → resistors, heater power → controlled current source. Applying Newton's law of cooling at each interface gives one ODE per lump.

- **System state:** $T_{air}$ and $T_{wall}$ over time.
- **Inputs:** $T_{ext}$ (disturbance) and $Q_{heater}$ / $Q_{cool}$ (control).

At the air–wall interface:
$$C_{air} \frac{dT_{air}}{dt} = \frac{T_{wall} - T_{air}}{R_{interior}} + Q_{heater}$$

At the wall–exterior interface:
$$C_{wall} \frac{dT_{wall}}{dt} = \frac{T_{air} - T_{wall}}{R_{interior}} + \frac{T_{ext} - T_{wall}}{R_{exterior}}$$

The two equations are coupled through $T_{wall}$ and $T_{air}$ and can be written together in matrix form:

$$\dot{\mathbf{T}} = A\mathbf{T} + B\mathbf{u} + \mathbf{d}_t$$

where:
- $\mathbf{T} = [T_{air},\ T_{wall}]^\top$ — the state vector of the two lump temperatures
- $\mathbf{u}_t = [u_{air,t},\ u_{wall,t}]^\top = [Q_{heater} \cdot u_t,\ 0]^\top$ — the control vector; $u_t \in [0,1]$ is the heater output fraction at timestep $t$; only the air lump receives direct heat input, so the wall component is always zero
- $A$ — the conductance matrix; each off-diagonal entry is a heat pathway between two lumps, each diagonal entry is the total heat loss rate out of that lump
- $B$ — the input matrix; scales each control component by the inverse capacity of the corresponding lump
- $\mathbf{d}_t$ — a time-varying disturbance vector carrying the exterior temperature forcing (see below)

$$A = \begin{pmatrix} -\dfrac{1}{R_{int} C_{air}} & \dfrac{1}{R_{int} C_{air}} \\[8pt] \dfrac{1}{R_{int} C_{wall}} & -\dfrac{\frac{1}{R_{int}} + \frac{1}{R_{ext}}}{C_{wall}} \end{pmatrix}, \qquad B = \begin{pmatrix} \dfrac{1}{C_{air}} & 0 \\[8pt] 0 & \dfrac{1}{C_{wall}} \end{pmatrix}, \qquad \mathbf{u}_t = \begin{pmatrix} Q_{heater} \cdot u_t \\[4pt] 0 \end{pmatrix}$$

Component by component:

| Entry | Expression | Meaning |
|---|---|---|
| $A_{11}$ | $-\dfrac{1}{R_{int}\,C_{air}}$ | Heat loss rate of air to wall — air cools as it drives heat through $R_{int}$ |
| $A_{12}$ | $+\dfrac{1}{R_{int}\,C_{air}}$ | Heat gain rate of air from wall — wall drives heat back through $R_{int}$ |
| $A_{21}$ | $+\dfrac{1}{R_{int}\,C_{wall}}$ | Heat gain rate of wall from air |
| $A_{22}$ | $-\dfrac{\frac{1}{R_{int}}+\frac{1}{R_{ext}}}{C_{wall}}$ | Total heat loss rate of wall — leaks to air via $R_{int}$ and to outside via $R_{ext}$ |
| $B_{11}$ | $+\dfrac{1}{C_{air}}$ | Scales air control input by inverse air capacity |
| $B_{22}$ | $+\dfrac{1}{C_{wall}}$ | Scales wall control input by inverse wall capacity (multiplied by zero — no wall heater) |
| $u_{air,t}$ | $Q_{heater} \cdot u_t$ | Actual heater power delivered to air at timestep $t$; $u_t \in [0,1]$ is the decision variable |
| $u_{wall,t}$ | $0$ | No direct heat source on the wall |

With numerical values ($C_{air}=0.56$, $C_{wall}=3.5$, $R_{int}=1.0$, $R_{ext}=6.06$):

$$A \approx \begin{pmatrix} -1.79 & 1.79 \\[4pt] 0.29 & -0.33 \end{pmatrix}, \qquad B \approx \begin{pmatrix} 1.79 & 0 \\[4pt] 0 & 0.29 \end{pmatrix}$$

The two diagonal entries of $A$ have very different magnitudes — $|A_{11}| \approx 1.79$ vs $|A_{22}| \approx 0.33$ — which directly reflects the two time constants: air responds roughly 5× faster than the wall. This is the separation of timescales that makes the heat-battery effect possible.

The same conductance-matrix structure appears in hydraulic networks and RC circuits in electrical engineering — heat, charge, and fluid all obey the same conservation law at each node.

**On the disturbance vector $\mathbf{d}_t$:** it is not zero. Expanding the wall–exterior heat flux: $\frac{T_{ext} - T_{wall}}{R_{ext}} = -\frac{T_{wall}}{R_{ext}} + \frac{T_{ext}}{R_{ext}}$. The $-T_{wall}/R_{ext}$ term is absorbed into $A_{22}$ (it is proportional to a state variable); the $+T_{ext}/R_{ext}$ term cannot enter $A$ (since $T_{ext}$ is not a state) nor $B$ (since it is not a control) — so it becomes a time-varying forcing term driven by the measured exterior temperature forecast:

$$\mathbf{d}_t = \begin{pmatrix} 0 \\[4pt] \dfrac{T_{ext}(t)}{R_{ext}\, C_{wall}} \end{pmatrix}$$

The air row is zero because the heater is the only external input to the air lump — its thermal coupling to the wall is already fully captured in $A$.

**Parameter values:** Estimated following the methodology in [Estimating the heat capacity of my house](https://protonsforbreakfast.wordpress.com/2022/12/19/estimating-the-heat-capacity-of-my-house/).

| Parameter | Value | Units | Description |
|---|---|---|---|
| $C_{air}$ | 0.56 | kWh/°C | Heat capacity of indoor air |
| $C_{wall}$ | 3.5 | kWh/°C | Heat capacity of the insulated wall |
| $R_{interior}$ | 1.0 | °C/kW | Thermal resistance between air and wall |
| $R_{exterior}$ | 6.06 | °C/kW | Thermal resistance between wall and outside |
| $Q_{heater}$ | 2.0 | kW | Heater power |
| $T_{min}$ / $T_{max}$ | 18 / 24 | °C | Comfort band |

<!-- TODO: expand on the estimation procedure — what measurements or inputs does the method require, and how were they obtained for Jan's apartment? -->

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

To embed the continuous model inside an optimization problem, the ODEs are discretized with a forward Euler step of $\Delta t = 1\text{h}$:

$$\mathbf{T}_{t+1} = \mathbf{T}_t + \Delta t \left( A\mathbf{T}_t + B\mathbf{u}_t + \mathbf{d}_t \right)$$

This turns each timestep's state propagation into a linear equality constraint. The full LP — 24 such constraints, heater output $u_t \in [0,1]$ as decision variable, comfort bounds as inequality constraints, and the day-ahead price vector as cost coefficient — is implemented in [`find_heating_output`](../../strom/optimization_utils.py) (`strom/optimization_utils.py`, line 96). The result is a single LP solved once per hour over the full 24-step horizon.

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
