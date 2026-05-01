# Case Study: Smart Heating Optimization

## Abstract

<!-- TODO: update savings % once two-year backtest with thermostat baseline completes -->
**~X% cost savings** over two years, zero comfort violations, running live on a smart plug in Barcelona.

A price-aware heating and cooling optimizer for households on dynamic electricity tariffs. Uses a physics-derived thermal model of the building and a linear program solved in milliseconds to shift energy purchases to cheap hours — keeping the house comfortable while automatically exploiting intra-day price swings. The same principles underpin industrial demand-response systems coordinating thousands of devices across European grids.

---

## 0. The Story

[Jan Balanya](https://janbalanya.com/) wanted to automate his electric stove around dynamic electricity tariffs — prices that swing hour by hour — without checking an app manually every hour.

What started as a weekend project is a full-stack applied mathematics exercise: API data collection from energy and weather services, physical modeling of the building's thermal dynamics, convex optimization to produce the heating schedule, and direct IoT control of a smart plug to execute it. Each layer connects to the next, and each has a real engineering justification.

---

## 1. The Business Problem

**Stakeholder:** Individual household on a variable tariff.

**Context:** Spain has one of Europe's highest shares of renewable generation, driving large intra-day price swings (the "duck curve"). As solar penetration grows, the spread between cheap and expensive hours keeps widening, and dynamic pricing plans now expose these fluctuations directly to consumers.

**Opportunity:** *Load shifting* — consuming energy at lower-cost times without changing total consumption. Heating and cooling are attractive targets: large loads, and comfort tolerates a few degrees and several hours of flexibility, which is exactly the slack needed.

**Objective:** Minimize the electricity bill while keeping indoor temperature within a defined comfort band. A physical model of the building lets us formulate this as a linear program: solved in milliseconds, globally optimal, and fully interpretable — "pre-heat fully between 02:00 and 05:00 when the price hits its floor, then coast through the morning peak."

**Price data realism:** ENTSO-E day-ahead prices are at megawatt industrial scale, where prices can go zero or negative during oversupply. A real consumer tariff always has a minimum — network charges, taxes, and provider margins don't disappear. A constant floor $P_{base}$ is added to all ENTSO-E prices before optimization.

**Value proposition:** Lower bills for consumers; demand-side flexibility for the grid.

---

## 2. Intuitive Explanation of the Dynamics

The key reframe: *this is a thermal storage problem*. A house stores heat the way a battery stores charge — and once we see it that way, the path to a formal model becomes clear.

**Why thermal storage?** Electricity is the most perishable good there is: any surplus vanishes immediately. Thermal mass is one of the cheapest storage forms available — pre-heat the house when power is cheap, let it coast through expensive hours.

**Building as a battery:** Picture two chambers connected by a narrow pipe. The *air chamber* is thin — a small amount of heat raises its temperature quickly. The *wall chamber* is wide — it absorbs a lot of energy before its temperature moves much. A second, leaky pipe connects the wall to the outdoors; how narrow that pipe is reflects how well-insulated the building is. Chamber volumes map to *thermal capacity* (energy per degree), pipe widths map to *thermal resistance* (how fast heat leaks).

**Key terms:**
- *Heat Capacity ($C$):* Energy required to raise a component's temperature by 1 °C.
- *Thermal Resistance ($R$):* Rate at which heat leaks between two components (or to the outside).
- *Comfort Band:* The acceptable temperature range — the hard constraint the optimizer must never violate.

**Newton's law of cooling** governs how each lump exchanges heat with its neighbours:

$$C \frac{dT}{dt} = \frac{T_{\text{neighbour}} - T}{R} + Q$$

Each interface in the building adds one such equation to the system. Active cooling follows the same model with $Q_{cool}$ entering with a negative sign.

---

## 3. Modeling Alternatives & Model Selection

A building can be modeled at many levels of granularity. The right choice depends on what dynamics matter for the application.

| Model | Granularity | Pros | Cons |
|---|---|---|---|
| **1R1C (1 lump)** | Air only | Extremely simple | Misses wall dynamics; over-optimistic about heat retention |
| **2R2C (2 lumps — chosen)** | Air + wall | Captures slow wall vs. fast air dynamics; remains linear | Lumped approximation |
| **White-box / FEM (e.g. EnergyPlus)** | Full spatial | Highly accurate | Computationally heavy; not suited for real-time optimization |

**Why 2R2C:** The wall stores energy over hours, not minutes — this is the heat-battery effect. A 1R1C model collapses air and wall into one lump and loses this dynamic entirely. A full FEM model adds resolution that isn't needed here and breaks the linearity required to embed the model inside a convex optimization. The 2R2C is the minimal model that captures the essential physics.

---

## 4. The Physical Model

### 4.1 The ODE System

With 2R2C selected, the building maps to an electrical circuit: thermal masses → capacitors, insulation → resistors, heater power → controlled current source.

Applying Newton's law of cooling at each interface gives one ODE per lump:

At the air–wall interface:
$$C_{air} \frac{dT_{air}}{dt} = \frac{T_{wall} - T_{air}}{R_{int}} + Q_{heater}$$

At the wall–exterior interface:
$$C_{wall} \frac{dT_{wall}}{dt} = \frac{T_{air} - T_{wall}}{R_{int}} + \frac{T_{ext} - T_{wall}}{R_{ext}}$$

### 4.2 State-Space Form

The two coupled ODEs can be written in matrix form:

$$\dot{\mathbf{T}} = A\mathbf{T} + B\mathbf{u}_t + \mathbf{d}_t$$

| Symbol | Meaning |
|---|---|
| $\mathbf{T} = [T_{air},\ T_{wall}]^\top$ | State vector |
| $\mathbf{u}_t = [Q_{heater} \cdot u_t,\ 0]^\top$, $u_t \in [0,1]$ | Control vector; only air lump receives direct heat input |
| $A$ | Conductance matrix; off-diagonal = heat pathway, diagonal = total loss rate |
| $B$ | Input matrix; scales by inverse capacity |
| $\mathbf{d}_t$ | Time-varying disturbance carrying exterior temperature forcing |

$$A = \begin{pmatrix} -\dfrac{1}{R_{int} C_{air}} & \dfrac{1}{R_{int} C_{air}} \\[8pt] \dfrac{1}{R_{int} C_{wall}} & -\dfrac{\frac{1}{R_{int}} + \frac{1}{R_{ext}}}{C_{wall}} \end{pmatrix}, \qquad B = \begin{pmatrix} \dfrac{1}{C_{air}} & 0 \\[8pt] 0 & \dfrac{1}{C_{wall}} \end{pmatrix}$$

With numerical values ($C_{air}=0.26$, $C_{wall}=19.1$, $R_{int}=0.42$, $R_{ext}=8.86$):

$$A \approx \begin{pmatrix} -9.16 & 9.16 \\[4pt] 0.125 & -0.131 \end{pmatrix}$$

The diagonal entries differ by a factor of ~70, reflecting strong separation of timescales between air and wall.

**On $\mathbf{d}_t$:** the exterior temperature term $T_{ext}/R_{ext}$ from the wall–exterior flux cannot enter $A$ (not a state variable) nor $B$ (not a control), so it becomes a time-varying forcing:

$$\mathbf{d}_t = \begin{pmatrix} 0 \\[4pt] \dfrac{T_{ext}(t)}{R_{ext}\, C_{wall}} \end{pmatrix}$$

### 4.3 Parameters and Timescales

Parameters identified from Péan et al. (2018)[^pean2018] for a multi-family apartment in Sant Adrià de Besòs (Barcelona), via PRBS excitation of a TRNSYS model validated against metered data.

| Parameter | Value | Units | Description |
|---|---|---|---|
| $C_{air}$ | 0.26 | kWh/°C | Heat capacity of indoor air |
| $C_{wall}$ | 19.1 | kWh/°C | Heat capacity of the insulated wall |
| $R_{int}$ | 0.42 | °C/kW | Thermal resistance, air–wall |
| $R_{ext}$ | 8.86 | °C/kW | Thermal resistance, wall–outside |
| $Q_{heater}$ | 2.0 | kW | Heater power |
| $T_{min}$ / $T_{max}$ | 18 / 24 | °C | Comfort band |

**Timescales:**
- $\tau_{air} = R_{int} \cdot C_{air} \approx 6.5\ \text{min}$ — air responds within minutes
- $\tau_{wall} = R_{ext} \cdot C_{wall} \approx 7\ \text{days}$ — wall stores heat over days

**Air-wall gap at full heater output** ($u=1$, quasi-steady):
$$\Delta T \approx R_{int} \cdot Q_{heater} = 0.42 \times 2.0 = 0.84\ ^\circ\text{C}$$

---

## 5. Optimization Method & Alternatives

**The challenge:** find the optimal heater command sequence over a 24-hour horizon, subject to linear system dynamics and hard temperature constraints.

| Method | Pros | Cons |
|---|---|---|
| **Rule-based (thermostat)** | Simple, off-the-shelf | Myopic; no price awareness; will not pre-heat |
| **Reinforcement Learning** | No explicit model needed | No hard comfort guarantees — a single cold night loses user trust permanently |
| **Linear MPC (chosen)** | Global optimum; hard constraints by construction; re-solves hourly with fresh forecasts | Requires a linear model |

**Why linear optimization:** With a linear physical model and a linear cost function (power × price), the full problem is a linear program. CVXPY dispatches to highly optimized backends (CLARABEL, OSQP) and finds the global optimum in milliseconds. Temperature bounds enter as hard constraints, not soft penalties — comfort is guaranteed, not just encouraged.

**Discretization:** ODEs are discretized with a forward Euler step of $\Delta t = 1\text{h}$:

$$\mathbf{T}_{t+1} = \mathbf{T}_t + \Delta t \left( A\mathbf{T}_t + B\mathbf{u}_t + \mathbf{d}_t \right)$$

Each timestep becomes a linear equality constraint. The full LP — 24 constraints, $u_t \in [0,1]$ decision variable, comfort bounds as inequality constraints, day-ahead price vector as cost coefficient — is implemented in `find_heating_output` (`strom/optimization_utils.py`, line 96).

**Euler stability:** $|1 + \Delta t \cdot \lambda| < 1$ requires $\Delta t < 13\ \text{min}$ for the fast eigenmode.
- *Real-time mode:* tolerates $\Delta t = 1\text{h}$ because only `HeaterOutput[0]` is ever applied; drift never accumulates.
- *Backtests:* run at $\Delta t = 5\ \text{min}$ to stay inside the stability bound, since there is no live feedback to correct accumulated error.

---

## 6. Tools & Tech Stack

| Tool | Role |
|---|---|
| **Python + CVXPY** | LP formulation and solving; chunked monthly solve for multi-year backtests |
| **pandas / numpy** | Time-series manipulation |
| **ENTSO-E API** | Day-ahead electricity price forecasts |
| **OpenWeatherMap API** | Exterior temperature forecasts |
| **python-kasa** | Async control of the TP-Link smart plug |

---

## 7. Results

All comparisons are against a **deadband thermostat**: heat when $T_{air} < T_{min}$, cool when $T_{air} > T_{max}$, off otherwise. No price awareness, no look-ahead. This is the realistic baseline — what a household would already have without the optimizer.

### One Week — 23–29 November 2024

![Weekly comparison, 23–29 Nov 2024](_static/images/compare_costs_temps_Barcelona_23-29_Nov.png)
*Optimal vs. thermostat policy across one week, 23–29 November 2024.*

Around 24 November, electricity prices drop to nearly zero. The optimizer sees this in the day-ahead forecast and fully pre-charges the heat battery — pushing $T_{air}$ to $T_{max}$ — then coasts for the following days without heating as prices return to normal. The thermostat cannot do this: it has no look-ahead, so it heats reactively and pays peak prices.

- **48% cost savings** vs. thermostat baseline — a fair comparison since November exterior temperatures stay well below $T_{min}$, so both strategies are actively heating throughout.
- The optimizer buys *more* total energy than the thermostat: **98 kWh vs 78 kWh** — savings come from *when* energy is purchased, not from consuming less.
- Interior range [18.0, 21.7] °C; max swing 3.7 °C; zero comfort violations.
- During the pre-charge on 24–25 Nov, $T_{air}$ settles ~1 °C above $T_{wall}$, matching the §4.3 prediction $\Delta T \approx 0.84\ ^\circ\text{C}$.

### November 2024

![Monthly comparison, November 2024](_static/images/compare_costs_temps_Barcelona_Nov.png)
*Optimal vs. thermostat policy across November 2024.*

The week result is not a cherry-pick — the load-shifting pattern repeats consistently across the month wherever the price schedule creates cheap windows.

- **29% cost savings** over the full month; optimizer again buys more total energy (187 kWh vs 165 kWh).
- Interior range [18.0, 21.5] °C; zero comfort violations.

### Two-Year Backtest — March 2023 to March 2025

The two-year window is solved as a *chunked monthly LP*: each calendar month is an independent LP; the final state $(T_{air}, T_{wall})$ of month $N$ seeds month $N+1$. Each chunk is persisted to disk for resumability. This decomposition is valid because $\tau_{wall} \approx 7\ \text{days}$ — roughly four wall time-constants fit inside a 30-day chunk, giving meaningful scheduling slack. The one honest trade-off: each monthly LP has no incentive to keep the heat battery charged at the end of the month, so there is a small seam suboptimality at month boundaries.

![Two-year comparison, Mar 2023 – Mar 2025](_static/images/compare_costs_temps_Barcelona_Mar23_Mar25.png)
*Optimal vs. thermostat policy over two years, heating and cooling included.*

<!-- TODO: update table and figures below once backtest with thermostat baseline completes -->

| | Thermostat | Optimal | Savings |
|---|---|---|---|
| **Total cost** | — € | — € | **~X%** |
| **Heating energy** | — kWh | — kWh | — |
| **Cooling energy** | — kWh | — kWh | — |
| **Comfort violations** | — | 0 | — |

**Seasonal pattern:** Spring and autumn months dominate savings — exterior temperature sits near or inside the comfort band, creating large scheduling slack. December and January hover around 10–15% — the heater runs nearly full-time compensating heat loss, leaving little room for cheap-hour shifting.

**Swing amplitude:** Mean ~1 °C, P95 ~2 °C. The tail matters for user experience — individual large excursions erode trust faster than the mean suggests. A rate constraint on $\Delta T$ per timestep is the natural next step.

**Comfort:** $T_{min}$ and $T_{max}$ never breached across all 25 months.

---

## 8. Conclusion

This project demonstrates full-stack applied mathematics in practice: from a homeowner's informal request to a working system through physics modeling, optimization, and IoT control. At each step the coupling runs both directions — the physics is only as complex as the optimization needs, the optimization only as fast as the model permits. No layer is over- or under-engineered.

The result is a globally optimal, interpretable schedule with mathematically guaranteed comfort. A human operator can look at the 24-hour plan and understand *why* the optimizer pre-heats at 02:00 and coasts through 09:00. This auditability — absent from black-box ML approaches — matters for user trust and regulatory compliance.

---

## 9. Future Work

- Coordinator-level optimization across multiple devices or households
- Stochastic optimization under price and weather uncertainty
- V2G energy management (see separate study)
- Temperature rate constraint to cap large swing events before wider deployment
- Passive thermal control: roller shutters (stays LP); opening windows makes $R_{ext}$ a decision variable → bilinear → SDP relaxations
- **June 2024 isolated study:** optimizer cost 0.00€ vs. non-zero thermostat; clean illustration of the cooling-avoidance mechanism without the noise of a full two-year plot

---

## 10. References

[^pean2018]: Péan, T., Salom, J., & Costa-Castelló, R. (2018). Configurations of model predictive control to exploit energy flexibility in building thermal loads. In *Proc. 57th IEEE Conference on Decision and Control (CDC)*, Miami, FL, USA, pp. 3177–3182. DOI: 10.1109/CDC.2018.8619095. [Local copy](papers/pean2018.pdf)
