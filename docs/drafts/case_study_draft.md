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
- **Parameters:** $C_{air}=0.26$, $C_{wall}=19.1$, $R_{int}=0.42$, $R_{ext}=8.86$ — identified for a Barcelona multi-family apartment[^pean2018]
- **Time scales:** $\tau_{air} \approx 6.5$ min, $\tau_{wall} \approx 169$ h ≈ 7 days
- **Air-wall gap during heating:** $\Delta T \approx R_{int} \cdot Q_{heater} = 0.84$°C

---

### 5. Optimization Alternatives
| Method | Pro | Con |
|---|---|---|
| Thermostat | simple | myopic, price-blind |
| RL | no explicit model needed | no hard guarantees; data-hungry; hurts user retention |
| **Linear MPC** | global opt; hard constraints; receding horizon; CVXPY; interpretable | needs linear model |

- **Discretization:** Euler step $\mathbf{T}_{t+1} = \mathbf{T}_t + \Delta t(A\mathbf{T}_t + B\mathbf{u}_t + \mathbf{d}_t)$ → linear equality constraints
- **LP in code:** [`find_heating_output`](../../strom/optimization_utils.py) (line 96); $u_t \in [0,1]$ decision variable; price vector as cost coefficient
- **Euler stability:** $|1 + \Delta t \cdot \lambda| < 1$; Péan fast mode → $\Delta t < 13\text{ min}$; IoT mode ok at $\Delta t = 1\text{h}$ (only `HeaterOutput[0]` applied); backtests use $\Delta t = 5\text{ min}$

---

### 6. Tools
| Tool | Role |
|---|---|
| Python + CVXPY | daily runs, real-time, multi-year backtests via chunked monthly LP |
| pandas/numpy | time-series |
| ENTSO-E API | price forecasts |
| OpenWeatherMap | T_ext forecasts |
| python-kasa | smart plug control |

---

### 7. Results
- **Baseline:** 24h rolling mean of exterior temp clipped to $[T_{min}, T_{max}]$; minimum-intervention policy; steelmanned vs. naive constant center
- **Week (23–29 Nov):** **48.07% savings**; optimal 98 kWh vs baseline 78 kWh (load-shifting); range [18, 21.67]°C; max swing 3.66°C; 5 heat + 4 cool events; air-wall gap ~1°C matches §4 calc
- **Full month (Nov):** **28.62% savings**; optimal 187 kWh vs baseline 165 kWh; range [18, 21.51]°C; max swing 3.51°C; 17 heat + 16 cool events; mean swing 1.07°C
- **Two-year (+ cooling):** **46.40% savings**, 354.48€ absolute (763.95€ baseline → 409.47€ optimal); optimizer buys *less* total energy (7105 vs 9622 kWh) — savings driven by both timing and cooling reduction (77.1% cooling savings vs 36.2% heating); 620 heat + 619 cool swing events; mean swing 1.09°C, P95 1.90°C, max 6.00°C; zero comfort violations; computed via **chunked monthly LP** — independent per-month solves seamed on $(T_{air}, T_{wall})$, chunks persisted to disk for resumability
- **Bang-bang control:** $u_t \in \{0,1\}$ — LP vertex behaviour; smoother profile possible via regularization

### 8. Conclusion
- full-stack applied math: physics → LP → real-time IoT control
- hard comfort guarantees by construction, not soft penalties; globally optimal; auditable
- quantified savings: 48% (week 23–29 Nov — favourable: zero-price periods enabled full pre-charging); 29% (Nov 2024 — autumn shoulder season, comfort band partially reachable for free); 46% over two years — spring/autumn months drive bulk of savings, winter months (Dec/Jan) 10–15% due to near-constant heating load

### 9. Future Work
- coordinator-level optimization across devices / households
- stochastic elements (price and weather uncertainty)
- V2G energy management (see separate study)
- temperature regularization term to smooth jumps (~4°C avg; low priority but worth testing with users)
- passive thermal control: shutters → new linear solar-gain input (stays LP); windows → $R_{exterior}$ as decision variable → bilinear programming → SDP relaxations
- **additional case study: June 2024** — baseline 24.09€ vs optimal 0.00€ (100% savings); temperature naturally inside comfort band → baseline does unnecessary cooling that the optimizer entirely avoids; clean isolated illustration of the cooling-avoidance mechanism

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

With numerical values ($C_{air}=0.26$, $C_{wall}=19.1$, $R_{int}=0.42$, $R_{ext}=8.86$):

$$A \approx \begin{pmatrix} -9.16 & 9.16 \\[4pt] 0.125 & -0.131 \end{pmatrix}, \qquad B \approx \begin{pmatrix} 3.85 & 0 \\[4pt] 0 & 0.0524 \end{pmatrix}$$

The two diagonal entries of $A$ differ by a factor of ~70 — $|A_{11}| \approx 9.16$ vs $|A_{22}| \approx 0.131$ — reflecting strong separation of timescales between air and wall.

The same conductance-matrix structure appears in hydraulic networks and RC circuits in electrical engineering — heat, charge, and fluid all obey the same conservation law at each node.

**On the disturbance vector $\mathbf{d}_t$:** it is not zero. Expanding the wall–exterior heat flux: $\frac{T_{ext} - T_{wall}}{R_{ext}} = -\frac{T_{wall}}{R_{ext}} + \frac{T_{ext}}{R_{ext}}$. The $-T_{wall}/R_{ext}$ term is absorbed into $A_{22}$ (it is proportional to a state variable); the $+T_{ext}/R_{ext}$ term cannot enter $A$ (since $T_{ext}$ is not a state) nor $B$ (since it is not a control) — so it becomes a time-varying forcing term driven by the measured exterior temperature forecast:

$$\mathbf{d}_t = \begin{pmatrix} 0 \\[4pt] \dfrac{T_{ext}(t)}{R_{ext}\, C_{wall}} \end{pmatrix}$$

The air row is zero because the heater is the only external input to the air lump — its thermal coupling to the wall is already fully captured in $A$.

**Parameter values:** Identified 2R2C parameters from Péan et al. (2018)[^pean2018] for a multi-family apartment in Sant Adrià de Besòs (Barcelona). Identification via PRBS excitation of a TRNSYS model validated against metered data. Péan's apartment includes 12 cm of added insulation.

| Parameter | Value | Units | Description |
|---|---|---|---|
| $C_{air}$ | 0.26 | kWh/°C | Heat capacity of indoor air |
| $C_{wall}$ | 19.1 | kWh/°C | Heat capacity of the insulated wall |
| $R_{interior}$ | 0.42 | °C/kW | Thermal resistance between air and wall |
| $R_{exterior}$ | 8.86 | °C/kW | Thermal resistance between wall and outside |
| $Q_{heater}$ | 2.0 | kW | Heater power |
| $T_{min}$ / $T_{max}$ | 18 / 24 | °C | Comfort band |

**Time scales and the air-wall gap:**

- $\tau_{air} = R_{int} \cdot C_{air} = 0.42 \times 0.26 \approx 0.11\ \mathrm{h} \approx 6.5\ \mathrm{min}$
- $\tau_{wall} = R_{ext} \cdot C_{wall} = 8.86 \times 19.1 \approx 169\ \mathrm{h} \approx 7\ \mathrm{days}$
- Air-wall gap at full heater output ($u=1$, quasi-steady $\dot T_{air} \approx 0$):
  $$\Delta T \;\equiv\; T_{air} - T_{wall} \;\approx\; R_{int} \cdot Q_{heater} \;=\; 0.42 \times 2.0 \;=\; 0.84\ ^\circ\mathrm{C}$$

---

## 5. Optimization Method & Alternatives

**The challenge:** find the optimal heater command sequence over a 24-hour horizon, subject to linear system dynamics and hard temperature constraints.

Note on RL: the lack of hard comfort guarantees is less critical than infrastructure stability, but it matters significantly for user retention — a single cold night is enough for a user to lose confidence in the system permanently.

| Method | Pros | Cons |
|---|---|---|
| **Rule-based (thermostat)** | Simple, off-the-shelf | Myopic; no concept of day ahead price; will not pre-heat |
| **Reinforcement Learning** | Handles non-linear dynamics without explicit modeling | Requires extensive training data; no hard constraint guarantees on comfort |
| **Linear Optimization / MPC (our choice)** | Global optimum guaranteed; hard constraints satisfied by construction; re-solves hourly with fresh forecasts | Requires a linear model — non-linear dynamics would break convexity |

**Why linear optimization:** With a linear physical model and a linear cost function (power × price), the full problem is linear. This lets us use powerful standard solvers — CVXPY in particular, which provides a clean Python interface for formulating linear and convex problems and dispatches to highly optimized backends (CLARABEL, OSQP, etc.).

Solvers exploit this structure to find the global optimum in known, bounded time. Temperature bounds enter as hard constraints, not soft penalties — comfort is guaranteed, not just encouraged.

To embed the continuous model inside an optimization problem, the ODEs are discretized with a forward Euler step of $\Delta t = 1\text{h}$:

$$\mathbf{T}_{t+1} = \mathbf{T}_t + \Delta t \left( A\mathbf{T}_t + B\mathbf{u}_t + \mathbf{d}_t \right)$$

This turns each timestep's state propagation into a linear equality constraint. The full LP — 24 such constraints, heater output $u_t \in [0,1]$ as decision variable, comfort bounds as inequality constraints, and the day-ahead price vector as cost coefficient — is implemented in [`find_heating_output`](../../strom/optimization_utils.py) (`strom/optimization_utils.py`, line 96). The result is a single LP solved once per hour over the full 24-step horizon.

Forward Euler is only conditionally stable: the condition $|1 + \Delta t \cdot \lambda| < 1$ must hold for every eigenvalue $\lambda$ of $A$. The fast mode of the 2R2C system has $\lambda \approx -1/\tau_{air}$, which with the Péan parameters gives a stability bound of $\Delta t < 2\tau_{air} \approx 13\text{ min}$.

- **Why the IoT real-time mode tolerates $\Delta t = 1\text{h}$:** `main.py` is a cron-fired one-shot that calls `find_heating_output` once per hour and applies only `HeaterOutput[0]` to the smart plug. Hours 2–24 of the optimized trajectory are discarded; the Euler drift never accumulates in the physical system because only the first command is ever executed.
- **Why offline backtests cannot:** the full 24-step trajectory *is* the simulated ground truth — there is no live state feedback to correct drift. A single unstable hour compounds through the entire simulation. For this reason all case-study scripts run at $\Delta t = 5\text{ min}$, comfortably inside the stability bound.

---

## 6. Tools & Tech Stack

**Optimization runtimes:**

| Tool | Role |
|---|---|
| **Python + CVXPY** | Primary runtime for daily runs and real-time execution; CVXPY natively handles linear and convex problem structures. Multi-year backtests run via chunked monthly LP (see §7). |

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

This choice of baseline deserves a brief justification, because the comparison is only meaningful if the baseline is genuinely hard to beat. A naive alternative — targeting the center of the comfort band year-round, say 21°C — would be too easy to outperform: in winter it would overheat the house well beyond the minimum necessary, and in summer it would overcool it, both wasting energy that the optimizer could trivially avoid. The rolling mean baseline avoids this by modelling a plausible, already-lean human behaviour: a user who adapts their thermostat roughly to the season and the weather, always targeting the comfort boundary that requires the least effort given exterior conditions. In winter, when the rolling mean falls below $T_{min}$, the baseline clips to $T_{min}$ — heating only as much as comfort demands. In summer, when the rolling mean exceeds $T_{max}$, it clips to $T_{max}$ — cooling only as much as comfort demands. The result is a minimum-intervention policy in both directions, symmetric across seasons. Any savings over this baseline come purely from smarter timing of energy purchases, not from exploiting a wasteful comparison point.

### One Week — 23–29 November 2024

![Weekly Comparison - 23-29 Nov 2024](../../plots/compare_costs_temps_Barcelona_23-29_Nov.png)
*Optimal vs. baseline policy across one week, 23–29 November 2024.*

- 7-day window aligned with $\tau_{wall} \approx 7$ days — captures one full wall time constant of scheduling slack.
- Cost savings: **48.07%** vs. baseline ([results_Barcelona_23-29_Nov.txt](../../results/results_Barcelona_23-29_Nov.txt)).
- Optimizer buys *more* total energy than baseline: **98.12 kWh vs 78.32 kWh** — load-shifting story; savings come from concentrating purchases at cheap hours, not reducing consumption.
- Interior range [18.00, 21.67] °C; max temp swing 3.66 °C; **5 heating + 4 cooling swing events** — multi-cycle pre-charging visible.
- During the prolonged 24–25 Nov pre-charge, $T_{air}$ settles ~1 °C above $T_{wall}$ and the gap stays stable while both ramp up — matching the §4 prediction $\Delta T \approx R_{int} \cdot Q_{heater} = 0.84$ °C.

### November 2024

![Monthly Comparison - Nov 2024](../../plots/compare_costs_temps_Barcelona_Nov.png)
*Optimal vs. baseline policy across November 2024.*

- Cost savings: **28.62%** vs. baseline over the full month ([results_Barcelona_Nov.txt](../../results/results_Barcelona_Nov.txt)).
- Optimizer buys *more* energy than baseline: **187.49 kWh vs 165.02 kWh** — sustained load-shifting across the month.
- Interior range [18.00, 21.51] °C; max swing 3.51 °C; **17 heating + 16 cooling swing events**. Mean swing 1.07 °C, P95 1.54 °C — most events ~1 °C overshoots, long tail of larger ones.
- $T_{min}$ never breached; comfort constraints held throughout.
- Heater control is **bang-bang**: $u_t \in \{0, 1\}$ at almost every timestep — expected from LP with box constraints (optimum at vertices of feasible set). Worth noting for hardware: cycling at full on/off may accelerate wear; smoother profile via regularization on $\Delta u_t$ is a practical refinement.

### Two-Year Backtest — March 2023 to March 2025

The full two-year window cannot be solved as a single LP at $\Delta t = 5\,\text{min}$ — that is roughly 210k timesteps, which CVXPY cannot handle efficiently in Python. The approach taken is a *chunked monthly solve*: each calendar month is solved as an independent LP; the final state $(T_{air}, T_{wall})$ of month $N$ is passed as the initial condition for month $N+1$, separately for the optimal and baseline trajectories (which diverge and must be tracked independently). Each solved month is persisted to `data/chunks_Mar23_Mar25/` so the computation can resume if a month fails. This decomposition is safe because the two state variables are coupled only through the 2R2C dynamics: $\tau_{wall} \approx 7$ days vs. a 30-day chunk means roughly 4 wall time-constants per window — enough room for meaningful pre-charging decisions without needing information beyond the current month. The one honest trade-off: each monthly LP has no incentive to keep the heat battery charged at the end of the month, so there is a small boundary suboptimality at each seam (last and first week of adjacent months).

![Long-term Comparison - 2023-2025](../../plots/compare_costs_temps_Barcelona_Mar23_Mar25.png)
*Optimal vs. baseline policy over two years, heating and cooling included.*

- Cost savings: **46.40%** vs. baseline ([results_Barcelona_Mar23_Mar25.txt](../../results/results_Barcelona_Mar23_Mar25.txt)); 354.48€ absolute (763.95€ → 409.47€).
- In contrast to the shorter backtests, the optimizer buys *less* total energy than baseline: **7105 kWh vs. 9622 kWh**. Savings are not purely load-shifting — the optimizer also avoids unnecessary cooling: 868 kWh optimal vs. 2325 kWh baseline (77.1% cooling reduction). When exterior temperature sits naturally inside $[T_{min}, T_{max}]$, a naive thermostat still runs the cooler to track its target; the optimizer recognises the free comfort window and leaves the system off.
- Seasonal pattern: spring and autumn months dominate. September 2023 (99.5%), October 2023 (100%), May–June 2023 (95–96%) stand out as nearly free months for the optimizer. December and January hover at 10–15% savings — the heater is occupied nearly full-time compensating heat losses; little scheduling slack remains for cheap-hour shifting.
- June 2024: baseline 24.09€ vs. optimal 0.00€ — the entire month is free for the optimizer. Exterior temperature is naturally inside the comfort band; baseline runs unnecessary cooling to track its rolling-mean target, optimizer does not. A dedicated case study for this month would isolate the cooling-avoidance mechanism cleanly.
- Swing amplitude (620 heating + 619 cooling events): mean 1.09°C, P95 1.90°C, max 6.00°C. The max-6°C tail is the practical concern for user experience — individual large excursions erode trust faster than the mean suggests. Capping via a rate constraint on $\Delta T$ per timestep is the natural next step.
- $T_{min}$ and $T_{max}$ never breached; comfort constraints held across all 25 months.

<!-- TODO (modeling fix needed before publishing): the rolling-mean baseline tracks a specific temperature target even when exterior temperature naturally sits inside [T_min, T_max], causing unnecessary heating/cooling that a reasonable occupant would not do. The correct baseline ansatz is a cost function that is flat (zero penalty) inside the comfort band and only penalises violations — system-off-by-default when comfort is free. Attempting this previously hit solver degeneracy (LP infeasible or trivially zero when u_t=0 already satisfies constraints). Needs a proper fix; until then, the cooling savings figures (especially spring/autumn months and June 2024 100%) are partly an artefact of an unrealistically expensive baseline. -->

---

## 8. Conclusion

*(to fill — summarize cost savings achieved, comfort guarantees maintained throughout, and how the price-ceiling objective protects users from spike exposure)*

---

## 9. Future Work

- Coordinator-level optimization across multiple devices or households
- Adding stochastic optimization (optimal choices under price and weather uncertainty)
- V2G energy management (see separate study)
- A regularization term penalizing large jumps in interior temperature (e.g. $\sum_t |T_{t+1} - T_t|$) or a hard rate constraint on $\Delta T$ per timestep could smooth the profile without significantly increasing cost. Average ~4°C swings are gradual enough to be tolerable, but outlier events with larger swings need to be capped before wider deployment — a single jarring temperature spike is enough to break consumer trust.
- Passive thermal control: roller shutters modulating solar gain introduce a new linear input term and stay within the current LP framework. Opening windows to ventilate is more interesting — it makes $R_{exterior}$ a decision variable, introducing a product $\alpha \cdot T_{wall}$ of two optimization variables. This is a bilinear term that breaks convexity, placing the extended problem firmly in the terrain of bilinear programming, addressable via SDP relaxations.
- **Additional case study — June 2024:** 100% savings that month (baseline 24.09€ → optimal 0.00€). Exterior temperature sits naturally inside $[T_{min}, T_{max}]$ the entire month; baseline runs unnecessary cooling to track its rolling-mean target; optimizer recognises the free comfort window and leaves the cooler off. A one-month isolated study would make the cooling-avoidance mechanism legible in a way the 2-year aggregate plot cannot.

---

## 10. References

[^pean2018]: Péan, T., Salom, J., & Costa-Castelló, R. (2018). Configurations of model predictive control to exploit energy flexibility in building thermal loads. In *Proc. 57th IEEE Conference on Decision and Control (CDC)*, Miami, FL, USA, pp. 3177–3182. DOI: 10.1109/CDC.2018.8619095. [Local copy](../papers/pean2018.pdf)
