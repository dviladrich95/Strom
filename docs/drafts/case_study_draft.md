# Case Study: Smart Heating Optimization

<!----------------------------------------------------------------------------->
<!-- SKETCH — structural overview only; iterate here, do not develop content -->
<!----------------------------------------------------------------------------->

## Sketch

### Abstract
- **~X% savings** over 2 years vs. thermostat — TODO once backtest completes
- Physics model + LP; live on smart plug; zero comfort violations

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
<!-- SKELETON — structural scaffold; develop prose in case_study_textwall.md -->
<!----------------------------------------------------------------------------->

---
---

## Abstract
<!-- HR scan: first thing visible, no scrolling. 3 lines max. -->
- **~X% savings** over 2 years vs. thermostat — TODO: fill once backtest completes
- Physics model + LP solved in ms; runs live on €15 smart plug
- Zero comfort violations

---

## 0. The Story
<!-- 2 paragraphs max. Hook, not explanation. Comes after abstract. -->
- Jan's electric stove + dynamic tariffs → weekend project → full-stack applied math
- Household proof-of-concept for industrial demand response (same principles, larger scale)

---

## 1. The Business Problem
<!-- Context → opportunity → objective → price realism → value prop -->
- Spain duck curve + dynamic tariffs expose price swings to consumers
- Load shifting = timing, not reducing consumption; heating/cooling = large loads with comfort slack
- LP: ms solve, global opt, interpretable schedule ("pre-heat 02:00, coast 09:00")
- $P_{base}$ floor bridges ENTSO-E wholesale to retail prices

---

## 2. Intuitive Explanation of the Dynamics
<!-- Accessible: no equations, just the chamber/pipe analogy + Newton cooling intro -->
- Electricity perishable → thermal mass = cheapest storage
- Two chambers + pipe: air (thin/fast) + wall (wide/slow) + leaky pipe to outside
- $C$ = chamber volume, $R$ = pipe width; Newton cooling: one equation per lump
- Cooling = same model, $Q_{cool}$ enters with negative sign

---

## 3. Modeling Alternatives & Model Selection
<!-- Table + one-paragraph justification -->
- Table: 1R1C (misses wall) / **2R2C chosen** / FEM (breaks linearity)
- 2R2C: minimal model capturing both time constants; 1R1C loses heat-battery; FEM overkill

---

## 4. The Physical Model
<!-- Break into 3 subsections to avoid wall -->

### 4.1 The ODE System
- RC circuit analogy: masses→capacitors, insulation→resistors, heater→current source
- Air–wall ODE; wall–exterior ODE (two coupled equations)

### 4.2 State-Space Form
- $\dot{\mathbf{T}} = A\mathbf{T} + B\mathbf{u}_t + \mathbf{d}_t$; variable table; numerical $A$ (factor ~70 between diagonals)
- $\mathbf{d}_t$ derivation: $T_{ext}$ term splits into state part (→ $A_{22}$) + forcing (→ $\mathbf{d}_t$)

### 4.3 Parameters and Timescales
- Parameter table (Péan 2018, Barcelona apartment, PRBS-identified)
- $\tau_{air} \approx 6.5\ \text{min}$; $\tau_{wall} \approx 7\ \text{days}$
- Air-wall gap $\Delta T \approx 0.84\ ^\circ\text{C}$ at full heater output — verifiable in week plot

---

## 5. Optimization Method & Alternatives
<!-- Table + LP motivation + discretization + stability note -->
- Table: thermostat (myopic) / RL (no hard guarantees → single cold night = lost user) / **LP chosen**
- Linear model + linear cost → LP; CVXPY → CLARABEL; global optimum; hard comfort constraints
- Euler $\Delta t=1\text{h}$ ok in real-time (only first command applied); backtests use $\Delta t=5\text{min}$

---

## 6. Tools & Tech Stack
<!-- Single merged table -->
- Python + CVXPY; pandas/numpy; ENTSO-E API; OpenWeatherMap API; python-kasa

---

## 7. Results
<!-- Three windows with distinct jobs: story / bridge / anchor -->

### One Week — 23–29 November 2024
<!-- Story: show how optimizer thinks. Image leads. -->
- **[image: compare_costs_temps_Barcelona_23-29_Nov.png]**
- Pre-charge on 24 Nov (near-zero price) → coast for days; thermostat can't look ahead
- Surprise pair: **48% savings + bought MORE energy** (98 vs 78 kWh) = timing, not efficiency
- Fair comparison: Nov exterior temps stay below $T_{min}$ throughout
- $\Delta T \approx 0.84\ ^\circ\text{C}$ during pre-charge matches §4.3 prediction

### November 2024
<!-- Bridge: consistency across the month. One paragraph + image. -->
- **[image: compare_costs_temps_Barcelona_Nov.png]**
- 29% savings; load-shifting pattern repeats wherever price schedule creates cheap windows

### Two-Year Backtest — March 2023 to March 2025
<!-- Anchor: quantitative bottom line. Chunking → table → image → seasonal. -->
- **[image: compare_costs_temps_Barcelona_Mar23_Mar25.png]**
- Chunked monthly LP: seam suboptimality at boundaries; ~4 $\tau_{wall}$ per chunk → valid
- **TODO: fill table once backtest completes**
- Seasonal: spring/autumn dominate savings; Dec/Jan 10–15% (heater fully occupied)
- Swing: mean ~1°C, P95 ~2°C, max 6°C → rate constraint needed before wider deployment
- Zero comfort violations across 25 months

---

## 8. Conclusion
<!-- 2 paragraphs only: (1) what physics + opt achieve; (2) problem-solving approach -->
- 2R2C: minimal, validated, stays linear; two time constants = the heat-battery mechanism
- LP: global optimum, hard constraints, interpretable schedule; auditable unlike black-box ML
- Translation chain: problem → physics → math → convex form → code; coupling runs both ways

---

## 9. Future Work
- Multi-device coordinator; stochastic opt; V2G
- Temperature rate constraint to cap 6°C tail events
- Passive control: shutters (stays LP) → windows ($R_{ext}$ as variable → bilinear → SDP)
- June 2024 isolated study: optimizer 0€ vs. thermostat — clean cooling-avoidance illustration

---

## 10. References

[^pean2018]: Péan, T., Salom, J., & Costa-Castelló, R. (2018). Configurations of model predictive control to exploit energy flexibility in building thermal loads. In *Proc. 57th IEEE Conference on Decision and Control (CDC)*, Miami, FL, USA, pp. 3177–3182. DOI: 10.1109/CDC.2018.8619095. [Local copy](../papers/pean2018.pdf)
