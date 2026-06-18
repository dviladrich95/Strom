# Case Study: Smart Heating Optimization

<!-- TODO (second pass): unify narrator voice — smooth and sober, between the casual flourishes and the formal math notation. Currently oscillates. -->
<!-- TODO (second pass): expand citations beyond [^pean2018] — MPC-in-buildings literature (e.g. Oldewurtel et al.) and RC thermal modeling references. -->

## Abstract

**27% lower electricity cost** over two years. On the days the house charges itself with cheap heat, it naturally runs warmer too. It runs live on a €15 smart plug and takes a few seconds to solve.

The system uses a physics model of the building and a linear program to decide when to heat and when to wait, using day-ahead forecasts of electricity prices and outdoor temperature. The optimization method guarantees that the temperature stays within the comfort band, and it is this reliability that allows new energy companies to scale the same approach to thousands of homes across European grids.

![Two-year comparison, Mar 2023 – Mar 2025](_static/images/compare_costs_temps_Barcelona_Mar23_Mar25.png)
*Optimal vs. thermostat, March 2023 – March 2025.*

---

## 0. The Story

[Jan Balanya](https://janbalanya.com/) is on a dynamic electricity tariff: prices change every hour, cheapest at night, expensive in the morning. He had a plug-in electric heater at home, which makes heating even more expensive, and he wanted to take advantage of the price swings automatically, without checking an app.

The result is a full-stack project: it pulls price and weather forecasts from APIs, builds a physical model of the building's thermal behaviour, solves an optimization problem to find the cheapest heating schedule, and sends commands to a smart plug via a custom interface built to control the plug-in heater remotely.

---

## 1. The Business Perspective

Renewable energy is cheap but volatile, depending on weather conditions. Solar overproduction drives prices to near zero some afternoons, then demand peaks push them back up. People on dynamic tariffs see this directly in their bills.

The opportunity is load shifting: buy energy when it's cheap, not when you need it. Heating is a good target because thermal mass gives you slack: a few degrees and a few hours of flexibility before comfort suffers. The optimizer exploits that slack systematically.

The objective is simple: minimize the electricity bill while keeping the temperature inside a defined comfort band at all times. With a linear model of the building, this becomes a linear program that can be solved efficiently and reliably, guaranteed to be globally optimal and interpretable.

---

## 2. How the Building Works as a Battery

Electricity is perishable: the grid has to match supply and demand in real time. The house's thermal mass lets us store that cheap electricity by heating it up, then give off that heat over the next hours, spreading out its use.

The physics of each part follows Newton's law of cooling:

$$C \frac{dT_{\text{interior}}}{dt} = \frac{T_{\text{exterior}} - T_{\text{interior}}}{R} + Q$$

| Symbol | Meaning | Units |
|---|---|---|
| $C$ | Heat capacity: energy needed to raise temperature by 1°C | kWh/°C |
| $R$ | Thermal resistance: how hard heat flows between two regions | °C/kW |
| $T$ | Temperature (interior, exterior, etc.) | °C |
| $\tau = R \cdot C$ | Timescale: time to halve a temperature gap | h |
| $Q$ | Net thermal power injected into the interior | kW |

The injected power $Q$ is the *net* thermal flow: positive when heating, negative when cooling. In practice it is the difference of two non-negative duty cycles,

$$Q = Q_h\,\alpha_h(t) - Q_c\,\alpha_c(t),$$

where $Q_h, Q_c$ are the heater and cooler nominal powers and $\alpha_h, \alpha_c \in [0,1]$ are their PWM duty cycles. Splitting heating and cooling into two independent variables keeps the optimization convex; a single signed $Q$ would force the optimizer to choose its sign, breaking linearity.

But homes are not a single lump of mass. They are shells that separate a thin envelope of air — which we want to hold inside a comfortable temperature range — from the exterior. The radiator heats the air, the air heats the walls, and heat eventually leaks outside.

So the model has two meaningful lumps. The air heats up and cools down in minutes; the wall's thermal mass takes days to fully charge or discharge. A thermal resistance between them measures how hard it is for heat to pass; another resistance to the outside reflects the insulation quality.

Pre-heating the wall is what actually extends the coasting time inside the comfort zone — it is the slow, large reservoir that keeps the house warm after the heater turns off.

The same logic runs in reverse for cooling: the optimizer can pre-cool during cheap night hours so the house can coast through a hot afternoon, with $\alpha_c$ replacing $\alpha_h$ in the same equations.

---

## 3. The Physical Model

### 3.0 Which Model to Use?

The two natural lumped-parameter options:

| Model | What it captures | The trade-off |
|---|---|---|
| **1R1C** | Single air+wall lump | Simple and useful for grid-scale aggregation, but misses the air/wall split — it cannot tell how warm people actually feel inside |
| **2R2C** | Air + wall | Two time constants, still linear; captures the heat-battery effect that drives both comfort and cost |

> *Beyond lumped models lie full spatial simulations (FEM-based tools like EnergyPlus) that resolve geometry and localized heat leaks. They are accurate but expensive — useful for design and certification, not for real-time control.*

2R2C is the minimal model that captures comfort, since it separates the fast-responding air from the slow thermal reservoir. It is also linear, which means the cost-minimization problem in §4 can be solved exactly using a standard linear program, with hard comfort constraints rather than soft penalties.

---

### 3.1 Two Equations

Map the building to a circuit: thermal masses are capacitors, insulation layers are resistors, the heater and cooler are current sources. Newton's law at each node gives one differential equation per part.

For the air:
$$C_{air} \frac{dT_{air}}{dt} = \frac{T_{wall} - T_{air}}{R_{int}} + Q_h\,\alpha_h(t) - Q_c\,\alpha_c(t)$$

For the wall:
$$C_{wall} \frac{dT_{wall}}{dt} = \frac{T_{air} - T_{wall}}{R_{int}} + \frac{T_{ext} - T_{wall}}{R_{ext}}$$

The air node receives the net injection from heater and cooler — both with non-negative duty cycles $\alpha_h, \alpha_c \in [0,1]$ — and exchanges with the wall. The wall exchanges with the air and slowly leaks to the outside.

### 3.2 Matrix Form

Written together:

$$\frac{d\mathbf{T}(t)}{dt} = A\,\mathbf{T}(t) + B\,\mathbf{u}(t) + \mathbf{d}(t)$$

| Symbol | What it is |
|---|---|
| $\mathbf{T}(t) = [T_{air}(t),\ T_{wall}(t)]^\top$ | The two temperatures (the system's state) |
| $\mathbf{u}(t) = [\alpha_h(t),\ \alpha_c(t)]^\top$, $\alpha_h, \alpha_c \in [0,1]$ | Heater and cooler duty cycles |
| $A$ | How temperatures drive each other; off-diagonals are heat pathways, diagonals are loss rates |
| $B$ | Generic input-gain matrix. Only the air row is excited by the current actuators (heater positively, cooler negatively); the wall row stays zero unless an interior-wall heater is added later |
| $\mathbf{d}(t)$ | Exterior temperature forcing |

$$A = \begin{pmatrix} -\dfrac{1}{R_{int} C_{air}} & \dfrac{1}{R_{int} C_{air}} \\[8pt] \dfrac{1}{R_{int} C_{wall}} & -\dfrac{\frac{1}{R_{int}} + \frac{1}{R_{ext}}}{C_{wall}} \end{pmatrix}, \qquad B = \begin{pmatrix} \dfrac{Q_h}{C_{air}} & -\dfrac{Q_c}{C_{air}} \\[8pt] 0 & 0 \end{pmatrix}$$

With the actual parameter values:

$$A \approx \begin{pmatrix} -9.16 & 9.16 \\[4pt] 0.125 & -0.131 \end{pmatrix}\ \text{hr}^{-1}$$

The diagonal entries differ by a factor of 70. That's the model's way of saying what we already knew: air responds 70 times faster than the wall.

On $\mathbf{d}(t)$: the exterior temperature term from the wall equation splits when you expand it. The part proportional to $T_{wall}$ gets absorbed into $A_{22}$, since it is already a state. The part proportional to $T_{ext}$ has nowhere else to go, since $T_{ext}$ is something we measure but do not control:

$$\mathbf{d}(t) = \begin{pmatrix} 0 \\[4pt] \dfrac{T_{ext}(t)}{R_{ext}\, C_{wall}} \end{pmatrix}$$

### 3.3 Parameters and Timescales

Parameters are taken from Péan et al. (2018)[^pean2018], who identified them for a multi-family apartment in Barcelona. We adopt them as a reasonable starting point for an apartment of similar build; full system identification on Jan's specific apartment is left as future work.

| Parameter | Value | Units | What it measures |
|---|---|---|---|
| $C_{air}$ | 0.26 | kWh/°C | Energy needed to heat the air by 1°C |
| $C_{wall}$ | 19.1 | kWh/°C | Same for the wall, 73× larger |
| $R_{int}$ | 0.42 | °C/kW | Resistance between air and wall |
| $R_{ext}$ | 8.86 | °C/kW | Insulation from outside |
| $Q_h$ | 2.0 | kW | Heater nominal power |
| $Q_c$ | 2.0 | kW | Cooler nominal power |
| $T_{min}$ / $T_{max}$ | 18 / 24 | °C | Comfort band |

The two timescales:
- $\tau_{air} = R_{int} \cdot C_{air} \approx 6.5\ \text{min}$: air responds in minutes
- $\tau_{wall} = R_{ext} \cdot C_{wall} \approx 7\ \text{days}$: wall holds heat for nearly a week

At full heater power and no cooling ($\alpha_h = 1, \alpha_c = 0$), the air runs about 0.84°C warmer than the wall in steady state, a small but measurable gap that shows up in the plots:
$$\Delta T \approx R_{int} \cdot Q_h = 0.42 \times 2.0 = 0.84\ ^\circ\text{C}$$

---

## 4. The Optimization

The task: pick heater and cooler duty cycles $\alpha_h(t), \alpha_c(t) \in [0,1]$ over the next 24 hours so that the indoor temperature stays inside the comfort band and the total electricity cost is minimized. Because only the *next step* is ever applied — the LP re-solves every interval with a fresh forecast — this is a **Model Predictive Control** (MPC) loop.

**Why not a thermostat:** it heats when cold and stops when warm. No awareness of what electricity costs at 03:00 vs. 09:00, so it cannot pre-heat. All the savings in this project come from that look-ahead.

**Why not reinforcement learning:** RL can learn complex policies, but it cannot give hard guarantees. A temperature constraint entered as a soft penalty is just that — soft. One cold night is enough to lose a user's trust in an automated system.

**Why a linear program:** the physical model is linear, the cost function (power × price) is linear, and the constraints (comfort band, duty-cycle bounds) are linear. A linear program finds the global optimum exactly, with the comfort band as a hard constraint, not a suggestion.

To embed the continuous dynamics in the LP, each ODE step becomes a linear equality constraint using forward Euler with $\Delta t = 5\ \text{min}$:

$$\mathbf{T}(t+1) = \mathbf{T}(t) + \Delta t \left( A\,\mathbf{T}(t) + B\,\mathbf{u}(t) + \mathbf{d}(t) \right)$$

That step size is set by stability: forward Euler is only stable when $\Delta t < 2/|\lambda_{\text{fast}}| \approx 13\ \text{min}$ for the air's fast eigenmode. A 24-hour horizon at 5 min gives 288 timesteps, so the full LP has roughly $288 \times 2$ states + $288 \times 2$ controls $\approx 1{,}150$ decision variables, plus the dynamics, comfort, and duty-cycle constraints.

CVXPY assembles the problem and ships it to **CLARABEL**, an interior-point conic solver. In practice, the solve itself is the cheap part — milliseconds — while CVXPY's symbolic constraint construction takes a few seconds.

<!-- TODO: quantify build vs. solve times exactly. The ~1,150-variable LP is small and worth a speed test against alternative formulations (e.g. JuMP/Julia, sparse direct LP) to see whether the constraint-construction overhead is intrinsic or Python-side. -->

The continuous duty cycles $\alpha_h, \alpha_c \in [0,1]$ output by the LP are realized on the physical relay through PWM: the smart plug spends a fraction $\alpha$ of each interval ON and the rest OFF, time-averaging to the requested mean power.

One practical detail: ENTSO-E wholesale prices can go negative during oversupply, but a real household tariff never does — there are always network charges, taxes, and margins. A constant floor $P_{base}$ is added to all prices before the optimization runs.

---

## 5. Tools

The deployment splits cleanly into two pipelines: a **historical-data backtest** that produces the multi-year savings numbers, and a **live controller** that runs the same LP every interval against a real plug. The historical pipeline is where the analytical work lives; the live loop is short and conventional.

| Tool | What it does here |
|---|---|
| **Python + CVXPY** | Formulates the LP; chunked monthly solves stitch a two-year horizon |
| **CLARABEL** | Interior-point conic solver — lightweight, robust, and well-suited to LPs of this size |
| **pandas / numpy** | Time-series alignment, resampling, and per-month accounting |
| **ENTSO-E API** | Day-ahead wholesale electricity prices for the historical backtest and the live forecast |
| **OpenWeatherMap API** | Historical and forecast exterior temperatures |
| **python-kasa** | Sends ON/OFF commands to the TP-Link smart plug in the live loop |

The historical pipeline pulls multi-year price and weather series, aligns them onto a common 5-minute grid, and feeds them into the chunked monthly solve in §6. The live loop is a thin wrapper: at each interval it pulls the freshest forecast, solves the LP, and applies the first step's $\alpha_h$ via PWM on the relay.

---

## 6. Results

The comparison throughout is a **deadband thermostat**: heat when the temperature drops below $T_{min}$, cool when it rises above $T_{max}$, off otherwise. No price awareness, no forecasting. That's what a household without this system would run.

### One Week: 23–29 November 2024

![Weekly comparison, 23–29 Nov 2024](_static/images/compare_costs_temps_Barcelona_23-29_Nov.png)
*Optimal vs. thermostat, 23–29 November 2024.*

Around 24 November, electricity prices fall to nearly zero. The optimizer sees this a day ahead and runs the heater flat-out, pushing the house to the top of the comfort band. Then it coasts: no heating for two days while prices are high and the house slowly gives back the stored heat. The thermostat can't do any of this. It just reacts.

The result: **48% cost savings** over the week, while the optimizer actually bought *more* energy: 98 kWh vs 78 kWh. Lower cost from buying more. That's the whole trick: timing matters more than quantity.

During the pre-charge on 24–25 Nov, the air temperature ran about 1°C above the wall, exactly what the model predicts ($\Delta T \approx 0.84\ ^\circ\text{C}$, §3.3). The theory checks out in the data.

### November 2024 (second half)

![Monthly comparison, November 2024](_static/images/compare_costs_temps_Barcelona_Nov.png)
*Optimal vs. thermostat, 15–30 November 2024.*

The week wasn't cherry-picked. The same pattern repeats wherever the price schedule offers a cheap window: **36% savings** over the fortnight, again buying more total energy (187 kWh vs 168 kWh). November is heating-dominated, so the optimizer hugs the lower edge of the comfort band most of the time — only pushing higher when pre-charging before a price spike. In summer the pattern reverses: with cooling as the dominant cost, the optimizer hugs the upper edge and pre-cools toward the lower bound when night prices drop.

### Two-Year Backtest: March 2023 to March 2025

Running the full two years as a single LP isn't feasible: at 5-minute steps that's 210,000 timesteps, too large for CVXPY to handle efficiently. Instead, each calendar month is solved independently, with the final temperatures of month $N$ passed as initial conditions to month $N+1$. Each month is saved to disk so the computation can resume if anything fails.

This works because the wall's 7-day timescale fits comfortably inside a 30-day window, leaving enough room within each month for meaningful pre-heating decisions. The one trade-off: the optimizer doesn't know about next month, so it has no incentive to keep the heat battery charged at the end of a month. There's a small suboptimality at each boundary.

![Two-year comparison, Mar 2023 – Mar 2025](_static/images/compare_costs_temps_Barcelona_Mar23_Mar25.png)
*Optimal vs. thermostat, March 2023 – March 2025.*

| | Thermostat | Optimal | Savings |
|---|---|---|---|
| **Total cost** | 561.71 € | 409.47 € | **27%** |
| **Heating energy** | 5,699 kWh | 6,237 kWh | — |
| **Cooling energy** | 735 kWh | 869 kWh | — |

The optimizer spends less while using more energy. Savings come entirely from timing.

The blended 27% number masks three regimes:

- **Free-coast months** (March 2023, October 2023, June 2024) cost essentially zero for both systems — the exterior temperature stays inside the comfort band, so no actuation is required. These months confirm the optimizer correctly does nothing when nothing is needed; they are sanity checks, not wins.
- **Shoulder seasons** (April, May, July, August, late autumn, March 2025) are where active control matters most: wide price swings combined with ample comfort-band slack let the optimizer save **35–70%** on bills of 5–35 €.
- **Deep winter** (December, January, February) saturates the heater — it runs nearly full-time just to compensate heat loss, leaving little slack for cheap-hour shifting. Savings here drop to **14–22%**, but these months also dominate the absolute bill. That weighting is why the blended figure (27%) sits closer to deep-winter savings than to shoulder savings.

There is a side effect worth naming. Buying when prices are at or near zero means buying surplus — the periods of solar overproduction or low overnight demand that the grid would otherwise have to spill or curtail. At household scale this is a footnote; at fleet scale, optimizers like this one act as distributed flexibility, absorbing renewable overproduction the grid cannot use directly.

**Temperature smoothness.** Per-step changes stay below 0.7°C / 5 min. Peak-to-peak swing amplitudes within the comfort band average ~1°C, with a 95th percentile of ~2°C; the maximum observed amplitude reaches the full 6°C band on a handful of days when the optimizer fully exploits both bounds — typically by riding from the lower edge to the upper edge during a deep pre-charge.

<!-- TODO: identify the specific day(s) of the 6°C peak-to-peak swing and characterize the conditions (price profile, weather, time of year). Consider a rate constraint on |dT_air/dt| since 0.7°C / 5 min ≈ 8°C/hr could feel uncomfortable to occupants. -->

---

## 7. Conclusion

The path from "automate my heating" to a working system runs through four layers: a physical model of the building, a linear program that exploits it, an API pipeline that feeds it forecasts, and a Model Predictive Control loop that executes the output. Each layer is as simple as the next one allows. The model is linear, which is what makes the LP fast; the LP is fast, which is what makes the MPC loop tractable; and a tractable MPC loop is what turns a forecast into a plug command every five minutes.

That same linearity is what makes the result interpretable. The schedule the optimizer produces can be read line by line: it pre-heats at 02:00 because electricity is cheapest then, and coasts through 09:00 because the heat stored in the wall is enough to stay comfortable. A black-box model gives you a number. This gives you a reason.

---

## 8. Future Work

- Coordinating multiple devices or households, using the same LP structure with a larger state space
- Handling price and weather uncertainty explicitly, instead of treating day-ahead forecasts as ground truth
- V2G energy management <!-- TODO: link to the V2G case study once published -->
- Identifying the apartment-specific RC parameters from real measurements rather than literature defaults
- A rate constraint on $|dT_{air}/dt|$ to cap rare large swings; current per-step changes can reach ~0.7°C / 5 min, which is fast enough to feel
- Roller shutters as a new control input: they modulate solar gain linearly, so the LP stays intact. Opening windows is more interesting: it makes $R_{ext}$ a decision variable, which introduces a bilinear term and breaks convexity — territory of SDP relaxations

---

## 9. References

[^pean2018]: Péan, T., Salom, J., & Costa-Castelló, R. (2018). Configurations of model predictive control to exploit energy flexibility in building thermal loads. In *Proc. 57th IEEE Conference on Decision and Control (CDC)*, Miami, FL, USA, pp. 3177–3182. DOI: 10.1109/CDC.2018.8619095.
