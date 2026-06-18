---
toc: true
title: Strom — Smart Heating Optimisation
---

<link rel="stylesheet" href="style.css">

# Smart Heating Optimisation · Barcelona

<div class="tldr-box">
  <div class="tldr-label">tl;dr</div>
  <p class="tldr-text">
    A physics model of a Barcelona apartment, a linear programme, and a €15 smart plug. Run every hour against day-ahead electricity prices and a 48-hour weather forecast, the system finds the cheapest heating schedule that keeps the flat inside the comfort band at all times — <strong>hard constraint, not a soft penalty.</strong>
  </p>
  <ul class="tldr-findings">
    <li><strong>27% lower electricity cost</strong> over two years vs. a reactive thermostat — €152 saved while buying <em>more</em> energy.</li>
    <li><strong>Zero comfort violations</strong> across 17,568 hours and two full heating seasons. The comfort band is a hard LP constraint.</li>
    <li><strong>Shoulder seasons dominate.</strong> October and September savings reach 80–99%; deep winter drops to 14–22% because the heater runs nearly full-time regardless.</li>
    <li><strong>Buying more can cost less.</strong> During the cheap-price window of 23–29 November the optimizer bought 98 kWh vs. the thermostat's 78 kWh — and cut the bill by 48%.</li>
    <li><strong>Thermal mass is the storage medium.</strong> The wall's 7-day time constant ($\tau_{wall} \approx 7$ days) lets the optimizer pre-charge the house and coast through expensive hours with no discomfort.</li>
  </ul>
</div>

---

## 0. The Story

[Jan Balanya](https://janbalanya.com/) is on a dynamic electricity tariff: prices change every hour, cheapest at night, expensive in the morning. He had a plug-in electric heater at home, and wanted to take advantage of the price swings automatically, without checking an app.

The result is a full-stack project: it pulls price and weather forecasts from APIs, builds a physical model of the building's thermal behaviour, solves an optimization problem to find the cheapest heating schedule, and sends commands to a smart plug via a custom interface built to control the plug-in heater remotely.

---

## 1. The Business Perspective

Renewable energy is cheap but volatile, depending on weather conditions. Solar overproduction drives prices to near zero some afternoons, then demand peaks push them back up. People on dynamic tariffs see this directly in their bills.

The opportunity is load shifting: buy energy when it's cheap, not when you need it. Heating is a good target because thermal mass gives you slack: a few degrees and a few hours of flexibility before comfort suffers. The optimizer exploits that slack systematically.

The objective is simple: minimize the electricity bill while keeping the temperature inside a defined comfort band at all times. With a linear model of the building, this becomes a linear programme that can be solved efficiently and reliably, guaranteed to be globally optimal and interpretable.

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

---

## 3. The Physical Model

### 3.0 Which Model to Use?

The two natural lumped-parameter options:

| Model | What it captures | The trade-off |
|---|---|---|
| **1R1C** | Single air+wall lump | Simple and useful for grid-scale aggregation, but misses the air/wall split — it cannot tell how warm people actually feel inside |
| **2R2C** | Air + wall | Two time constants, still linear; captures the heat-battery effect that drives both comfort and cost |

> *Beyond lumped models lie full spatial simulations (FEM-based tools like EnergyPlus) that resolve geometry and localized heat leaks. They are accurate but expensive — useful for design and certification, not for real-time control.*

2R2C is the minimal model that captures comfort, since it separates the fast-responding air from the slow thermal reservoir. It is also linear, which means the cost-minimization problem in §4 can be solved exactly using a standard linear programme, with hard comfort constraints rather than soft penalties.

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
| $B$ | Input-gain matrix. Only the air row is excited by the current actuators |
| $\mathbf{d}(t)$ | Exterior temperature forcing |

$$A = \begin{pmatrix} -\dfrac{1}{R_{int} C_{air}} & \dfrac{1}{R_{int} C_{air}} \\[8pt] \dfrac{1}{R_{int} C_{wall}} & -\dfrac{\frac{1}{R_{int}} + \frac{1}{R_{ext}}}{C_{wall}} \end{pmatrix}, \qquad B = \begin{pmatrix} \dfrac{Q_h}{C_{air}} & -\dfrac{Q_c}{C_{air}} \\[8pt] 0 & 0 \end{pmatrix}$$

With the actual parameter values:

$$A \approx \begin{pmatrix} -9.16 & 9.16 \\[4pt] 0.125 & -0.131 \end{pmatrix}\ \text{hr}^{-1}$$

The diagonal entries differ by a factor of 70. That's the model's way of saying what we already knew: air responds 70 times faster than the wall.

### 3.3 Parameters and Timescales

Parameters are taken from Péan et al. (2018)[^pean2018], identified for a multi-family apartment in Barcelona.

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

At full heater power and no cooling, the air runs about 0.84°C warmer than the wall in steady state — a small but measurable gap that shows up in the plots:
$$\Delta T \approx R_{int} \cdot Q_h = 0.42 \times 2.0 = 0.84\ ^\circ\text{C}$$

---

## 4. The Optimisation

The task: pick heater and cooler duty cycles $\alpha_h(t), \alpha_c(t) \in [0,1]$ over the next 24 hours so that the indoor temperature stays inside the comfort band and the total electricity cost is minimized. Because only the *next step* is ever applied — the LP re-solves every interval with a fresh forecast — this is a **Model Predictive Control** (MPC) loop.

**Why not a thermostat:** it heats when cold and stops when warm. No awareness of what electricity costs at 03:00 vs. 09:00, so it cannot pre-heat. All the savings in this project come from that look-ahead.

**Why not reinforcement learning:** RL can learn complex policies, but it cannot give hard guarantees. A temperature constraint entered as a soft penalty is just that — soft. One cold night is enough to lose a user's trust in an automated system.

**Why a linear programme:** the physical model is linear, the cost function (power × price) is linear, and the constraints (comfort band, duty-cycle bounds) are linear. A linear programme finds the global optimum exactly, with the comfort band as a hard constraint, not a suggestion.

To embed the continuous dynamics in the LP, each ODE step becomes a linear equality constraint using forward Euler with $\Delta t = 5\ \text{min}$:

$$\mathbf{T}(t+1) = \mathbf{T}(t) + \Delta t \left( A\,\mathbf{T}(t) + B\,\mathbf{u}(t) + \mathbf{d}(t) \right)$$

That step size is set by stability: forward Euler is only stable when $\Delta t < 2/|\lambda_{\text{fast}}| \approx 13\ \text{min}$ for the air's fast eigenmode. A 24-hour horizon at 5 min gives 288 timesteps — roughly 1,150 decision variables, solved by CLARABEL in milliseconds.

---

## 5. Tools

| Tool | What it does here |
|---|---|
| **Python + CVXPY** | Formulates the LP; chunked monthly solves stitch a two-year horizon |
| **CLARABEL** | Interior-point conic solver — lightweight, robust, and well-suited to LPs of this size |
| **pandas / numpy** | Time-series alignment, resampling, and per-month accounting |
| **ENTSO-E API** | Day-ahead wholesale electricity prices for the historical backtest and the live forecast |
| **OpenWeatherMap API** | Historical and forecast exterior temperatures |
| **python-kasa** | Sends ON/OFF commands to the TP-Link smart plug in the live loop |

---

## 6. Results

The comparison throughout is a **deadband thermostat**: heat when the temperature drops below $T_{min}$, cool when it rises above $T_{max}$, off otherwise. No price awareness, no forecasting. That's what a household without this system would run.

### 25 November 2024 — single day

<div class="plot-figure">
  <img src="images/compare_costs_temps_Barcelona_25th_Nov.png" alt="Barcelona 25th November comparison"/>
  <div class="plot-caption">Optimal vs. thermostat, 25 November 2024.</div>
</div>

A representative winter day: the optimiser front-loads heating into the cheapest morning hours, avoiding the expensive early-evening peak while keeping the flat within the comfort band throughout.

### November 2024 (second half)

<div class="plot-figure">
  <img src="images/compare_costs_temps_Barcelona_Nov.png" alt="Barcelona November comparison"/>
  <div class="plot-caption">Optimal vs. thermostat, 15–30 November 2024.</div>
</div>

The day wasn't cherry-picked. The same pattern repeats wherever the price schedule offers a cheap window: **36% savings** over the fortnight, again buying more total energy (187 kWh vs 168 kWh). November is heating-dominated, so the optimizer hugs the lower edge of the comfort band most of the time — only pushing higher when pre-charging before a price spike. In summer the pattern reverses: with cooling as the dominant cost, the optimizer hugs the upper edge and pre-cools toward the lower bound when night prices drop.

### Two-year backtest: March 2023 to March 2025

<div class="plot-figure">
  <img src="images/compare_costs_temps_Barcelona_Mar23_Mar25.png" alt="Barcelona Mar 2023 – Mar 2025 comparison"/>
  <div class="plot-caption">Optimal vs. thermostat, March 2023 – March 2025.</div>
</div>

Running the full two years as a single LP isn't feasible: at 5-minute steps that's 210,000 timesteps. Instead, each calendar month is solved independently, with the final temperatures of month $N$ passed as initial conditions to month $N+1$. Each month is saved to disk so the computation can resume if anything fails.

<div class="metrics-grid">
  <div class="metric-card">
    <div class="metric-title">Thermostat cost</div>
    <div class="metric-value neutral">€561.71</div>
  </div>
  <div class="metric-card">
    <div class="metric-title">Optimal cost</div>
    <div class="metric-value neutral">€409.47</div>
  </div>
  <div class="metric-card">
    <div class="metric-title">Saved</div>
    <div class="metric-value positive">€152.24</div>
  </div>
  <div class="metric-card">
    <div class="metric-title">Relative saving</div>
    <div class="metric-value positive">27%</div>
  </div>
</div>

The optimizer spends less while using more energy. Savings come entirely from timing.

The blended 27% number masks three regimes:

- **Free-coast months** (March 2023, October 2023, June 2024): the exterior temperature stays inside the comfort band, so no actuation is required. These confirm the optimizer correctly does nothing when nothing is needed — sanity checks, not wins.
- **Shoulder seasons** (April, May, July, August, late autumn, March 2025): wide price swings combined with ample comfort-band slack let the optimizer save **35–98%** on bills of 5–35 €.
- **Deep winter** (December, January, February): the heater runs nearly full-time just to compensate heat loss, leaving little slack for cheap-hour shifting. Savings here drop to **14–22%**, but these months also dominate the absolute bill.

There is a side effect worth naming. Buying when prices are at or near zero means buying surplus — the periods of solar overproduction or low overnight demand that the grid would otherwise have to spill or curtail. At household scale this is a footnote; at fleet scale, optimizers like this one act as distributed flexibility, absorbing renewable overproduction the grid cannot use directly.

**Temperature smoothness.** Per-step changes stay below 0.7°C / 5 min. Peak-to-peak swing amplitudes within the comfort band average ~1°C, with a 95th percentile of ~2°C; the maximum observed amplitude reaches the full 6°C band on a handful of days when the optimizer fully exploits both bounds.

---

## 7. Conclusion

The path from "automate my heating" to a working system runs through four layers: a physical model of the building, a linear programme that exploits it, an API pipeline that feeds it forecasts, and a Model Predictive Control loop that executes the output. Each layer is as simple as the next one allows. The model is linear, which is what makes the LP fast; the LP is fast, which is what makes the MPC loop tractable; and a tractable MPC loop is what turns a forecast into a plug command every five minutes.

That same linearity is what makes the result interpretable. The schedule the optimizer produces can be read line by line: it pre-heats at 02:00 because electricity is cheapest then, and coasts through 09:00 because the heat stored in the wall is enough to stay comfortable. A black-box model gives you a number. This gives you a reason.

---

## 8. Future Work

- Coordinating multiple devices or households, using the same LP structure with a larger state space
- Handling price and weather uncertainty explicitly, instead of treating day-ahead forecasts as ground truth
- V2G energy management
- Identifying the apartment-specific RC parameters from real measurements rather than literature defaults
- A rate constraint on $|dT_{air}/dt|$ to cap rare large swings; current per-step changes can reach ~0.7°C / 5 min
- Roller shutters as a new control input: they modulate solar gain linearly, so the LP stays intact

---

## 9. References

[^pean2018]: Péan, T., Salom, J., & Costa-Castelló, R. (2018). Configurations of model predictive control to exploit energy flexibility in building thermal loads. In *Proc. 57th IEEE Conference on Decision and Control (CDC)*, Miami, FL, USA, pp. 3177–3182. DOI: 10.1109/CDC.2018.8619095.
