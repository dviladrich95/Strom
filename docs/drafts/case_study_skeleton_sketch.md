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
