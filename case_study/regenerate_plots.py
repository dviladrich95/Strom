"""Regenerate all three case-study PNGs at 150 dpi into docs/observable/src/images/."""
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strom.optimization_utils import House
from strom.plot_utils import plot_combined_cases, plot_combined_cases_years
from strom.data_utils import remove_temperature_spikes
from case_study.results_utils import solve_or_load_case

DPI = 150
OUT = Path("docs/observable/src/images")
OUT.mkdir(parents=True, exist_ok=True)

with open("config/house_config.json") as f:
    house_cfg = json.load(f)

# ── 1. Single-day view: Nov 24-25 ────────────────────────────────────────────
print("=== Plot 1: Nov 24-25 ===")
nov_df = pd.read_csv(
    "data/Temp_Price_Barcelona_Nov.csv",
    index_col="Timestamp", parse_dates=["Timestamp"],
)
nov_df["ExteriorTemperature"] = remove_temperature_spikes(nov_df["ExteriorTemperature"])

# Slice to Nov 24 00:00 – Nov 26 00:00
day_df = nov_df[
    (nov_df.index >= "2024-11-24") & (nov_df.index < "2024-11-26")
]
house = House(**house_cfg, freq="5min")
opt_day, base_day, _ = solve_or_load_case(day_df, house, "data/chunks_25thNov")
fig = plot_combined_cases(opt_day, base_day)
dest = OUT / "compare_costs_temps_Barcelona_25th_Nov.png"
fig.savefig(dest, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {dest}")

# ── 2. November second-half view ─────────────────────────────────────────────
print("=== Plot 2: Nov 15-30 ===")
half = len(nov_df) // 2
nov_half_df = nov_df.iloc[half:]
house = House(**house_cfg, freq="5min")
opt_nov, base_nov, _ = solve_or_load_case(nov_half_df, house, "data/chunks_Nov")
fig = plot_combined_cases(opt_nov, base_nov)
dest = OUT / "compare_costs_temps_Barcelona_Nov.png"
fig.savefig(dest, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {dest}")

# ── 3. Two-year view (pre-computed CSVs from Julia run) ───────────────────────
print("=== Plot 3: Mar 2023 – Mar 2025 ===")
opt2  = pd.read_csv("data/optimal_state2.csv",  index_col="Timestamp", parse_dates=["Timestamp"])
base2 = pd.read_csv("data/baseline_state2.csv", index_col="Timestamp", parse_dates=["Timestamp"])
fig = plot_combined_cases_years(opt2, base2)
dest = OUT / "compare_costs_temps_Barcelona_Mar23_Mar25.png"
fig.savefig(dest, dpi=DPI, bbox_inches="tight")
plt.close(fig)
print(f"  Saved {dest}")

print("Done.")
