"""Fast 2-year backtest for the website: optimal vs thermostat, 5-min steps.

Runs at the faithful 5-minute resolution (matching the live controller) and skips
the unused LP-baseline. Fast thanks to the vectorized dynamics in
find_heating_output — per-month LP build dropped from ~45 s to ~1-2 s, so the whole
two years solves in a couple of minutes. Writes the opt-vs-thermostat 2-year plot
into docs/observable/src/images/ and prints the cost summary + monthly
vs-thermostat breakdown for updating the case study numbers.
"""
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from strom.optimization_utils import House, find_heating_output, find_heating_output_thermostat
from strom.plot_utils import plot_combined_cases_years

FREQ = "5min"
CHUNK_DIR = Path("data/chunks_Mar23_Mar25_5m")
INPUT_CSV = "data/Temp_Price_Barcelona_Mar23_Mar25.csv"
OUT = Path("docs/observable/src/images")

with open("config/house_config.json") as f:
    house_cfg = json.load(f)

temp_price_df = pd.read_csv(INPUT_CSV, index_col="Timestamp", parse_dates=["Timestamp"])
temp_price_df = temp_price_df.rename(columns={"Exterior Temperature": "ExteriorTemperature"})
months = sorted({(d.year, d.month) for d in temp_price_df.index})
CHUNK_DIR.mkdir(parents=True, exist_ok=True)


def _house(seam_T_int, seam_T_wall):
    return House(**{**house_cfg, "T_interior_init": seam_T_int, "T_wall_init": seam_T_wall}, freq=FREQ)


def solve_or_load(month_df, mode, seam, path):
    if path.exists():
        return pd.read_csv(path, index_col="Timestamp", parse_dates=["Timestamp"])
    chunk = find_heating_output(month_df, _house(*seam), mode)
    if chunk["InteriorTemperature"].isna().any():
        raise RuntimeError(f"Infeasible {mode} LP for {path.stem}")
    chunk.to_csv(path)
    return chunk


def solve_or_load_thermostat(month_df, seam, path):
    if path.exists():
        return pd.read_csv(path, index_col="Timestamp", parse_dates=["Timestamp"])
    chunk = find_heating_output_thermostat(month_df, _house(*seam))
    chunk.to_csv(path)
    return chunk


seam_opt = (house_cfg["T_interior_init"], house_cfg["T_wall_init"])
seam_therm = (house_cfg["T_interior_init"], house_cfg["T_wall_init"])
opt_chunks, therm_chunks = [], []

for year, month in months:
    tag = f"{year:04d}-{month:02d}"
    print(f"=== Month {tag} ===", flush=True)
    month_df = temp_price_df[(temp_price_df.index.year == year) & (temp_price_df.index.month == month)]
    opt_chunk   = solve_or_load(month_df, "optimal", seam_opt, CHUNK_DIR / f"optimal_{tag}.csv")
    therm_chunk = solve_or_load_thermostat(month_df, seam_therm, CHUNK_DIR / f"thermostat_{tag}.csv")
    seam_opt   = (opt_chunk["InteriorTemperature"].iloc[-1],   opt_chunk["WallTemperature"].iloc[-1])
    seam_therm = (therm_chunk["InteriorTemperature"].iloc[-1], therm_chunk["WallTemperature"].iloc[-1])
    opt_chunks.append(opt_chunk)
    therm_chunks.append(therm_chunk)

optimal_state_df    = pd.concat(opt_chunks)
thermostat_state_df = pd.concat(therm_chunks)

OUT.mkdir(parents=True, exist_ok=True)
fig = plot_combined_cases_years(optimal_state_df, thermostat_state_df, compare_label="Thermostat")
fig.savefig(OUT / "compare_costs_temps_Barcelona_Mar23_Mar25.png", dpi=150, bbox_inches="tight")
fig.savefig("./plots/compare_costs_temps_Barcelona_Mar23_Mar25.png", dpi=150, bbox_inches="tight")
plt.close(fig)

opt_cost   = optimal_state_df["Cost"].sum()
therm_cost = thermostat_state_df["Cost"].sum()
print(f"\n================ SUMMARY ({FREQ}, opt vs thermostat) ================", flush=True)
print(f"Thermostat cost : {therm_cost:.4f} EUR", flush=True)
print(f"Optimal    cost : {opt_cost:.4f} EUR", flush=True)
print(f"Saved           : {therm_cost - opt_cost:.4f} EUR ({(therm_cost - opt_cost) / therm_cost * 100:.2f}%)", flush=True)

opt_m   = optimal_state_df["Cost"].resample("ME").sum()
therm_m = thermostat_state_df["Cost"].resample("ME").sum()
print("\nMonthly vs-thermostat:", flush=True)
for m in opt_m.index:
    pct = (therm_m[m] - opt_m[m]) / therm_m[m] * 100 if therm_m[m] else float("nan")
    print(f"  {m.strftime('%Y-%m')}  therm {therm_m[m]:7.2f}  opt {opt_m[m]:7.2f}  saved {pct:6.1f}%", flush=True)
print("DONE", flush=True)
