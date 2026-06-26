import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from strom.optimization_utils import House, find_heating_output, find_heating_output_thermostat
from strom.plot_utils import plot_combined_cases_years
from case_study.results_utils import compute_and_save_results

FREQ = "5min"
CHUNK_DIR = Path("data/chunks_Mar23_Mar25")
INPUT_CSV = "data/Temp_Price_Barcelona_Mar23_Mar25.csv"

with open("config/house_config.json") as f:
    house_cfg = json.load(f)

temp_price_df = pd.read_csv(INPUT_CSV, index_col="Timestamp", parse_dates=["Timestamp"])
temp_price_df = temp_price_df.rename(columns={"Exterior Temperature": "ExteriorTemperature"})
assert isinstance(temp_price_df.index, pd.DatetimeIndex)
months = sorted({(d.year, d.month) for d in temp_price_df.index})
CHUNK_DIR.mkdir(parents=True, exist_ok=True)


def solve_or_load(month_df, mode, seam_T_int, seam_T_wall, path):
    if path.exists():
        return pd.read_csv(path, index_col="Timestamp", parse_dates=["Timestamp"])
    house = House(
        **{**house_cfg, "T_interior_init": seam_T_int, "T_wall_init": seam_T_wall},
        freq=FREQ,
    )
    chunk = find_heating_output(month_df, house, mode)
    if chunk["InteriorTemperature"].isna().any():
        raise RuntimeError(f"Infeasible {mode} LP for {path.stem}")
    chunk.to_csv(path)
    return chunk


def solve_or_load_thermostat(month_df, seam_T_int, seam_T_wall, path):
    if path.exists():
        return pd.read_csv(path, index_col="Timestamp", parse_dates=["Timestamp"])
    house = House(
        **{**house_cfg, "T_interior_init": seam_T_int, "T_wall_init": seam_T_wall},
        freq=FREQ,
    )
    chunk = find_heating_output_thermostat(month_df, house)
    chunk.to_csv(path)
    return chunk


seam_opt   = (house_cfg["T_interior_init"], house_cfg["T_wall_init"])
seam_base  = (house_cfg["T_interior_init"], house_cfg["T_wall_init"])
seam_therm = (house_cfg["T_interior_init"], house_cfg["T_wall_init"])

opt_chunks, base_chunks, therm_chunks = [], [], []
for year, month in months:
    tag = f"{year:04d}-{month:02d}"
    print(f"\n=== Month {tag} ===")
    month_df = temp_price_df[
        (temp_price_df.index.year == year) & (temp_price_df.index.month == month)
    ]

    opt_chunk   = solve_or_load(month_df, "optimal",  *seam_opt,   CHUNK_DIR / f"optimal_{tag}.csv")
    base_chunk  = solve_or_load(month_df, "baseline", *seam_base,  CHUNK_DIR / f"baseline_{tag}.csv")
    therm_chunk = solve_or_load_thermostat(month_df,  *seam_therm, CHUNK_DIR / f"thermostat_{tag}.csv")

    seam_opt   = (opt_chunk["InteriorTemperature"].iloc[-1],   opt_chunk["WallTemperature"].iloc[-1])
    seam_base  = (base_chunk["InteriorTemperature"].iloc[-1],  base_chunk["WallTemperature"].iloc[-1])
    seam_therm = (therm_chunk["InteriorTemperature"].iloc[-1], therm_chunk["WallTemperature"].iloc[-1])

    opt_chunks.append(opt_chunk)
    base_chunks.append(base_chunk)
    therm_chunks.append(therm_chunk)

optimal_state_df    = pd.concat(opt_chunks)
baseline_state_df   = pd.concat(base_chunks)
thermostat_state_df = pd.concat(therm_chunks)

house = House(**house_cfg, freq="1h")

fig = plot_combined_cases_years(optimal_state_df, thermostat_state_df, compare_label="Thermostat")
fig.savefig("./plots/compare_costs_temps_Barcelona_Mar23_Mar25.png", dpi=150, bbox_inches="tight")
fig.savefig("docs/observable/src/images/compare_costs_temps_Barcelona_Mar23_Mar25.png", dpi=150, bbox_inches="tight")

compute_and_save_results(
    optimal_state_df,
    baseline_state_df,
    thermostat_state_df,
    "./results/results_Barcelona_Mar23_Mar25.txt",
    house,
    label="Barcelona — March 2023 to March 2025",
)

plt.close()
