from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from strom.optimization_utils import House, compare_output_costs


def solve_or_load_case(
    temp_price_df: pd.DataFrame,
    house: House,
    cache_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache_dir = Path(cache_dir)
    opt_path   = cache_dir / "optimal.csv"
    base_path  = cache_dir / "baseline.csv"
    therm_path = cache_dir / "thermostat.csv"

    if opt_path.exists() and base_path.exists() and therm_path.exists():
        print(f"Loading cached results from {cache_dir}")
        return (
            pd.read_csv(opt_path,   index_col="Timestamp", parse_dates=["Timestamp"]),
            pd.read_csv(base_path,  index_col="Timestamp", parse_dates=["Timestamp"]),
            pd.read_csv(therm_path, index_col="Timestamp", parse_dates=["Timestamp"]),
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    optimal_df, baseline_df, thermostat_df = compare_output_costs(temp_price_df, house)
    optimal_df.to_csv(opt_path)
    baseline_df.to_csv(base_path)
    thermostat_df.to_csv(therm_path)
    print(f"Saved results to {cache_dir}")
    return optimal_df, baseline_df, thermostat_df


def compute_and_save_results(
    optimal_state_df: pd.DataFrame,
    baseline_state_df: pd.DataFrame,
    thermostat_state_df: pd.DataFrame,
    output_path: str,
    house: House,
    label: str = "",
) -> None:
    """Compute quantitative metrics and write them to a text file.

    Metrics saved:
      - Three-way cost summary (LP-baseline, thermostat, optimal)
      - Energy consumption breakdown (heating vs cooling)
      - Temperature constraint audit (optimal)
      - Temperature volatility (optimal)
      - Monthly cost breakdown for runs > 60 days
    """
    lines = []
    if label:
        lines += [f"Results: {label}", "=" * (len(label) + 9), ""]

    dt = (optimal_state_df.index[1] - optimal_state_df.index[0]).total_seconds() / 3600.0

    # ── Cost ─────────────────────────────────────────────────────────────────
    opt_cost   = optimal_state_df["Cost"].sum()
    base_cost  = baseline_state_df["Cost"].sum()
    therm_cost = thermostat_state_df["Cost"].sum()

    def _savings(ref, opt):
        abs_s = ref - opt
        rel_s = (abs_s / ref * 100) if ref else float("nan")
        return abs_s, rel_s

    abs_vs_base,  rel_vs_base  = _savings(base_cost,  opt_cost)
    abs_vs_therm, rel_vs_therm = _savings(therm_cost, opt_cost)

    col = 28
    lines += [
        "Cost Summary",
        "───────────",
        f"  {'LP-baseline total cost':<{col}}: {base_cost:.4f} €",
        f"  {'Thermostat  total cost':<{col}}: {therm_cost:.4f} €",
        f"  {'Optimal     total cost':<{col}}: {opt_cost:.4f} €",
        f"  {'Savings vs LP-baseline':<{col}}: {abs_vs_base:.4f} € ({rel_vs_base:.2f}%)",
        f"  {'Savings vs thermostat':<{col}}: {abs_vs_therm:.4f} € ({rel_vs_therm:.2f}%)",
        "",
    ]

    # ── Energy ───────────────────────────────────────────────────────────────
    opt_heat_kwh   = (optimal_state_df["HeaterOutput"]    * house.Q_heater  * dt).sum()
    opt_cool_kwh   = (optimal_state_df["CoolingOutput"]   * house.Q_cooling * dt).sum()
    base_heat_kwh  = (baseline_state_df["HeaterOutput"]   * house.Q_heater  * dt).sum()
    base_cool_kwh  = (baseline_state_df["CoolingOutput"]  * house.Q_cooling * dt).sum()
    therm_heat_kwh = (thermostat_state_df["HeaterOutput"]  * house.Q_heater  * dt).sum()
    therm_cool_kwh = (thermostat_state_df["CoolingOutput"] * house.Q_cooling * dt).sum()

    lines += ["Energy Consumption", "──────────────────"]
    lines.append(f"  Optimal      heating : {opt_heat_kwh:.2f} kWh")
    if house.Q_cooling > 0:
        lines.append(f"  Optimal      cooling : {opt_cool_kwh:.2f} kWh")
    lines.append(f"  Optimal      total   : {opt_heat_kwh + opt_cool_kwh:.2f} kWh")
    lines.append(f"  LP-baseline  heating : {base_heat_kwh:.2f} kWh")
    if house.Q_cooling > 0:
        lines.append(f"  LP-baseline  cooling : {base_cool_kwh:.2f} kWh")
    lines.append(f"  LP-baseline  total   : {base_heat_kwh + base_cool_kwh:.2f} kWh")
    lines.append(f"  Thermostat   heating : {therm_heat_kwh:.2f} kWh")
    if house.Q_cooling > 0:
        lines.append(f"  Thermostat   cooling : {therm_cool_kwh:.2f} kWh")
    lines.append(f"  Thermostat   total   : {therm_heat_kwh + therm_cool_kwh:.2f} kWh")
    lines.append("")

    # ── Temperature constraints (optimal) ────────────────────────────────────
    tol   = 1e-3
    opt_T = optimal_state_df["InteriorTemperature"]
    lines += [
        "Temperature Constraints (Optimal)",
        "─────────────────────────────────",
        f"  Comfort band     : [{house.T_min} °C, {house.T_max} °C]",
        f"  T_min violations : {(opt_T < house.T_min - tol).sum()}",
        f"  T_max violations : {(opt_T > house.T_max + tol).sum()}",
        f"  Interior range   : [{opt_T.min():.2f} °C, {opt_T.max():.2f} °C]",
        "",
    ]

    # ── Temperature constraints (thermostat) ─────────────────────────────────
    th_T = thermostat_state_df["InteriorTemperature"]
    lines += [
        "Temperature Constraints (Thermostat)",
        "─────────────────────────────────────",
        f"  T_min violations : {(th_T < house.T_min - tol).sum()}",
        f"  T_max violations : {(th_T > house.T_max + tol).sum()}",
        f"  Interior range   : [{th_T.min():.2f} °C, {th_T.max():.2f} °C]",
        "",
    ]

    # ── Temperature volatility (optimal) ─────────────────────────────────────
    diff = opt_T.diff().abs().dropna()
    lines += [
        f"Temperature Volatility — Optimal (timestep = {dt:.4g} h)",
        "─────────────────────────────────────────────────────────",
        f"  Mean  |ΔT| : {diff.mean():.3f} °C",
        f"  Std   |ΔT| : {diff.std():.3f} °C",
        f"  P95   |ΔT| : {diff.quantile(0.95):.3f} °C",
        f"  Max   |ΔT| : {diff.max():.3f} °C",
        "",
    ]

    # ── Swing amplitude (optimal) ─────────────────────────────────────────────
    prominence_threshold = 0.5
    _, heat_props = find_peaks( opt_T.values, prominence=prominence_threshold)
    _, cool_props = find_peaks(-opt_T.values, prominence=prominence_threshold)
    heat_amps = heat_props["prominences"]
    cool_amps = cool_props["prominences"]
    all_amps  = np.concatenate([heat_amps, cool_amps])

    lines += [
        f"Swing Amplitude — Optimal (peak prominence ≥ {prominence_threshold} °C)",
        "─────────────────────────────────────────────────────────",
    ]
    if len(all_amps) == 0:
        lines += ["  (no swings detected above prominence threshold)", ""]
    else:
        lines += [
            f"  Heating events  : {len(heat_amps)}",
            f"  Cooling events  : {len(cool_amps)}",
            f"  Mean  amplitude : {all_amps.mean():.2f} °C",
            f"  P95   amplitude : {np.percentile(all_amps, 95):.2f} °C",
            f"  Max   amplitude : {all_amps.max():.2f} °C",
            "",
        ]

    # ── Seasonal breakdown (runs > 60 days) ───────────────────────────────────
    span_days = (optimal_state_df.index[-1] - optimal_state_df.index[0]).days
    if span_days > 60:
        opt_monthly   = optimal_state_df["Cost"].resample("ME").sum()
        base_monthly  = baseline_state_df["Cost"].resample("ME").sum()
        therm_monthly = thermostat_state_df["Cost"].resample("ME").sum()

        sav_base_abs  = base_monthly  - opt_monthly
        sav_therm_abs = therm_monthly - opt_monthly
        sav_base_pct  = (sav_base_abs  / base_monthly  * 100).round(1)
        sav_therm_pct = (sav_therm_abs / therm_monthly * 100).round(1)

        header = (
            f"  {'Month':<10} {'LP-Base €':>10} {'Therm €':>9} {'Optimal €':>10}"
            f" {'vs LP €':>8} {'vs LP%':>7} {'vs Th €':>8} {'vs Th%':>7}"
        )
        sep  = "  " + "─" * (len(header) - 2)
        rows = [
            f"  {m.strftime('%Y-%m'):<10}"
            f" {base_monthly[m]:>10.2f}"
            f" {therm_monthly[m]:>9.2f}"
            f" {opt_monthly[m]:>10.2f}"
            f" {sav_base_abs[m]:>8.2f}"
            f" {sav_base_pct[m]:>6.1f}%"
            f" {sav_therm_abs[m]:>8.2f}"
            f" {sav_therm_pct[m]:>6.1f}%"
            for m in opt_monthly.index
        ]
        lines += ["Monthly Cost Breakdown (Three-Way)", "──────────────────────────────────", header, sep] + rows + [""]

        if house.Q_cooling > 0:
            opt_heat_cost   = (optimal_state_df["Price"]    * dt * optimal_state_df["HeaterOutput"]    * house.Q_heater).sum()
            opt_cool_cost   = (optimal_state_df["Price"]    * dt * optimal_state_df["CoolingOutput"]   * house.Q_cooling).sum()
            base_heat_cost  = (baseline_state_df["Price"]   * dt * baseline_state_df["HeaterOutput"]   * house.Q_heater).sum()
            base_cool_cost  = (baseline_state_df["Price"]   * dt * baseline_state_df["CoolingOutput"]  * house.Q_cooling).sum()
            therm_heat_cost = (thermostat_state_df["Price"] * dt * thermostat_state_df["HeaterOutput"]  * house.Q_heater).sum()
            therm_cool_cost = (thermostat_state_df["Price"] * dt * thermostat_state_df["CoolingOutput"] * house.Q_cooling).sum()

            def _pct(ref, opt):
                return (ref - opt) / ref * 100 if ref else float("nan")

            lines += [
                "Heating vs Cooling Cost Split",
                "─────────────────────────────",
                f"  {'':22} {'LP-base':>10} {'Thermostat':>11} {'Optimal':>9}",
                f"  {'Heating cost (€)':<22} {base_heat_cost:>10.2f} {therm_heat_cost:>11.2f} {opt_heat_cost:>9.2f}",
                f"  {'Cooling cost (€)':<22} {base_cool_cost:>10.2f} {therm_cool_cost:>11.2f} {opt_cool_cost:>9.2f}",
                f"  {'Heating savings vs LP (%)':<22} {'—':>10} {'—':>11} {_pct(base_heat_cost, opt_heat_cost):>8.1f}%",
                f"  {'Cooling savings vs LP (%)':<22} {'—':>10} {'—':>11} {_pct(base_cool_cost, opt_cool_cost):>8.1f}%",
                f"  {'Heating savings vs Th (%)':<22} {'—':>10} {'—':>11} {_pct(therm_heat_cost, opt_heat_cost):>8.1f}%",
                f"  {'Cooling savings vs Th (%)':<22} {'—':>10} {'—':>11} {_pct(therm_cool_cost, opt_cool_cost):>8.1f}%",
                "",
            ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Results saved: {output_path}")
