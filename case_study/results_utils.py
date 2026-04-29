from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from strom.optimization_utils import House


def compute_and_save_results(
    optimal_state_df: pd.DataFrame,
    baseline_state_df: pd.DataFrame,
    output_path: str,
    house: House,
    label: str = "",
) -> None:
    """Compute quantitative metrics and write them to a text file.

    Metrics saved:
      - Cost summary (absolute and relative savings, baseline total as denominator)
      - Energy consumption breakdown (heating vs cooling; confirms load shifting)
      - Temperature constraint audit (violation counts, interior range)
      - Temperature volatility (mean, std, max, P95 of |ΔT| per timestep)
      - Monthly cost breakdown and heating/cooling split (for runs > 60 days)
    """
    lines = []
    if label:
        lines += [f"Results: {label}", "=" * (len(label) + 9), ""]

    # Infer timestep from index
    dt = (optimal_state_df.index[1] - optimal_state_df.index[0]).total_seconds() / 3600.0

    # ── Cost ─────────────────────────────────────────────────────────────────
    opt_cost = optimal_state_df["Cost"].sum()
    base_cost = baseline_state_df["Cost"].sum()
    abs_savings = base_cost - opt_cost
    rel_savings = (abs_savings / base_cost * 100) if base_cost else float("nan")

    lines += [
        "Cost Summary",
        "───────────",
        f"  Baseline total cost : {base_cost:.4f} €",
        f"  Optimal  total cost : {opt_cost:.4f} €",
        f"  Absolute savings    : {abs_savings:.4f} €",
        f"  Relative savings    : {rel_savings:.2f}%",
        "",
    ]

    # ── Energy ───────────────────────────────────────────────────────────────
    opt_heat_kwh  = (optimal_state_df["HeaterOutput"]  * house.Q_heater  * dt).sum()
    opt_cool_kwh  = (optimal_state_df["CoolingOutput"] * house.Q_cooling * dt).sum()
    base_heat_kwh = (baseline_state_df["HeaterOutput"]  * house.Q_heater  * dt).sum()
    base_cool_kwh = (baseline_state_df["CoolingOutput"] * house.Q_cooling * dt).sum()

    lines += ["Energy Consumption", "──────────────────"]
    lines.append(f"  Optimal  heating : {opt_heat_kwh:.2f} kWh")
    if house.Q_cooling > 0:
        lines.append(f"  Optimal  cooling : {opt_cool_kwh:.2f} kWh")
    lines.append(f"  Optimal  total   : {opt_heat_kwh + opt_cool_kwh:.2f} kWh")
    lines.append(f"  Baseline heating : {base_heat_kwh:.2f} kWh")
    if house.Q_cooling > 0:
        lines.append(f"  Baseline cooling : {base_cool_kwh:.2f} kWh")
    lines.append(f"  Baseline total   : {base_heat_kwh + base_cool_kwh:.2f} kWh")
    lines.append(
        "  (Compare totals: equal => pure load shifting; optimal > baseline => savings from buying more at cheaper hours)"
    )
    lines.append("")

    # ── Temperature constraints ───────────────────────────────────────────────
    tol = 1e-3
    opt_T = optimal_state_df["InteriorTemperature"]
    tmin_viol = (opt_T < house.T_min - tol).sum()
    tmax_viol = (opt_T > house.T_max + tol).sum()

    lines += [
        "Temperature Constraints (Optimal)",
        "─────────────────────────────────",
        f"  Comfort band        : [{house.T_min} °C, {house.T_max} °C]",
        f"  T_min violations    : {tmin_viol}",
        f"  T_max violations    : {tmax_viol}",
        f"  Interior range      : [{opt_T.min():.2f} °C, {opt_T.max():.2f} °C]",
        "",
    ]

    # ── Temperature volatility ────────────────────────────────────────────────
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

    # ── Swing amplitude (charging-event peak prominence) ─────────────────────
    # Prominence = how far a peak rises above the deepest valley between it and
    # the next-higher peak. Captures what a user *feels*: the magnitude of
    # warm/cool excursions between heating events. The 0.5 °C threshold filters
    # bang-bang noise from the LP solution.
    prominence_threshold = 0.5
    _, heat_props = find_peaks(opt_T.values,  prominence=prominence_threshold)
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
        opt_monthly  = optimal_state_df["Cost"].resample("ME").sum()
        base_monthly = baseline_state_df["Cost"].resample("ME").sum()
        savings_abs  = base_monthly - opt_monthly
        savings_pct  = (savings_abs / base_monthly * 100).round(1)

        header = f"  {'Month':<10} {'Baseline €':>10} {'Optimal €':>10} {'Saved €':>8} {'Saved %':>8}"
        sep    = "  " + "─" * (len(header) - 2)
        rows   = [
            f"  {m.strftime('%Y-%m'):<10} {base_monthly[m]:>10.2f} {opt_monthly[m]:>10.2f}"
            f" {savings_abs[m]:>8.2f} {savings_pct[m]:>7.1f}%"
            for m in opt_monthly.index
        ]

        lines += ["Monthly Cost Breakdown", "──────────────────────", header, sep] + rows + [""]

        # Heating vs cooling cost split
        if house.Q_cooling > 0:
            opt_heat_cost  = (optimal_state_df["Price"]  * dt * optimal_state_df["HeaterOutput"]  * house.Q_heater).sum()
            opt_cool_cost  = (optimal_state_df["Price"]  * dt * optimal_state_df["CoolingOutput"] * house.Q_cooling).sum()
            base_heat_cost = (baseline_state_df["Price"] * dt * baseline_state_df["HeaterOutput"]  * house.Q_heater).sum()
            base_cool_cost = (baseline_state_df["Price"] * dt * baseline_state_df["CoolingOutput"] * house.Q_cooling).sum()

            heat_saved_pct = (base_heat_cost - opt_heat_cost) / base_heat_cost * 100 if base_heat_cost else float("nan")
            cool_saved_pct = (base_cool_cost - opt_cool_cost) / base_cool_cost * 100 if base_cool_cost else float("nan")

            lines += [
                "Heating vs Cooling Cost Split",
                "─────────────────────────────",
                f"  Optimal  heating cost : {opt_heat_cost:.2f} €",
                f"  Optimal  cooling cost : {opt_cool_cost:.2f} €",
                f"  Baseline heating cost : {base_heat_cost:.2f} €",
                f"  Baseline cooling cost : {base_cool_cost:.2f} €",
                f"  Heating savings       : {base_heat_cost - opt_heat_cost:.2f} € ({heat_saved_pct:.1f}%)",
                f"  Cooling savings       : {base_cool_cost - opt_cool_cost:.2f} € ({cool_saved_pct:.1f}%)",
                "",
            ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Results saved: {output_path}")
