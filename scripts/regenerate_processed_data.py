"""Regenerate processed Temp+Price CSVs from the raw VisualCrossing temperature
downloads and the historical price archive (data/Spain.csv).

Run from the project root:
    python scripts/regenerate_processed_data.py

strom.data_utils.get_temp_price_from_temp would call the live ENTSO-E API
via get_price_series, which only returns today + 24h. For historical
reprocessing we read data/Spain.csv directly (hourly EUR/MWh archive) and
replicate the temp-side logic from get_temp_price_from_temp here.
"""
from pathlib import Path

import pandas as pd

from strom.data_utils import join_data, remove_temperature_spikes

DATA_DIR = Path("data")
PRICE_ARCHIVE = DATA_DIR / "Spain.csv"

RAW_TO_PROCESSED = [
    ("Barcelona 2024-11-01 to 2024-11-30.csv", "Temp_Price_Barcelona_Nov.csv"),
    ("barcelona 2023-03-31 to 2025-03-31.csv", "Temp_Price_Barcelona_Mar23_Mar25.csv"),
]


def load_historical_prices() -> pd.Series:
    df = pd.read_csv(PRICE_ARCHIVE)
    timestamps = pd.to_datetime(df["Datetime (UTC)"]).dt.tz_localize("UTC")
    return pd.Series(
        df["Price (EUR/MWhe)"].values / 1000.0,  # EUR/MWh -> EUR/kWh
        index=timestamps,
        name="Price",
    )


def temp_csv_to_series(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    timestamps = (
        pd.to_datetime(df["datetimeEpoch"], unit="s")
        .dt.tz_localize("Europe/Madrid", ambiguous="NaT", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
    temps = df["temp"].values
    if temps.max() > 50:  # implausible in Celsius — VisualCrossing default is Imperial
        temps = (temps - 32) * 5.0 / 9.0
    series = pd.Series(temps, index=timestamps, name="ExteriorTemperature")
    series = series.groupby(series.index).mean().resample("h").interpolate("time")
    before = series.copy()
    series = remove_temperature_spikes(series)
    n_spikes = int((before - series).abs().gt(1e-6).sum())
    if n_spikes:
        print(f"  Despiked {n_spikes} salt-and-pepper spikes from {path.name}")
    series.index.name = "Timestamp"
    return series


def main() -> None:
    prices = load_historical_prices()
    print(
        f"Loaded {len(prices)} hourly prices from {prices.index.min()} "
        f"to {prices.index.max()}"
    )
    for raw_name, processed_name in RAW_TO_PROCESSED:
        temp = temp_csv_to_series(DATA_DIR / raw_name)
        print(
            f"\n{raw_name}: temp covers {temp.index.min()} -> {temp.index.max()} "
            f"({len(temp)} rows)"
        )
        if temp.index.max() > prices.index.max():
            gap_h = (temp.index.max() - prices.index.max()).total_seconds() / 3600
            print(
                f"  WARNING: temp extends {gap_h:.0f}h past the price archive. "
                f"Prices for the gap will be cubic-interpolated/extrapolated by "
                f"join_data and are NOT real market data. Refresh Spain.csv or "
                f"truncate the temp range before running the optimization."
            )
        if temp.index.min() < prices.index.min():
            gap_h = (prices.index.min() - temp.index.min()).total_seconds() / 3600
            print(f"  WARNING: temp starts {gap_h:.0f}h before the price archive.")

        merged = join_data(temp, prices)
        merged = merged.loc[temp.index.min() : temp.index.max()]
        merged.index.name = "Timestamp"
        out_path = DATA_DIR / processed_name
        merged.to_csv(out_path)
        print(f"  Wrote {out_path} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
