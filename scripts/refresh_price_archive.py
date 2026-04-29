"""Refresh data/Spain.csv with day-ahead electricity prices from ENTSO-E.

Run from the project root:
    python scripts/refresh_price_archive.py
    python scripts/refresh_price_archive.py --start 2025-01-09 --end 2025-04-01

With no arguments, appends new rows starting one hour after the latest
row already in Spain.csv, through tomorrow (covering today's day-ahead
auction result if the call is made after ~13:00 CET).

Uses entsoe-py, which wraps the ENTSO-E Transparency Platform REST API
(https://transparencyplatform.zendesk.com/hc/en-us/articles/15692855254548).
Requires PRICE_API_KEY in the environment, or the token file at
config/price_api_key.txt.
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient

from strom.api_utils import read_api_key

ARCHIVE_PATH = Path("data/Spain.csv")


def get_api_key() -> str:
    return os.getenv("PRICE_API_KEY") or read_api_key("./config/price_api_key.txt")


def fetch_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    client = EntsoePandasClient(api_key=get_api_key())
    print(f"Fetching ENTSO-E day-ahead prices for ES: {start} -> {end}")
    return client.query_day_ahead_prices("ES", start=start, end=end)


def to_archive_format(series: pd.Series) -> pd.DataFrame:
    utc = series.index.tz_convert("UTC").tz_localize(None)
    local = series.index.tz_convert("Europe/Madrid").tz_localize(None)
    return pd.DataFrame(
        {
            "Country": "Spain",
            "ISO3 Code": "ESP",
            "Datetime (UTC)": utc.strftime("%Y-%m-%d %H:%M:%S"),
            "Datetime (Local)": local.strftime("%Y-%m-%d %H:%M:%S"),
            "Price (EUR/MWhe)": series.values,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="Inclusive start date (YYYY-MM-DD), UTC.")
    parser.add_argument("--end", help="Exclusive end date (YYYY-MM-DD), UTC.")
    args = parser.parse_args()

    archive = pd.read_csv(ARCHIVE_PATH)
    last_utc = pd.to_datetime(archive["Datetime (UTC)"].iloc[-1])
    print(f"Archive currently ends at {last_utc} UTC ({len(archive)} rows).")

    start = (
        pd.Timestamp(args.start, tz="UTC")
        if args.start
        else (last_utc + pd.Timedelta(hours=1)).tz_localize("UTC")
    )
    end = (
        pd.Timestamp(args.end, tz="UTC")
        if args.end
        else pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=2)
    )

    if start >= end:
        print(f"Nothing to fetch (start {start} >= end {end}).")
        return

    series = fetch_prices(start, end)
    new_rows = to_archive_format(series)
    print(f"Got {len(new_rows)} new rows: {new_rows['Datetime (UTC)'].iloc[0]} -> {new_rows['Datetime (UTC)'].iloc[-1]}")

    combined = pd.concat([archive, new_rows], ignore_index=True)
    before = len(combined)
    combined.drop_duplicates(subset=["Datetime (UTC)"], keep="last", inplace=True)
    combined.to_csv(ARCHIVE_PATH, index=False)
    print(f"Wrote {ARCHIVE_PATH}: {before} -> {len(combined)} rows after dedup.")


if __name__ == "__main__":
    main()
