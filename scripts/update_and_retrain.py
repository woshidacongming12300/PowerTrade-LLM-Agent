"""
Auto-update script: fetch latest MISO LMP data, append to CSV, retrain LSTM.

Usage:
    python scripts/update_and_retrain.py          # one-shot update
    python scripts/update_and_retrain.py --cron   # cron mode: quiet unless error

Schedule (crontab, runs daily at 6 AM):
    0 6 * * * cd /path/to/project && python scripts/update_and_retrain.py --cron
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd
from algorithms.epf_predictor import train_model
from algorithms.price_data_provider import get_default_historical_csv_path, get_last_csv_delivery_date


def fetch_new_prices_since(last_date: date) -> list[dict]:
    """
    Fetch daily average MISO Indiana Hub LMP from gridstatus.
    Returns list of {"date": date, "price": float} from last_date+1 to today.
    """
    try:
        from gridstatus import MISO
    except ImportError:
        print("[Update] gridstatus not installed, skipping data fetch")
        return []

    from algorithms.gridstatus_feed import _market_from_env, _hub_locations, _tz

    market = _market_from_env()
    tz = _tz()
    locations = _hub_locations()
    m = MISO()

    start_d = last_date + timedelta(days=1)
    end_d = date.today()

    if start_d > end_d:
        print(f"[Update] Data already up to date (last: {last_date})")
        return []

    days_range = (end_d - start_d).days + 1
    if days_range < 1:
        return []

    print(f"[Update] Fetching MISO LMP from {start_d} to {end_d} ({days_range} days)...")

    try:
        df = m.get_lmp(
            pd.Timestamp(start_d),
            end=pd.Timestamp(end_d) + pd.Timedelta(days=1),
            market=market,
            locations=locations,
        )
    except Exception as e:
        print(f"[Update] Gridstatus fetch failed: {e}")
        return []

    if df is None or df.empty:
        print("[Update] No new data available (MISO publish delay)")
        return []

    if "LMP" not in df.columns or "Interval Start" not in df.columns:
        print("[Update] Unexpected gridstatus response format")
        return []

    s = pd.to_datetime(df["Interval Start"])
    if s.dt.tz is None:
        s = s.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    else:
        s = s.dt.tz_convert(tz)

    delivery_date = s.dt.date
    daily = (
        pd.DataFrame({"delivery_date": delivery_date, "LMP": df["LMP"].astype(float)})
        .groupby("delivery_date", sort=True)["LMP"]
        .mean()
    )

    results = []
    for d in sorted(daily.index):
        if d > last_date and d <= end_d:
            results.append({
                "date": d,
                "price": round(float(daily.loc[d]), 4),
            })

    print(f"[Update] Got {len(results)} new daily price records")
    return results


def append_to_csv(new_records: list[dict], csv_path: str):
    """Append new price records to the CSV, keeping existing columns."""
    if not new_records:
        return 0

    existing = pd.read_csv(csv_path)

    new_rows = []
    for rec in new_records:
        d = rec["date"]
        new_rows.append({
            "Price hub": "Indiana Hub LMP Daily",
            "Trade date": d.isoformat(),
            "Delivery start date": d.isoformat(),
            "Delivery \nend date": d.isoformat(),
            "High price $/MWh": "",
            "Low price $/MWh": "",
            "Wtd avg price $/MWh": rec["price"],
            "Change": "",
            "Daily volume MWh": "",
            "Number of trades": "",
            "Number of counterparties": "",
        })

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing, new_df], ignore_index=True)

    # Backup before overwrite
    backup = csv_path + ".bak"
    existing.to_csv(backup, index=False)

    combined.to_csv(csv_path, index=False)
    print(f"[Update] Appended {len(new_rows)} rows to CSV (backup at {backup})")
    return len(new_rows)


def main():
    parser = argparse.ArgumentParser(description="Update MISO data and retrain LSTM")
    parser.add_argument("--cron", action="store_true", help="Quiet mode for cron jobs")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data fetch, only retrain")
    args = parser.parse_args()

    csv_path = get_default_historical_csv_path()
    quiet = args.cron

    # 1. Fetch new data
    if not args.skip_fetch:
        last_date = get_last_csv_delivery_date(csv_path)
        if last_date is None:
            print("[Update] No CSV data found, cannot determine fetch range")
            return 1

        new_records = fetch_new_prices_since(last_date)
        append_to_csv(new_records, csv_path)

    # 2. Retrain LSTM
    if not quiet:
        print("[Update] Retraining LSTM model...")
    try:
        train_model(csv_path)
        if not quiet:
            print("[Update] Model updated successfully")
    except Exception as e:
        msg = f"[Update] Training failed: {e}"
        if quiet:
            print(msg, file=sys.stderr)
        else:
            print(msg)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
