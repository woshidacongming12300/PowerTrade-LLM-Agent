"""
EIA (U.S. Energy Information Administration) daily wholesale electricity price fetcher.

Uses the EIA API v2 — accessible from mainland China (api.eia.gov is NOT blocked).

Pre-requisites:
    1. Register for a free API key at: https://www.eia.gov/opendata/register.php
    2. Set environment variable: EIA_API_KEY=<your_key>
    3. Or set in .env file

Data routes (discovered via API route tree):
    - electricity/wholesale-markets  →  wholesale electricity & natural gas market data
    - electricity/rto/region-data    →  RTO region-level hourly/daily data
    - electricity/rto/fuel-type-data →  RTO fuel-type generation data

The EIA wholesale data is published biweekly (every ~2 weeks), NOT daily.
For daily data, this module uses the RTO region data as a proxy.
"""
from __future__ import annotations

import os
import sys
from datetime import date, timedelta
from typing import List, Optional

import requests

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

EIA_BASE = "https://api.eia.gov/v2"


def _get_api_key() -> str:
    """Get EIA API key from environment or .env file."""
    key = os.environ.get("EIA_API_KEY", "").strip()
    if key:
        return key

    # Try loading from .env
    env_file = os.path.join(_project_root, ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("EIA_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    os.environ["EIA_API_KEY"] = val
                    return val
    return ""


def _eia_get(route: str, params: dict, timeout: int = 30) -> Optional[dict]:
    """Make an EIA API v2 GET request. Returns parsed JSON or None on failure."""
    api_key = _get_api_key()
    if not api_key:
        print("[EIA] No API key configured. Set EIA_API_KEY in .env")
        return None

    url = f"{EIA_BASE}/{route}/data/"
    params = {**params, "api_key": api_key}

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"[EIA] HTTP {resp.status_code} for {route}: {resp.text[:200]}")
            return None
    except Exception as e:
        print(f"[EIA] Request failed for {route}: {e}")
        return None


def fetch_wholesale_prices_daily(
    start_date: date,
    end_date: date,
) -> List[dict]:
    """
    Fetch daily wholesale electricity prices from EIA.

    Tries multiple routes in order:
    1. electricity/wholesale-markets — wholesale hub prices (ICE data)
    2. electricity/rto/region-data — RTO regional data

    Returns list of {"date": date, "price": float} sorted chronologically.
    """
    if not _get_api_key():
        return []

    records: List[dict] = []

    # --- Route 1: Wholesale Markets ---
    print(f"[EIA] Trying wholesale-markets route ({start_date} to {end_date})...")
    data = _eia_get(
        "electricity/wholesale-markets",
        {
            "frequency": "daily",
            "data[0]": "price",
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000,
        },
    )

    if data and data.get("response", {}).get("data"):
        raw = data["response"]["data"]
        records = _parse_eia_response(raw, start_date, end_date)
        if records:
            print(f"[EIA] wholesale-markets: {len(records)} records")
            return records
        else:
            print("[EIA] wholesale-markets returned data but no parseable records")

    # --- Route 2: RTO Region Data ---
    print(f"[EIA] Trying rto/region-data route...")
    data = _eia_get(
        "electricity/rto/region-data",
        {
            "frequency": "daily",
            "data[0]": "price",
            "facets[respondent][]": "MISO",
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000,
        },
    )

    if data and data.get("response", {}).get("data"):
        raw = data["response"]["data"]
        records = _parse_eia_response(raw, start_date, end_date)
        if records:
            print(f"[EIA] rto/region-data: {len(records)} records")
            return records

    # --- Route 3: Try without facets ---
    print(f"[EIA] Trying rto/region-data without MISO filter...")
    data = _eia_get(
        "electricity/rto/region-data",
        {
            "frequency": "daily",
            "data[0]": "price",
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": 5000,
        },
    )

    if data and data.get("response", {}).get("data"):
        raw = data["response"]["data"]
        records = _parse_eia_response(raw, start_date, end_date)
        if records:
            print(f"[EIA] rto/region-data (all): {len(records)} records")
            return records

    print("[EIA] No data found from any route")
    return []


def _parse_eia_response(
    raw: list,
    start_date: date,
    end_date: date,
) -> List[dict]:
    """Parse EIA API v2 response into standardized price records."""
    records = []
    for item in raw:
        period = item.get("period", "")
        price_str = item.get("price", "") or item.get("value", "")

        if not period or not price_str:
            continue

        try:
            d = date.fromisoformat(period[:10])
        except (ValueError, TypeError):
            continue

        if not (start_date <= d <= end_date):
            continue

        try:
            price = float(price_str)
        except (ValueError, TypeError):
            continue

        records.append({"date": d, "price": round(price, 4)})

    # Deduplicate by date (keep first)
    seen = set()
    deduped = []
    for r in records:
        if r["date"] not in seen:
            seen.add(r["date"])
            deduped.append(r)

    return sorted(deduped, key=lambda x: x["date"])


def get_latest_price() -> Optional[float]:
    """Quick check: get the most recent available wholesale price."""
    if not _get_api_key():
        return None

    data = _eia_get(
        "electricity/wholesale-markets",
        {
            "frequency": "daily",
            "data[0]": "price",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 5,
        },
    )

    if data and data.get("response", {}).get("data"):
        items = data["response"]["data"]
        if items:
            price = items[0].get("price", "") or items[0].get("value", "")
            period = items[0].get("period", "")
            try:
                return round(float(price), 4)
            except (ValueError, TypeError):
                pass
    return None


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    if not _get_api_key():
        print("EIA_API_KEY not configured. Register at https://www.eia.gov/opendata/register.php")
        print("Then add to .env: EIA_API_KEY=<your_key>")
        sys.exit(1)

    print(f"[EIA] Testing connection...")
    latest = get_latest_price()
    if latest:
        print(f"  Latest wholesale price: ${latest}/MWh")

    print(f"\n[EIA] Fetching last 7 days of data...")
    today = date.today()
    records = fetch_wholesale_prices_daily(today - timedelta(days=14), today)
    for r in records:
        print(f"  {r['date']}: ${r['price']}/MWh")
    print(f"\nTotal: {len(records)} records")
