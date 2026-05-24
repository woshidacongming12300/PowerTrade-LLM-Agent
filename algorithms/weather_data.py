"""Weather data layer: historical archive + forecast from Open-Meteo (free, no API key).

Used by:
- Training pipeline: fetch_historical_weather() → merge with price CSV
- Inference pipeline: fetch_weather_forecast() → feed as exogenous features to LSTM
- Agent tools: get_texas_weather() delegates to fetch_weather_forecast()
"""

import time
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests

# Hub coordinates
INDIANA_HUB = (39.77, -86.16)  # Indianapolis, IN (MISO)
TEXAS_HOUSTON = (29.76, -95.36)  # Houston, TX (ERCOT)

_DAILY_WEATHER_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max"
_COLUMN_MAP = {
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "precipitation_sum": "precip",
    "windspeed_10m_max": "wind_speed",
}


def _default_lat_lon(region: Optional[str] = None) -> tuple[float, float, str]:
    """Resolve region name to (lat, lon, label). Defaults to Indiana Hub."""
    if region and ("texas" in region.lower() or "houston" in region.lower() or "ercot" in region.lower()):
        return TEXAS_HOUSTON[0], TEXAS_HOUSTON[1], "Texas Houston (ERCOT)"
    return INDIANA_HUB[0], INDIANA_HUB[1], "Indiana Indianapolis (MISO)"


def fetch_historical_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily historical weather from Open-Meteo Archive API.

    Returns DataFrame with columns: date, temp_max, temp_min, precip, wind_speed
    """
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily={_DAILY_WEATHER_VARS}"
        f"&timezone=auto"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    if not daily:
        return pd.DataFrame()

    df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
    for api_name, local_name in _COLUMN_MAP.items():
        if api_name in daily:
            df[local_name] = daily[api_name]

    return df


def fetch_weather_forecast(
    lat: float,
    lon: float,
    forecast_days: int = 7,
) -> dict:
    """Fetch daily weather forecast from Open-Meteo Forecast API.

    Returns dict with keys: dates (list[str]), temp_max, temp_min, precip, wind_speed
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily={_DAILY_WEATHER_VARS}"
        f"&timezone=auto&forecast_days={forecast_days}"
    )
    resp = requests.get(url, timeout=8)
    resp.raise_for_status()
    data = resp.json()

    daily = data.get("daily", {})
    return {
        "dates": daily.get("time", []),
        "temp_max": daily.get("temperature_2m_max", []),
        "temp_min": daily.get("temperature_2m_min", []),
        "precip": daily.get("precipitation_sum", []),
        "wind_speed": daily.get("windspeed_10m_max", []),
    }


def fetch_historical_weather_for_df(
    price_df: pd.DataFrame,
    date_col: str = "Delivery start date",
    lat: float = INDIANA_HUB[0],
    lon: float = INDIANA_HUB[1],
) -> pd.DataFrame:
    """Fetch historical weather covering the date range of a price DataFrame.

    Returns a weather DataFrame indexed by date, ready for merging.
    """
    dates = pd.to_datetime(price_df[date_col])
    start = dates.min().strftime("%Y-%m-%d")
    end = dates.max().strftime("%Y-%m-%d")

    print(f"[Weather] Fetching historical weather {start} → {end} ...")
    weather_df = fetch_historical_weather(lat, lon, start, end)

    if weather_df.empty:
        raise RuntimeError(f"No weather data returned for {start} → {end}")

    print(f"[Weather] Got {len(weather_df)} days of historical weather")
    return weather_df


if __name__ == "__main__":
    # Quick smoke test
    print("=== Historical weather test (last 30 days) ===")
    end = date.today()
    start = end.replace(day=1)  # start of current month
    df = fetch_historical_weather(
        INDIANA_HUB[0], INDIANA_HUB[1],
        start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"),
    )
    print(df.head())
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")

    print("\n=== Forecast test (7 days) ===")
    fc = fetch_weather_forecast(INDIANA_HUB[0], INDIANA_HUB[1], forecast_days=7)
    for i, d in enumerate(fc["dates"]):
        print(
            f"  {d}: {fc['temp_min'][i]}~{fc['temp_max'][i]}°C, "
            f"wind {fc['wind_speed'][i]} km/h, rain {fc['precip'][i]}mm"
        )

    print("\n[OK] Weather data layer works")
