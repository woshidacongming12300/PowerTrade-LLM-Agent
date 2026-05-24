"""
通过 gridstatus 拉取 MISO 节点电价，聚合成「日度均价」序列，供 LSTM / Chronos 使用。

与训练 CSV「Indiana Hub」口径接近：使用官方 Indiana Hub 的逐小时 LMP 日简单平均。
（若 CSV 为「RT Peak」等细分产品，数值可能与全时段均值略有差异。）

环境变量（可选）：
  GRIDSTATUS_ENABLED   设为 1 / true / yes 时，在应用加载 agent_tools 时自动注册实时 fetcher
  GRIDSTATUS_HUB       默认 INDIANA.HUB（gridstatus 常量名）
  GRIDSTATUS_MARKET    默认 REAL_TIME_HOURLY_FINAL；可改为 DAY_AHEAD_HOURLY 等 Markets 枚举名
  GRIDSTATUS_TIMEZONE  聚合到日历日时使用的时区，默认 America/Chicago（MISO）
"""
from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import List, Optional

import pandas as pd
from zoneinfo import ZoneInfo

# Suppress noisy gridstatus/urllib3 download logs
logging.getLogger("gridstatus").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def _market_from_env():
    from gridstatus import Markets

    name = os.environ.get("GRIDSTATUS_MARKET", "REAL_TIME_HOURLY_FINAL").strip().upper()
    if not hasattr(Markets, name):
        raise ValueError(
            f"未知 GRIDSTATUS_MARKET={name!r}，应为 gridstatus.Markets 上的枚举名，"
            f"例如 REAL_TIME_HOURLY_FINAL、DAY_AHEAD_HOURLY"
        )
    return getattr(Markets, name)


def _hub_locations() -> List[str]:
    hub = os.environ.get("GRIDSTATUS_HUB", "INDIANA.HUB").strip()
    return [hub]


def _tz() -> ZoneInfo:
    key = os.environ.get("GRIDSTATUS_TIMEZONE", "America/Chicago").strip()
    return ZoneInfo(key)


def fetch_miso_hub_daily_lmp_sequence(
    anchor_date: date, n: int, max_lookback: int = 14
) -> Optional[List[float]]:
    """
    Fetch daily average LMP for [anchor_date - (n-1), anchor_date] (old -> new).
    MISO RT Final data has ~7-10 day publication delay. If the requested range
    returns 404, we progressively shift back by up to max_lookback days.

    Returns None when no data is available even after lookback, triggering CSV fallback.
    """
    if n < 1:
        return None
    try:
        from gridstatus import MISO
    except ImportError:
        return None

    market = _market_from_env()
    tz = _tz()
    m = MISO()

    for offset in range(0, max_lookback + 1):
        shifted_anchor = anchor_date - timedelta(days=offset)
        start_d = shifted_anchor - timedelta(days=n - 1)

        try:
            df = m.get_lmp(
                pd.Timestamp(start_d),
                end=pd.Timestamp(shifted_anchor) + pd.Timedelta(days=1),
                market=market,
                locations=_hub_locations(),
            )
        except Exception:
            continue

        if df is None or df.empty:
            continue
        if "LMP" not in df.columns or "Interval Start" not in df.columns:
            continue

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

        out: List[float] = []
        all_present = True
        for i in range(n):
            d = start_d + timedelta(days=i)
            if d not in daily.index:
                all_present = False
                break
            out.append(round(float(daily.loc[d]), 4))

        if all_present and out:
            if offset > 0:
                print(
                    f"[Gridstatus] Data shifted by {offset} days "
                    f"(MISO RT Final publish delay). Range: {start_d} to {shifted_anchor}"
                )
            return out

    return None


def register_gridstatus_fetcher_from_env() -> None:
    """注册到 price_data_provider，由 agent_tools 在 GRIDSTATUS_ENABLED 时调用。"""
    from algorithms.price_data_provider import set_live_price_fetcher

    set_live_price_fetcher(fetch_miso_hub_daily_lmp_sequence)
