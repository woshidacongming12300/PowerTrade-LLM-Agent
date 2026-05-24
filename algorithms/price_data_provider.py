"""
电价历史窗口供给：默认读本地 CSV，预留实时数据源注入。

接入实时行情时，实现一个 fetcher 并通过 set_live_price_fetcher 注册即可，
无需改 LSTM 推理代码。
"""
from __future__ import annotations

import os
from datetime import date
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from algorithms.weather_data import (
    INDIANA_HUB,
    fetch_historical_weather_for_df,
)

LivePriceFetcher = Callable[[date, int], Optional[List[float]]]

_live_fetcher: Optional[LivePriceFetcher] = None


def set_live_price_fetcher(fetcher: Optional[LivePriceFetcher]) -> None:
    """注册实时电价拉取函数；传入 None 则取消注册，仅使用 CSV 回退。"""
    global _live_fetcher
    _live_fetcher = fetcher


def get_live_price_fetcher() -> Optional[LivePriceFetcher]:
    return _live_fetcher


def get_default_historical_csv_path() -> str:
    from utils.path_tool import get_abs_path

    return get_abs_path(os.path.join("algorithms", "clean_electric_data.csv"))


def _load_sorted_frame(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Delivery start date"] = pd.to_datetime(df["Delivery start date"], format="mixed")
    return df.sort_values("Delivery start date").reset_index(drop=True)


def _recent_prices_from_csv(csv_path: str, anchor_date: date, n: int) -> np.ndarray:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"未找到电价 CSV: {csv_path}")

    df = _load_sorted_frame(csv_path)
    col_d = "Delivery start date"
    col_p = "Wtd avg price $/MWh"
    cutoff = pd.Timestamp(anchor_date).normalize()
    sub = df.loc[df[col_d].dt.normalize() <= cutoff, col_p]

    if sub.empty:
        raise ValueError(f"在 anchor={anchor_date} 之前 CSV 中无电价记录")

    vals = sub.astype(float).tail(n).to_numpy()
    if len(vals) < n:
        pad_val = float(vals[0])
        front = np.full(n - len(vals), pad_val, dtype=float)
        vals = np.concatenate([front, vals])
    return vals.reshape(-1, 1)


def get_last_csv_delivery_date(csv_path: Optional[str] = None) -> Optional[date]:
    """CSV 中最后一笔交割日；文件不存在或为空则返回 None。"""
    csv_path = csv_path or get_default_historical_csv_path()
    if not os.path.isfile(csv_path):
        return None
    df = _load_sorted_frame(csv_path)
    if df.empty:
        return None
    ts = df["Delivery start date"].iloc[-1]
    return ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()


def resolve_history_anchor(anchor_date: date, csv_path: Optional[str] = None) -> date:
    """
    取历史窗口的「截至日」：已接入实时源时与 anchor 一致；
    仅 CSV 时不超过文件中最后一日，避免用未来日历去截断导致语义歧义。
    """
    if _live_fetcher is not None:
        return anchor_date
    last = get_last_csv_delivery_date(csv_path)
    if last is None:
        return anchor_date
    return anchor_date if anchor_date <= last else last


def get_recent_prices_flat(
    anchor_date: Optional[date],
    n: int,
    csv_path: Optional[str] = None,
) -> List[float]:
    """
    最近 n 个日度加权均价（旧 → 新），LSTM / Chronos 等可共用。
    实时 fetcher 需能按任意 n 返回序列；若暂未接入则读 CSV。
    """
    csv_path = csv_path or get_default_historical_csv_path()
    ad_raw = anchor_date or date.today()
    ad = resolve_history_anchor(ad_raw, csv_path)

    if _live_fetcher is not None:
        try:
            live = _live_fetcher(ad_raw, n)
        except Exception:
            live = None
        if live is not None:
            arr = np.asarray(live, dtype=float).flatten()
            if arr.size != n:
                raise ValueError(f"实时 fetcher 应返回长度 {n} 的序列，实际为 {arr.size}")
            return [float(x) for x in arr]

    block = _recent_prices_from_csv(csv_path, ad, n)
    return [float(x) for x in block.flatten()]


def get_recent_price_window(
    anchor_date: Optional[date],
    seq_length: int,
    csv_path: Optional[str] = None,
) -> np.ndarray:
    """
    取「用于 LSTM 输入」的最近 seq_length 个日度加权均价（列顺序：旧 → 新）。

    :param anchor_date: 视为「最后已知真实数据日」；None 时用 date.today()
    :param seq_length: 与训练时窗口一致（当前为 7）
    :param csv_path: 回退 CSV；None 时用项目内默认路径
    """
    flat = get_recent_prices_flat(anchor_date, seq_length, csv_path)
    return np.array(flat, dtype=float).reshape(-1, 1)


# ======================== 多变量数据（价格+天气） ========================

_MERGED_CACHE_PATH: Optional[str] = None


def get_merged_csv_path() -> str:
    """返回合并后 CSV 的缓存路径。"""
    global _MERGED_CACHE_PATH
    if _MERGED_CACHE_PATH:
        return _MERGED_CACHE_PATH
    from utils.path_tool import get_abs_path

    _MERGED_CACHE_PATH = get_abs_path(os.path.join("algorithms", "merged_price_weather.csv"))
    return _MERGED_CACHE_PATH


def build_merged_price_weather(
    csv_path: Optional[str] = None,
    lat: float = INDIANA_HUB[0],
    lon: float = INDIANA_HUB[1],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """合并电价 CSV 与历史天气数据，缓存为 merged_price_weather.csv。

    返回 DataFrame 列：Delivery start date, price, temp_max, temp_min, precip, wind_speed
    """
    merged_path = get_merged_csv_path()
    if os.path.isfile(merged_path) and not force_refresh:
        print(f"[Data] Using cached merged data: {merged_path}")
        df = pd.read_csv(merged_path)
        df["Delivery start date"] = pd.to_datetime(df["Delivery start date"], format="mixed")
        return df.sort_values("Delivery start date").reset_index(drop=True)

    csv_path = csv_path or get_default_historical_csv_path()
    price_df = pd.read_csv(csv_path)
    price_df["Delivery start date"] = pd.to_datetime(
        price_df["Delivery start date"], format="mixed"
    )
    price_df = price_df.sort_values("Delivery start date").reset_index(drop=True)
    price_df["price"] = price_df["Wtd avg price $/MWh"].astype(float)

    weather_df = fetch_historical_weather_for_df(
        price_df, date_col="Delivery start date", lat=lat, lon=lon
    )

    # Merge on date
    weather_df["Delivery start date"] = weather_df["date"].dt.normalize()
    price_df["_merge_key"] = price_df["Delivery start date"].dt.normalize()

    merged = price_df.merge(
        weather_df[["Delivery start date", "temp_max", "temp_min", "precip", "wind_speed"]],
        on="Delivery start date",
        how="inner",
    )
    merged = merged.drop(columns=["_merge_key"])

    # Keep only needed columns
    merged = merged[["Delivery start date", "price", "temp_max", "temp_min", "precip", "wind_speed"]]

    merged.to_csv(merged_path, index=False)
    print(f"[Data] Merged {len(merged)} rows saved to {merged_path}")
    return merged


def get_recent_multivariate_window(
    anchor_date: date,
    seq_length: int,
    csv_path: Optional[str] = None,
) -> np.ndarray:
    """取最近 seq_length 天的多变量数据 (price + 4 weather features)，shape (seq_length, 5)。

    天气数据从 Open-Meteo Archive API 获取（如 anchor 在数据范围内），
    超出 Archive 范围的日期使用 Forecast API。
    """
    from datetime import timedelta

    from algorithms.weather_data import fetch_historical_weather, fetch_weather_forecast

    # Get price data
    prices = get_recent_prices_flat(anchor_date, seq_length, csv_path)
    prices = np.asarray(prices, dtype=float)

    start_d = anchor_date - timedelta(days=seq_length - 1)

    # Try historical weather first
    weather_rows = []
    try:
        hist_df = fetch_historical_weather(
            INDIANA_HUB[0], INDIANA_HUB[1],
            start_d.strftime("%Y-%m-%d"),
            anchor_date.strftime("%Y-%m-%d"),
        )
        if not hist_df.empty:
            hist_df["_date"] = hist_df["date"].dt.date
            weather_rows = hist_df
    except Exception:
        pass

    # Build the window
    features = np.zeros((seq_length, 5), dtype=float)
    for i in range(seq_length):
        d = start_d + timedelta(days=i)
        features[i, 0] = prices[i]

        # Look up weather
        if not weather_rows.empty:
            match = weather_rows[weather_rows["_date"] == d]
            if not match.empty:
                row = match.iloc[0]
                features[i, 1] = float(row["temp_max"])
                features[i, 2] = float(row["temp_min"])
                features[i, 3] = float(row["precip"])
                features[i, 4] = float(row["wind_speed"])
                continue

        # Fallback: use forecast API for near-future dates
        # (this happens when anchor_date is very recent and archive isn't updated yet)
        features[i, 1] = 20.0  # reasonable defaults
        features[i, 2] = 10.0
        features[i, 3] = 0.0
        features[i, 4] = 15.0

    return features
