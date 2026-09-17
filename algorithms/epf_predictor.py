import os
import re
import sys
import warnings
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from algorithms.price_data_provider import (
    build_merged_price_weather,
    get_default_historical_csv_path,
    get_recent_price_window,
    get_recent_multivariate_window,
)
from algorithms.weather_data import fetch_weather_forecast, INDIANA_HUB

warnings.filterwarnings("ignore")

SEQ_LENGTH = 7
MAX_ROLL_STEPS = 30
N_FEATURES = 5  # price, temp_max, temp_min, precip, wind_speed
FEATURE_NAMES = ["price", "temp_max", "temp_min", "precip", "wind_speed"]

# Historical MAE from single-variable model evaluation (base confidence band)
_HISTORICAL_MAE = 5.30
_HISTORICAL_PRICE_STD = 6.29


class ElectricPriceLSTM(nn.Module):
    def __init__(self, input_size=N_FEATURES, hidden_size=64, num_layers=2, output_size=1):
        super(ElectricPriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_model(csv_path: Optional[str] = None, seq_length: int = SEQ_LENGTH):
    csv_path = csv_path or get_default_historical_csv_path()
    print("[LSTM] Building merged price + weather dataset...")
    merged = build_merged_price_weather(csv_path, force_refresh=True)

    feature_cols = ["price", "temp_max", "temp_min", "precip", "wind_speed"]
    data = merged[feature_cols].values.astype(float)

    # Normalize each feature independently
    scalers = []
    scaled_data = np.zeros_like(data)
    for i, col in enumerate(feature_cols):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data[:, i] = scaler.fit_transform(data[:, i].reshape(-1, 1)).flatten()
        scalers.append(scaler)

    # Build sequences
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i : i + seq_length])  # (seq_length, N_FEATURES)
        y.append(scaled_data[i + seq_length, 0])   # predict next day's *price* only

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    model = ElectricPriceLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    print(f"[LSTM] Training multivariate model on {len(X)} samples ({len(merged)} days)...")
    epochs = 200
    best_loss = float("inf")
    patience = 20
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        if current_loss < best_loss - 1e-5:
            best_loss = current_loss
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {current_loss:.6f}")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}, best loss: {best_loss:.6f}")
            break

    model_save_path = os.path.join(os.path.dirname(__file__), "lstm_epf_multivariate.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scalers_data_min": np.array([s.data_min_[0] for s in scalers], dtype=np.float32),
            "scalers_data_max": np.array([s.data_max_[0] for s in scalers], dtype=np.float32),
            "n_features": N_FEATURES,
            "feature_names": FEATURE_NAMES,
        },
        model_save_path,
    )
    print(f"[LSTM] Training complete, model saved to {model_save_path}")

    # Compute training MAE for confidence bands
    model.eval()
    with torch.no_grad():
        preds = model(X).numpy().flatten()
    actuals = y.numpy().flatten()
    # Denormalize to real price scale
    price_scaler = scalers[0]
    preds_real = preds * (price_scaler.data_max_[0] - price_scaler.data_min_[0]) + price_scaler.data_min_[0]
    actuals_real = actuals * (price_scaler.data_max_[0] - price_scaler.data_min_[0]) + price_scaler.data_min_[0]
    train_mae = np.mean(np.abs(preds_real - actuals_real))
    print(f"[LSTM] Training MAE (denormalized): {train_mae:.2f} $/MWh")


def _parse_iso_date(s: str) -> Optional[date]:
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _cn_number(s: str) -> int:
    """Convert a small Chinese numeral (一二三…十) or digits string to int."""
    s = s.strip()
    if s.isdigit():
        return int(s)
    mapping = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
               "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if s in mapping:
        return mapping[s]
    if s.startswith("十"):
        return 10 + mapping.get(s[1:], 0)
    if s.endswith("十"):
        return mapping.get(s[0], 1) * 10
    return 1


def parse_forecast_end_date(target_date: str, forecast_anchor: date) -> date:
    """
    将自然语言或 ISO 日期解析为「预测窗口的最后一日」（含当日）。
    相对语义（明天/后天/N天后）均相对 forecast_anchor。
    """
    raw = target_date.strip()
    if not raw:
        return forecast_anchor + timedelta(days=1)

    iso = _parse_iso_date(raw)
    if iso is not None:
        return iso

    if "大后天" in raw or "3 days later" in raw.lower():
        return forecast_anchor + timedelta(days=3)
    if "后天" in raw or "day after tomorrow" in raw.lower():
        return forecast_anchor + timedelta(days=2)
    if "明天" in raw or "tomorrow" in raw.lower():
        return forecast_anchor + timedelta(days=1)
    if "today" in raw.lower() or "今天" in raw:
        return forecast_anchor

    # Week-based semantics: 未来一周 / 一周后 / 下周 / 一星期 / next week / a week later
    if re.search(r"(未来一周|一周[后後]?|下周|下星期|一星期[后後]?|下个?礼拜|next\s*week|a\s*week\s*later|one\s*week)", raw.lower()):
        return forecast_anchor + timedelta(days=7)
    # Month-based semantics: N个月后 / 下个月 / next month
    m_month = re.search(r"([一二两三四五六七八九十0-9]+)\s*个?月[后後]|next\s*month", raw.lower())
    if m_month:
        n = _cn_number(m_month.group(1)) if m_month.group(1) else 1
        return forecast_anchor + timedelta(days=30 * n)
    if re.search(r"下个月", raw):
        return forecast_anchor + timedelta(days=30)

    m = re.search(r"(\d+)\s*天[后後]|(\d+)\s*days?\s*(later|from now)", raw)
    if m:
        n = int(m.group(1) or m.group(2))
        return forecast_anchor + timedelta(days=n)

    raise ValueError(
        f"无法解析目标日期: {target_date!r}，请使用 YYYY-MM-DD、明天/tomorrow、后天、N天后、未来一周 等形式"
    )


def forecast_steps(forecast_anchor: date, end_date: date) -> int:
    steps = (end_date - forecast_anchor).days
    if steps < 1:
        return 1
    return min(steps, MAX_ROLL_STEPS)


def _load_lstm_checkpoint(model_path: str):
    checkpoint = torch.load(model_path, weights_only=False)
    return (
        checkpoint["scalers_data_min"],
        checkpoint["scalers_data_max"],
        checkpoint["model_state_dict"],
    )


def _roll_lstm(
    model: ElectricPriceLSTM,
    scaler_min: np.ndarray,
    scaler_max: np.ndarray,
    initial_window: np.ndarray,   # (seq_length, N_FEATURES) real-scale
    weather_forecast: np.ndarray,  # (steps, N_FEATURES-1) real-scale weather for future days
    steps: int,
) -> List[float]:
    """Autoregressive rolling prediction with exogenous weather features.

    At each step, the most recent seq_length rows form the input.
    Price column is rolled with predictions; weather columns use the forecast.
    """
    scaler_range = scaler_max - scaler_min
    scaler_range[scaler_range == 0] = 1.0  # avoid div-by-zero for constant features

    window = initial_window.copy().tolist()  # list of [price, tmax, tmin, precip, wind]
    predictions: List[float] = []

    for i in range(steps):
        # Scale window
        arr = np.array(window[-SEQ_LENGTH:], dtype=float)
        scaled = (arr - scaler_min) / scaler_range
        tensor_input = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)  # (1, seq, 5)

        with torch.no_grad():
            pred_scaled = model(tensor_input).item()

        real_price = float(pred_scaled * scaler_range[0] + scaler_min[0])
        predictions.append(round(real_price, 4))

        # Build next row: predicted price + forecast weather
        if i < len(weather_forecast):
            fc_weather = weather_forecast[i].tolist()
        else:
            fc_weather = [scaler_min[j + 1] for j in range(N_FEATURES - 1)]

        next_row = [real_price] + fc_weather
        window.append(next_row)

    return predictions


def predict_future_prices(
    target_date: str,
    *,
    forecast_anchor: Optional[date] = None,
    csv_path: Optional[str] = None,
    with_confidence: bool = False,
) -> List[Dict[str, Any]]:
    """
    多变量自回归滚动预测：从 forecast_anchor 的下一日起，直至 target_date（含）的每日电价。

    内部自动拉取天气预报作为外生特征；返回 [{"forecast_date": "...", "price": 48.2, ...}, ...]
    当 with_confidence=True 时额外返回 lower/upper/confidence 字段。
    """
    model_path = os.path.join(os.path.dirname(__file__), "lstm_epf_multivariate.pth")
    if not os.path.exists(model_path):
        # Fallback to old univariate model
        model_path = os.path.join(os.path.dirname(__file__), "lstm_epf.pth")
        if not os.path.exists(model_path):
            return []
        return _predict_univariate_fallback(target_date, forecast_anchor, csv_path, model_path, with_confidence)

    scaler_min, scaler_max, state = _load_lstm_checkpoint(model_path)
    model = ElectricPriceLSTM()
    model.load_state_dict(state)
    model.eval()

    fa = forecast_anchor or date.today()
    end_d = parse_forecast_end_date(target_date, fa)
    steps = forecast_steps(fa, end_d)

    # Build initial multivariate window
    print(f"   >> [LSTM] Building multivariate window (anchor {fa})...")
    window = get_recent_multivariate_window(fa, SEQ_LENGTH, csv_path=csv_path)

    # Fetch weather forecast for future days (with fallback for network issues)
    print(f"   >> [LSTM] Fetching {steps}-day weather forecast...")
    weather_forecast = np.full((steps, N_FEATURES - 1), fill_value=np.nan, dtype=float)
    try:
        fc = fetch_weather_forecast(INDIANA_HUB[0], INDIANA_HUB[1], forecast_days=steps)
        for i in range(min(steps, len(fc["dates"]))):
            weather_forecast[i, 0] = fc["temp_max"][i] if i < len(fc["temp_max"]) else 20.0
            weather_forecast[i, 1] = fc["temp_min"][i] if i < len(fc["temp_min"]) else 10.0
            weather_forecast[i, 2] = fc["precip"][i] if i < len(fc["precip"]) else 0.0
            weather_forecast[i, 3] = fc["wind_speed"][i] if i < len(fc["wind_speed"]) else 15.0
    except Exception as e:
        print(f"   >> [LSTM] Weather API unavailable ({e}), using seasonal defaults")

    # Fill any missing weather values with seasonal defaults (Indiana spring/fall ~15-20°C)
    defaults = [20.0, 10.0, 0.0, 15.0]  # temp_max, temp_min, precip, wind
    for j in range(N_FEATURES - 1):
        weather_forecast[:, j] = np.nan_to_num(weather_forecast[:, j], nan=defaults[j])

    print(f"   >> [LSTM] Auto-regressive rolling {steps} steps (anchor {fa}, target {end_d})...")
    prices = _roll_lstm(model, scaler_min, scaler_max, window, weather_forecast, steps)

    out: List[Dict[str, Any]] = []
    for i, p in enumerate(prices, start=1):
        d = fa + timedelta(days=i)
        entry: Dict[str, Any] = {"forecast_date": d.isoformat(), "price": round(p, 2)}

        if with_confidence:
            entry["lower"] = round(p - _HISTORICAL_MAE, 2)
            entry["upper"] = round(p + _HISTORICAL_MAE, 2)
            entry["confidence"] = "medium"

        out.append(entry)
    return out


def _predict_univariate_fallback(
    target_date: str,
    forecast_anchor: Optional[date],
    csv_path: Optional[str],
    model_path: str,
    with_confidence: bool,
) -> List[Dict[str, Any]]:
    """Fallback: use old univariate model if multivariate not trained yet."""
    checkpoint = torch.load(model_path, weights_only=False)
    scaler_min = float(checkpoint["scaler_min"])
    scaler_max = float(checkpoint["scaler_max"])
    state = checkpoint["model_state_dict"]

    model = ElectricPriceLSTM(input_size=1)
    model.load_state_dict(state)
    model.eval()

    fa = forecast_anchor or date.today()
    end_d = parse_forecast_end_date(target_date, fa)
    steps = forecast_steps(fa, end_d)

    last_7 = get_recent_price_window(fa, SEQ_LENGTH, csv_path=csv_path)
    prices = _roll_lstm_univariate(model, scaler_min, scaler_max, last_7, steps)

    out: List[Dict[str, Any]] = []
    for i, p in enumerate(prices, start=1):
        d = fa + timedelta(days=i)
        entry: Dict[str, Any] = {"forecast_date": d.isoformat(), "price": round(float(p), 2)}
        if with_confidence:
            entry["lower"] = round(float(p) - _HISTORICAL_MAE, 2)
            entry["upper"] = round(float(p) + _HISTORICAL_MAE, 2)
            entry["confidence"] = "low"
        out.append(entry)
    return out


def _roll_lstm_univariate(
    model, scaler_min: float, scaler_max: float,
    last_n_prices: np.ndarray, steps: int,
) -> List[float]:
    """Old univariate rolling prediction (preserved for fallback)."""
    scaled_input = (last_n_prices - scaler_min) / (scaler_max - scaler_min)
    current_sequence = scaled_input.flatten().tolist()
    predictions: List[float] = []
    for _ in range(steps):
        tensor_input = torch.tensor(current_sequence, dtype=torch.float32).view(1, SEQ_LENGTH, 1)
        with torch.no_grad():
            pred_scaled = model(tensor_input).item()
        real_price = pred_scaled * (scaler_max - scaler_min) + scaler_min
        predictions.append(round(float(real_price), 4))
        current_sequence.append(pred_scaled)
        current_sequence.pop(0)
    return predictions


def predict_future_price(
    target_date: str,
    *,
    forecast_anchor: Optional[date] = None,
    csv_path: Optional[str] = None,
) -> float:
    """兼容旧接口：返回预测序列的最后一日电价。"""
    series = predict_future_prices(
        target_date, forecast_anchor=forecast_anchor, csv_path=csv_path
    )
    if not series:
        return 0.0
    return float(series[-1]["price"])


def compute_confidence_band(
    lstm_price: float,
    chronos_price: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute confidence band around an LSTM prediction.

    Uses historical MAE as the baseline band, and tightens/widens based on
    agreement between LSTM and Chronos-T5 cross-validation.
    """
    band = _HISTORICAL_MAE  # base ~68% confidence band
    confidence = "medium"

    if chronos_price is not None:
        delta = abs(lstm_price - chronos_price)
        if delta < 3:
            confidence = "high"
            band = _HISTORICAL_MAE * 0.7
        elif delta < 7:
            confidence = "medium"
            band = _HISTORICAL_MAE
        else:
            confidence = "low"
            band = _HISTORICAL_MAE * 1.5

    return {
        "price": round(lstm_price, 2),
        "lower": round(lstm_price - band, 2),
        "upper": round(lstm_price + band, 2),
        "confidence": confidence,
        "band_width": round(band, 2),
    }


if __name__ == "__main__":
    train_model()
    print("\n[LSTM] Testing multivariate inference:")
    test_price = predict_future_price("明天")
    print(f"Predicted tomorrow weighted avg price: {test_price} $/MWh")

    print("\n[LSTM] Testing with confidence bands:")
    series = predict_future_prices("3天后", with_confidence=True)
    for s in series:
        print(f"  {s['forecast_date']}: {s['price']} ({s.get('lower', '?')} ~ {s.get('upper', '?')}) [{s.get('confidence', '?')}]")
