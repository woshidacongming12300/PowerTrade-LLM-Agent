"""
LSTM electricity price prediction — evaluation script (multivariate)
Calculates MAE / RMSE / MAPE / R2 on held-out test set.

Compares old univariate model vs new multivariate (price+weather) model.
"""
from __future__ import annotations

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
from datetime import date, timedelta

from algorithms.epf_predictor import (
    ElectricPriceLSTM, SEQ_LENGTH, N_FEATURES,
    _load_lstm_checkpoint, _roll_lstm,
)
from algorithms.price_data_provider import (
    build_merged_price_weather,
    get_recent_multivariate_window,
)
from algorithms.weather_data import fetch_weather_forecast, INDIANA_HUB


def _load_data(test_days: int = 30):
    """Load merged price+weather data, return train/test splits."""
    merged = build_merged_price_weather()
    merged = merged.sort_values("Delivery start date").reset_index(drop=True)

    all_dates = merged["Delivery start date"].dt.date.unique()
    all_dates = sorted(all_dates)

    if len(all_dates) < test_days + SEQ_LENGTH:
        raise ValueError(f"Need {test_days + SEQ_LENGTH} days, got {len(all_dates)}")

    train_end_idx = len(all_dates) - test_days
    train_dates = all_dates[:train_end_idx]
    test_dates = all_dates[train_end_idx:]

    return merged, all_dates, train_dates, test_dates


def evaluate_multivariate(test_days: int = 30) -> dict:
    """Evaluate the new multivariate LSTM model on held-out test days."""
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "algorithms", "lstm_epf_multivariate.pth",
    )

    if not os.path.exists(model_path):
        return {"error": "Multivariate model not found. Run train_model() first."}

    merged, all_dates, train_dates, test_dates = _load_data(test_days)
    daily_prices = merged.groupby(merged["Delivery start date"].dt.date)["price"].mean()

    scaler_min, scaler_max, state = _load_lstm_checkpoint(model_path)
    model = ElectricPriceLSTM()
    model.load_state_dict(state)
    model.eval()

    actuals, predictions = [], []

    print(f"[Eval-MV] Test: {test_dates[0]} ~ {test_dates[-1]} ({test_days}d), multivariate model")
    print(f"[Eval-MV] Rolling predictions...")

    for i, test_date in enumerate(test_dates):
        anchor = test_date - timedelta(days=1)
        try:
            window = get_recent_multivariate_window(anchor, SEQ_LENGTH)
        except Exception:
            continue

        # Fetch 1-day weather forecast
        try:
            fc = fetch_weather_forecast(INDIANA_HUB[0], INDIANA_HUB[1], forecast_days=1)
            weather_fc = np.array([[
                fc["temp_max"][0] if fc.get("temp_max") else 20.0,
                fc["temp_min"][0] if fc.get("temp_min") else 10.0,
                fc["precip"][0] if fc.get("precip") else 0.0,
                fc["wind_speed"][0] if fc.get("wind_speed") else 15.0,
            ]])
        except Exception:
            weather_fc = np.array([[20.0, 10.0, 0.0, 15.0]])

        preds = _roll_lstm(model, scaler_min, scaler_max, window, weather_fc, steps=1)
        pred = preds[0]
        actual = daily_prices.get(test_date)
        if actual is None:
            continue
        actuals.append(actual)
        predictions.append(pred)

        if (i + 1) % 10 == 0:
            cur_errs = np.abs(np.array(actuals) - np.array(predictions))
            print(f"  {i+1}/{test_days} ... MAE so far: {np.mean(cur_errs):.2f}")

    return _compute_metrics(actuals, predictions, test_dates, "Multivariate (price+weather)")


def evaluate_univariate(test_days: int = 30) -> dict:
    """Evaluate the old univariate LSTM model for comparison."""
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "algorithms", "lstm_epf.pth",
    )

    if not os.path.exists(model_path):
        return {"error": "Univariate model not found"}

    from algorithms.price_data_provider import get_recent_price_window

    merged, all_dates, train_dates, test_dates = _load_data(test_days)
    daily_prices = merged.groupby(merged["Delivery start date"].dt.date)["price"].mean()

    checkpoint = __import__("torch").load(model_path, weights_only=False)
    scaler_min = float(checkpoint["scaler_min"])
    scaler_max = float(checkpoint["scaler_max"])
    state = checkpoint["model_state_dict"]

    model = ElectricPriceLSTM(input_size=1)
    model.load_state_dict(state)
    model.eval()

    actuals, predictions = [], []

    print(f"\n[Eval-UV] Test: {test_dates[0]} ~ {test_dates[-1]} ({test_days}d), univariate model")
    print(f"[Eval-UV] Rolling predictions...")

    for i, test_date in enumerate(test_dates):
        anchor = test_date - timedelta(days=1)
        try:
            window = get_recent_price_window(anchor, SEQ_LENGTH)
        except Exception:
            continue

        # Old univariate roll
        scaled_input = (window - scaler_min) / (scaler_max - scaler_min)
        seq = scaled_input.flatten().tolist()
        with __import__("torch").no_grad():
            tensor_input = __import__("torch").tensor(seq, dtype=__import__("torch").float32).view(1, SEQ_LENGTH, 1)
            pred_scaled = model(tensor_input).item()
        pred = float(pred_scaled * (scaler_max - scaler_min) + scaler_min)

        actual = daily_prices.get(test_date)
        if actual is None:
            continue
        actuals.append(actual)
        predictions.append(pred)

        if (i + 1) % 10 == 0:
            cur_errs = np.abs(np.array(actuals) - np.array(predictions))
            print(f"  {i+1}/{test_days} ... MAE so far: {np.mean(cur_errs):.2f}")

    return _compute_metrics(actuals, predictions, test_dates, "Univariate (price only)")


def _compute_metrics(actuals, predictions, test_dates, label):
    if not actuals:
        return {"error": f"No valid predictions for {label}"}

    actuals = np.array(actuals)
    predictions = np.array(predictions)
    errors = np.abs(actuals - predictions)

    return {
        "model": label,
        "test_days": len(actuals),
        "test_range": f"{test_dates[0]} ~ {test_dates[-1]}",
        "MAE_$/MWh": round(float(np.mean(errors)), 2),
        "RMSE_$/MWh": round(float(np.sqrt(np.mean(errors ** 2))), 2),
        "MAPE_%": f"{float(np.mean(errors / np.abs(actuals)) * 100):.1f}%",
        "R2": round(float(1 - np.sum(errors ** 2) / np.sum((actuals - np.mean(actuals)) ** 2)), 4),
        "price_mean_$/MWh": round(float(np.mean(actuals)), 2),
        "price_std_$/MWh": round(float(np.std(actuals)), 2),
    }


if __name__ == "__main__":
    results = []

    # Evaluate multivariate
    mv = evaluate_multivariate()
    results.append(mv)

    # Evaluate univariate for comparison
    uv = evaluate_univariate()
    results.append(uv)

    print()
    print("=" * 60)
    print("LSTM Price Prediction — Model Comparison Report")
    print("=" * 60)

    for r in results:
        print(f"\n--- {r.get('model', 'Unknown')} ---")
        if "error" in r:
            print(f"  [SKIP] {r['error']}")
        else:
            for k, v in r.items():
                if k != "model":
                    print(f"  {k}: {v}")

    # Show winner
    if "MAE_$/MWh" in results[0] and "MAE_$/MWh" in results[1]:
        mv_mae = results[0]["MAE_$/MWh"]
        uv_mae = results[1]["MAE_$/MWh"]
        winner = "Multivariate" if mv_mae < uv_mae else "Univariate"
        delta = abs(mv_mae - uv_mae)
        print(f"\n[Verdict] {winner} wins by {delta:.2f} $/MWh in MAE")
    print()
