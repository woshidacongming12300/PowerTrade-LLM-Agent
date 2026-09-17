"""Forecast endpoints: single-day and multi-day LSTM prediction (+ Chronos cross-check)."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from api.deps import require_api_key
from api.schemas import ForecastPoint, PredictRequest, PredictResponse

from algorithms.epf_predictor import predict_future_price, predict_future_prices
from agent.tools.agent_tools import forecast_electricity_prices, get_chronos_prediction

router = APIRouter(prefix="/api", tags=["forecast"], dependencies=[Depends(require_api_key)])


def _parse_forecast_json(raw: str) -> dict[str, Any]:
    """Parse the JSON string returned by forecast_electricity_prices."""
    try:
        data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail={
            "code": "forecast_unparseable",
            "message": f"Forecast tool returned invalid JSON: {exc}",
        }) from exc
    if not isinstance(data, dict) or not data.get("series"):
        msg = data.get("error") if isinstance(data, dict) else ""
        raise HTTPException(status_code=502, detail={
            "code": "forecast_failed",
            "message": msg or "Forecast returned no series",
        })
    return data


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Single-day prediction (LSTM) or multi-day forecast (LSTM + Chronos cross-check)."""
    try:
        if req.horizon_days is not None and req.horizon_days > 1:
            # Multi-day: use the tool that emits JSON with confidence + chronos.
            raw = forecast_electricity_prices(req.target_date)
            data = _parse_forecast_json(raw)
            series = [
                ForecastPoint(
                    forecast_date=str(r.get("forecast_date", "")),
                    price=float(r.get("price", 0.0)),
                    low=r.get("lower"),
                    high=r.get("upper"),
                )
                for r in data["series"]
            ]
            chronos = [
                ForecastPoint(
                    forecast_date=str(r.get("forecast_date", "")),
                    price=float(r["chronos_crosscheck"]),
                )
                for r in data["series"]
                if r.get("chronos_crosscheck") is not None
            ] or None
            return PredictResponse(
                mode="multi", series=series, chronos=chronos,
                meta={"summary": data.get("summary", {})},
            )

        # Single-day
        price = float(predict_future_price(req.target_date))
        series = [ForecastPoint(forecast_date="", price=price)]
        return PredictResponse(mode="single", series=series)

    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail={
            "code": "predict_error",
            "message": str(exc),
        }) from exc


@router.post("/predict/chronos", response_model=dict)
def predict_chronos(req: PredictRequest) -> dict:
    """Chronos-T5 single-point prediction as a cross-validation reference."""
    try:
        price = float(get_chronos_prediction(req.target_date))
        return {"target_date": req.target_date, "chronos_price": price}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail={
            "code": "chronos_error",
            "message": str(exc),
        }) from exc
