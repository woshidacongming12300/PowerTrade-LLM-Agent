"""Unit tests for natural-language date parsing (regression for the 未来一周 bug)."""

from __future__ import annotations

from datetime import date

import pytest

from algorithms.epf_predictor import (
    forecast_steps,
    parse_forecast_end_date,
    predict_future_prices,
    predict_future_price,
)

ANCHOR = date(2026, 9, 1)


@pytest.mark.parametrize(
    "text, expected_delta",
    [
        ("明天", 1),
        ("后天", 2),
        ("大后天", 3),
        ("今天", 0),
        ("3天后", 3),
        ("5天后", 5),
        ("tomorrow", 1),
        ("day after tomorrow", 2),
        ("3 days later", 3),
        ("未来一周", 7),
        ("一周后", 7),
        ("下周", 7),
        ("下星期", 7),
        ("一星期", 7),
        ("一星期后", 7),
        ("下个礼拜", 7),
        ("next week", 7),
        ("a week later", 7),
        ("one week", 7),
        ("下个月", 30),
        ("两个月后", 60),
        ("next month", 30),
    ],
)
def test_parse_forecast_end_date_relative(text, expected_delta):
    end = parse_forecast_end_date(text, ANCHOR)
    assert end == ANCHOR + __import__("datetime").timedelta(days=expected_delta)


def test_parse_forecast_end_date_iso():
    assert parse_forecast_end_date("2026-09-15", ANCHOR) == date(2026, 9, 15)
    assert parse_forecast_end_date("2026/09/15", ANCHOR) == date(2026, 9, 15)


def test_parse_forecast_end_date_empty_defaults_to_tomorrow():
    assert parse_forecast_end_date("", ANCHOR) == ANCHOR + __import__("datetime").timedelta(days=1)


def test_parse_forecast_end_date_invalid_raises():
    with pytest.raises(ValueError):
        parse_forecast_end_date("前天", ANCHOR)


def test_forecast_steps_clamps_to_max():
    # 60 days out clamps to MAX_ROLL_STEPS (30) — should not explode.
    steps = forecast_steps(ANCHOR, ANCHOR + __import__("datetime").timedelta(days=60))
    assert 1 <= steps <= 30


def test_predict_future_price_returns_number():
    """Single-point forecast returns a plausible positive price (pure torch, no network)."""
    price = predict_future_price("明天")
    assert isinstance(price, float)
    assert 0 < price < 200


def test_predict_future_prices_week_series():
    """The exact scenario from the bug report: 未来一周 yields 7 rows with confidence."""
    rows = predict_future_prices("未来一周", with_confidence=True)
    assert isinstance(rows, list)
    assert len(rows) == 7
    first = rows[0]
    assert "forecast_date" in first
    assert "price" in first
    assert "lower" in first
    assert "upper" in first
