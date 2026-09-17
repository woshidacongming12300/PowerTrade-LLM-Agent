"""Unit tests for the tool registry and tool failure semantics."""

from __future__ import annotations

import pytest

from agent.langgraph_agent import TOOLS


def test_tool_registry_has_expected_tools():
    names = {t.name for t in TOOLS}
    expected = {
        "rag_summarize",
        "get_texas_weather",
        "predict_electricity_price",
        "forecast_electricity_prices",
        "get_chronos_prediction",
        "get_data_freshness",
        "get_current_date",
        "get_trading_advice",
        "check_price_alerts",
        "generate_daily_briefing",
    }
    assert expected.issubset(names), f"missing tools: {expected - names}"
    assert len(TOOLS) >= 10


def test_tools_have_descriptions():
    for t in TOOLS:
        assert getattr(t, "name", ""), f"tool without name: {t}"
        assert getattr(t, "description", ""), f"tool without description: {t.name}"


def test_predict_tool_fails_loudly_on_bad_date():
    """Regression: single-day predict must raise (not return a fake 48.88 value)."""
    from agent.tools.agent_tools import predict_electricity_price

    with pytest.raises(ValueError, match="无法解析|预测失败|单日预测失败"):
        predict_electricity_price.invoke("不存在的日期描述xyz")
