"""Unit tests for the supervisor routing decision (pure parse_route, no LLM)."""

from __future__ import annotations

import pytest

from agent.supervisor_agent import parse_route, COORDINATOR_PROMPT


@pytest.mark.parametrize(
    "raw, expected_next",
    [
        ('{"next": "ForecastAgent", "target_date": "明天"}', "ForecastAgent"),
        ('{"next": "WeatherAgent", "target_date": ""}', "WeatherAgent"),
        ('{"next": "ResearchAgent", "target_date": ""}', "ResearchAgent"),
        ('{"next": "chat", "target_date": ""}', "chat"),
        # Markdown-fenced output (model likes to wrap in ```json).
        ('```json\n{"next": "ForecastAgent", "target_date": "3天后"}\n```', "ForecastAgent"),
        # Prose around the JSON.
        ('Sure! Here you go: {"next": "WeatherAgent", "target_date": "休斯顿"}', "WeatherAgent"),
        # Empty / garbage -> fallback to chat (answer directly).
        ("", "chat"),
        ("I don't know what you mean", "chat"),
        ("{invalid json", "chat"),
    ],
)
def test_parse_route(raw, expected_next):
    assert parse_route(raw)["next"] == expected_next


def test_parse_route_keeps_target_date():
    route = parse_route('{"next": "ForecastAgent", "target_date": "未来一周"}')
    assert route["target_date"] == "未来一周"


def test_parse_route_rejects_unknown_agent():
    """Model inventing a new agent name must not crash routing."""
    route = parse_route('{"next": "MysteryAgent", "target_date": ""}')
    assert route["next"] == "chat"


def test_coordinator_prompt_formats():
    """The prompt must format cleanly and embed the user input."""
    prompt = COORDINATOR_PROMPT.format(today="2026-09-01", user_input="预测明天的电价")
    assert "2026-09-01" in prompt
    assert "预测明天的电价" in prompt
    assert "{user_input}" not in prompt  # placeholder must be substituted
