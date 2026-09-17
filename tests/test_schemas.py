"""Unit tests for API Pydantic contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import ChatRequest, ChatStreamEvent, ToolStep, ToolsResponse


def test_chat_request_defaults():
    req = ChatRequest(message="预测明天的电价")
    assert req.mode == "langgraph"
    assert req.lang == "zh"
    assert req.user_id == "default"
    assert req.history == []


@pytest.mark.parametrize("mode", ["langgraph", "plan_execute", "react", "supervisor"])
def test_chat_request_allowed_modes(mode):
    req = ChatRequest(message="hi", mode=mode)
    assert req.mode == mode


def test_chat_request_rejects_unknown_mode():
    with pytest.raises(ValidationError):
        ChatRequest(message="hi", mode="mystery_agent")


def test_chat_request_rejects_empty_message():
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_tool_step_lifecycle():
    s = ToolStep(node="forecast", status="running")
    assert s.title == ""
    assert s.content == ""


def test_chat_stream_event_types():
    for t in ("step_running", "step_done", "response_chunk", "response_done", "error"):
        ev = ChatStreamEvent(type=t)
        assert ev.type == t


def test_tools_response_model():
    from api.schemas import ToolInfo

    resp = ToolsResponse(count=1, tools=[ToolInfo(name="get_texas_weather", description="wx")])
    assert resp.count == 1
    assert resp.tools[0].name == "get_texas_weather"
