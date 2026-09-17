"""Supervisor multi-agent orchestration (LangGraph).

A Coordinator LLM routes each user request to one of three specialist agents,
each of which owns a focused toolset:

- WeatherAgent   (气象感知): get_texas_weather
- ForecastAgent  (量化预测): predict_electricity_price / forecast_electricity_prices / get_chronos_prediction
- ResearchAgent  (政策研究): rag_summarize

The Coordinator also decides when to answer directly (casual chat) without
delegating. This is the classic supervisor/hierarchical multi-agent pattern:
one brain routes, specialists execute, and a synthesizer merges results.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Annotated, Literal, Sequence, TypedDict
import operator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from datetime import date as _date_today
from model.factory import chat_model
from agent.tools.agent_tools import (
    rag_summarize,
    get_texas_weather,
    predict_electricity_price,
    forecast_electricity_prices,
    get_chronos_prediction,
)

# ============================================================
# Specialist tools
# ============================================================
WEATHER_TOOLS = [get_texas_weather]
FORECAST_TOOLS = [
    predict_electricity_price,
    forecast_electricity_prices,
    get_chronos_prediction,
]
RESEARCH_TOOLS = [rag_summarize]

COORDINATOR_PROMPT = """You are the Coordinator of an electricity-market intelligence system. Today is {today}.

You MUST route every request to exactly one of these specialists. Never answer a domain question yourself:

- WeatherAgent: ANY weather/天气/温度/气温 question, real-time or forecast.
- ForecastAgent: ANY question about electricity price prediction/电价预测/价格走势/未来电价/明天电价/多少. The price is in $/MWh.
- ResearchAgent: ANY question about market rules/交易规则/政策/市场机制/LMP/日前市场/实时市场.
- chat: ONLY casual chat (greetings, thanks, who-are-you, off-topic). Never route domain questions to chat.

Routing examples:
- "预测明天的电价" -> {{"next": "ForecastAgent", "target_date": "明天"}}
- "休斯顿天气" -> {{"next": "WeatherAgent", "target_date": ""}}
- "LMP由哪几个分量构成" -> {{"next": "ResearchAgent", "target_date": ""}}
- "你好" -> {{"next": "chat", "target_date": ""}}
- "3天后电价" -> {{"next": "ForecastAgent", "target_date": "3天后"}}

Output ONLY a JSON object, no explanation:
{{"next": "<WeatherAgent|ForecastAgent|ResearchAgent|chat>", "target_date": "<date or ''>"}}

User: {user_input}"""


# ============================================================
# State & nodes
# ============================================================
class SupervisorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    target_date: str
    specialist_output: str


def parse_route(raw: str) -> dict:
    """Parse the Coordinator's raw LLM output into a routing decision.

    Tolerates markdown fences, surrounding prose and invalid JSON:
    falls back to `{"next": "chat"}` (answer directly) on any failure.
    """
    route = {"next": "chat", "target_date": ""}
    if not raw:
        return route
    try:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            parsed = json.loads(raw[start:end + 1])
            nxt = parsed.get("next", "chat")
            if nxt in ("WeatherAgent", "ForecastAgent", "ResearchAgent", "chat"):
                route["next"] = nxt
            route["target_date"] = parsed.get("target_date", "") or ""
    except Exception:  # noqa: BLE001
        pass
    return route


def coordinator_node(state: SupervisorState) -> dict:
    """The Coordinator decides which specialist to call."""
    user_input = ""
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            user_input = m.content
            break
    prompt = COORDINATOR_PROMPT.format(
        today=_date_today.today().isoformat(), user_input=user_input
    )
    raw = chat_model.invoke(prompt).content
    route = parse_route(raw)
    print(f"[Supervisor] route -> {route}")
    return {"next_agent": route["next"], "target_date": route["target_date"]}


def _run_tools(tools, input_text: str) -> str:
    """Run the first tool that accepts the input; collect all outputs."""
    parts = []
    for t in tools:
        try:
            out = t.invoke(input_text)
            parts.append(f"[{t.name}]: {out}")
        except Exception as e:  # noqa: BLE001
            parts.append(f"[{t.name}] error: {e}")
    return "\n".join(parts)


def weather_agent(state: SupervisorState) -> dict:
    out = _run_tools(WEATHER_TOOLS, state.get("target_date") or "forecast")
    return {"specialist_output": out, "messages": [AIMessage(content=out)]}


def forecast_agent(state: SupervisorState) -> dict:
    target = state.get("target_date") or "明天"
    out = _run_tools(FORECAST_TOOLS, target)

    # Persist forecast into long-term memory at the tool layer, so the
    # prediction history is recorded regardless of frontend (API or Chainlit).
    try:
        import json as _json
        import re as _re

        from memory.service import MemoryService

        price_match = _re.findall(r'"price":\s*([\d.]+)', out)
        date_match = _re.findall(r'"forecast_date":\s*"([^"]+)"', out)
        if price_match:
            dates = date_match or [""] * len(price_match)
            svc = MemoryService("default")
            for d, p in zip(dates, price_match[:5]):
                svc.remember_prediction(
                    target_date=d or target, price=float(p), mode="supervisor"
                )
    except Exception:  # noqa: BLE001 — memory must never break the flow
        pass

    return {"specialist_output": out, "messages": [AIMessage(content=out)]}


def research_agent(state: SupervisorState) -> dict:
    user_input = ""
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            user_input = m.content
            break
    out = _run_tools(RESEARCH_TOOLS, user_input)
    return {"specialist_output": out, "messages": [AIMessage(content=out)]}


SYNTHESIZER_PROMPT = """You are the synthesizer of an electricity-market intelligence system.

User request: {user_input}

Specialist output:
{specialist_output}

Synthesize a clear, concise answer in the same language as the user request.
Include concrete numbers where available. Use a table for multi-day forecasts."""


def synthesizer_node(state: SupervisorState) -> dict:
    user_input = ""
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            user_input = m.content
            break
    prompt = SYNTHESIZER_PROMPT.format(
        user_input=user_input,
        specialist_output=state.get("specialist_output", "")[:3000],
    )
    response = chat_model.invoke(prompt)
    return {"messages": [AIMessage(content=response.content)]}


def chat_node(state: SupervisorState) -> dict:
    """Direct answer path for casual chat."""
    user_input = ""
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            user_input = m.content
            break
    response = chat_model.invoke(
        f"你是一个电力市场 AI 助手。用友好简洁的方式回答（可用中文）：\n用户：{user_input}"
    )
    return {"messages": [AIMessage(content=response.content)]}


# ============================================================
# Graph
# ============================================================
def route_after_coordinator(state: SupervisorState) -> str:
    nxt = state.get("next_agent", "chat")
    if nxt in ("WeatherAgent", "ForecastAgent", "ResearchAgent"):
        return nxt
    return "chat"


graph = StateGraph(SupervisorState)
graph.add_node("coordinator", coordinator_node)
graph.add_node("WeatherAgent", weather_agent)
graph.add_node("ForecastAgent", forecast_agent)
graph.add_node("ResearchAgent", research_agent)
graph.add_node("chat", chat_node)
graph.add_node("synthesizer", synthesizer_node)

graph.set_entry_point("coordinator")
graph.add_conditional_edges(
    "coordinator",
    route_after_coordinator,
    {"WeatherAgent": "WeatherAgent", "ForecastAgent": "ForecastAgent",
     "ResearchAgent": "ResearchAgent", "chat": "chat"},
)
for spec in ("WeatherAgent", "ForecastAgent", "ResearchAgent", "chat"):
    graph.add_edge(spec, "synthesizer")
graph.add_edge("synthesizer", END)

supervisor_app = graph.compile()


def run_supervisor(query: str) -> str:
    """Run the supervisor multi-agent flow. Returns the final synthesized answer."""
    initial: SupervisorState = {
        "messages": [HumanMessage(content=query)],
        "next_agent": "chat",
        "target_date": "",
        "specialist_output": "",
    }
    final_text = ""
    for out in supervisor_app.stream(initial, stream_mode="values"):
        msgs = out.get("messages", [])
        if msgs and isinstance(msgs[-1], AIMessage) and msgs[-1].content:
            final_text = msgs[-1].content
    return final_text


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    for q in ["你好", "预测明天的电价", "休斯顿天气如何", "LMP由哪几个分量构成"]:
        print(f"\n{'='*50}\nQuery: {q}\n{'='*50}")
        print(run_supervisor(q)[:300])
