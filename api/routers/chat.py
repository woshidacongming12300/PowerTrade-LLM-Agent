"""Chat endpoints: full response and SSE streaming.

Reuses the exact generators the Chainlit UI uses (run_langgraph_chainlit /
run_plan_execute_chainlit / ReactAgent.execute_stream), so the API behavior
matches the UI behavior — one agent engine, two frontends.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Generator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.deps import require_api_key
from api.schemas import ChatRequest, ChatResponse, ChatStreamEvent, ToolStep

from agent.langgraph_agent import run_langgraph_chainlit
from agent.plan_execute_agent import run_plan_execute_chainlit
from agent.react_agent import ReactAgent
from agent.supervisor_agent import run_supervisor

router = APIRouter(prefix="/api/chat", tags=["chat"], dependencies=[Depends(require_api_key)])

# One ReactAgent instance is stateless per call (its graph holds no cross-call state).
_react_agent = ReactAgent()

_MODE_LABELS = {
    "langgraph": "LangGraph Agent",
    "plan_execute": "Plan-Execute Agent",
    "react": "ReAct Agent",
    "supervisor": "Supervisor Agent",
}


# ============================================================
# Internal: drive a chainlit-style generator and emit events
# ============================================================
def _iter_langgraph_events(query: str, history: list[dict[str, str]]) -> Generator[dict[str, Any], None, None]:
    """Emit step/response events from the LangGraph generator."""
    for item in run_langgraph_chainlit(query, history):
        node = item.get("node", "")
        status = item.get("status", "")
        content = item.get("content", "") or ""
        title = item.get("title", node)

        if node == "response" and status == "done":
            yield {"type": "response_done", "node": node, "title": title, "content": content}
        elif status == "running":
            yield {"type": "step_running", "node": node, "title": title, "content": content}
        else:
            yield {"type": "step_done", "node": node, "title": title, "content": content}


def _iter_plan_execute_events(query: str) -> Generator[dict[str, Any], None, None]:
    """Emit step/response events from the Plan-Execute generator."""
    for item in run_plan_execute_chainlit(query):
        node = item.get("node", "")
        status = item.get("status", "")
        content = item.get("content", "") or ""
        title = item.get("title", node)

        if node == "response" and status == "done":
            yield {"type": "response_done", "node": node, "title": title, "content": content}
        elif node == "plan" and status == "running":
            yield {"type": "step_running", "node": "plan", "title": "Planning", "content": content}
        else:
            yield {"type": "step_done", "node": node, "title": title, "content": content}


def _iter_react_events(query: str) -> Generator[dict[str, Any], None, None]:
    """Emit token-ish chunks from the ReAct stream.

    The underlying generator (stream_mode="values") also emits intermediate
    state noise (e.g. `{'next_node': ..., 'target_date': ...}` dict reprs and
    echoed inputs). Filter those out so the stream carries only real content.
    """
    for chunk in _react_agent.execute_stream(query):
        text = (chunk or "").strip()
        if not text:
            continue
        # Skip dict-repr intermediate states from the ReAct loop.
        if text.startswith("{") and text.endswith("}"):
            continue
        yield {"type": "response_chunk", "node": "response", "title": "", "content": chunk}


def _iter_events(req: ChatRequest) -> Generator[dict[str, Any], None, None]:
    """Dispatch to the requested agent mode's event generator."""
    if req.mode == "langgraph":
        yield from _iter_langgraph_events(req.message, req.history)
    elif req.mode == "plan_execute":
        yield from _iter_plan_execute_events(req.message)
    elif req.mode == "supervisor":
        # Supervisor runs end-to-end and returns one final answer.
        answer = run_supervisor(req.message)
        yield {"type": "response_done", "node": "response", "title": "Supervisor", "content": answer}
    else:
        yield from _iter_react_events(req.message)


def _extract_forecast_series(content: str) -> list[dict[str, Any]]:
    """Pull {"forecast_date": "...", "price": N} pairs out of a response (for charts)."""
    series: list[dict[str, Any]] = []
    try:
        matches = re.findall(r'"forecast_date":\s*"([^"]+)"\s*,\s*"price":\s*([\d.]+)', content)
        for d, p in matches:
            series.append({"forecast_date": d, "price": float(p)})
    except Exception:
        pass
    return series


# ============================================================
# POST /api/chat — full response
# ============================================================
@router.post("", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Run the agent end-to-end and return the complete answer + tool steps."""
    from observability.trace import get_recorder
    from memory.service import MemoryService

    response_text = ""
    steps: list[ToolStep] = []
    tool_usage: dict[str, int] = {}
    series: list[dict[str, Any]] = []

    recorder = get_recorder()
    recorder.start_run(req.mode, req.message, req.lang)
    import time as _time
    _t0 = _time.time()

    # Long-term memory: extract user preferences from this message.
    memory = MemoryService(req.user_id if getattr(req, "user_id", "") else "default")
    memory.extract_and_store(req.message)

    for event in _iter_events(req):
        etype = event["type"]
        node = event.get("node", "")
        content = event.get("content", "") or ""

        if etype == "response_done":
            response_text = content
        elif etype == "response_chunk":
            response_text += content
        elif etype in ("step_running", "step_done"):
            if node and node != "response" and node != "plan":
                tool_usage[node] = tool_usage.get(node, 0) + 1
                recorder.tool(name=node, ok=(etype == "step_done"),
                              latency_ms=0, input_=content[:200])
            steps.append(ToolStep(
                node=node,
                title=event.get("title", node),
                status="running" if etype == "step_running" else "done",
                content=content,
            ))
        if content:
            series = series or _extract_forecast_series(content)

    recorder.end_run(response=response_text, latency_s=_time.time() - _t0)

    # Persist forecast results into long-term memory.
    if series:
        for point in series:
            memory.remember_prediction(
                target_date=point.get("forecast_date") or "unknown",
                price=point.get("price", 0.0),
                lower=point.get("low"),
                upper=point.get("high"),
                mode=req.mode,
            )

    return ChatResponse(
        response=response_text,
        steps=steps,
        forecast_series=series,
        mode=req.mode,
        tool_usage=tool_usage,
    )


# ============================================================
# POST /api/chat/stream — SSE streaming
# ============================================================
@router.post("/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Stream the agent run as server-sent events.

    Event types: step_running | step_done | response_chunk | response_done | error.
    Each event is a JSON object on the `data:` line.
    """

    def event_source():
        from observability.trace import get_recorder

        recorder = get_recorder()
        recorder.start_run(req.mode, req.message, req.lang)
        import time as _time
        _t0 = _time.time()
        _full_response = ""
        try:
            for event in _iter_events(req):
                etype = event["type"]
                node = event.get("node", "")
                content = event.get("content", "") or ""
                if etype == "response_done":
                    # Emit the final answer in chunks for consistent UX with ReAct.
                    content = event.get("content", "") or ""
                    _full_response = content
                    for i in range(0, len(content), 80):
                        chunk = content[i:i + 80]
                        ev = ChatStreamEvent(
                            type="response_chunk", node="response", content=chunk
                        )
                        yield f"data: {ev.model_dump_json()}\n\n"
                    ev = ChatStreamEvent(type="response_done", node="response", content="")
                    yield f"data: {ev.model_dump_json()}\n\n"
                elif etype == "response_chunk":
                    _full_response += content
                    ev = ChatStreamEvent(
                        type=etype, node=node, title=event.get("title", ""), content=content,
                    )
                    yield f"data: {ev.model_dump_json()}\n\n"
                else:
                    if etype == "step_done" and node not in ("response", "plan"):
                        recorder.tool(name=node, ok=True, latency_ms=0,
                                      input_=content[:200], output=content[:500])
                    ev = ChatStreamEvent(
                        type=etype,
                        node=event.get("node", ""),
                        title=event.get("title", ""),
                        content=event.get("content", "") or "",
                    )
                    yield f"data: {ev.model_dump_json()}\n\n"
        except Exception as exc:  # noqa: BLE001 — surface any failure to the client
            ev = ChatStreamEvent(type="error", content=f"Agent error: {exc}")
            yield f"data: {ev.model_dump_json()}\n\n"
        finally:
            recorder.end_run(response=_full_response, latency_s=_time.time() - _t0)

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
