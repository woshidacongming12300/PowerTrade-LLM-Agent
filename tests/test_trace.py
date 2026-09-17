"""Unit tests for the pluggable observability/trace layer."""

from __future__ import annotations

import json

from observability.trace import JsonlTraceBackend, TraceRecorder


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_jsonl_backend_roundtrip(tmp_trace_path):
    backend = JsonlTraceBackend(tmp_trace_path)
    backend.record({"kind": "run_start", "run_id": "abc", "mode": "langgraph"})
    backend.record({"kind": "run_end", "run_id": "abc"})
    lines = _read_lines(tmp_trace_path)
    assert len(lines) == 2
    assert lines[0]["kind"] == "run_start"
    assert lines[0]["mode"] == "langgraph"


def test_recorder_run_lifecycle(tmp_trace_path):
    rec = TraceRecorder(JsonlTraceBackend(tmp_trace_path))
    with rec.run("supervisor", "预测明天的电价", lang="zh") as run_id:
        rec.tool(name="forecast_electricity_prices", ok=True, latency_ms=1200)
    rec.close()

    lines = _read_lines(tmp_trace_path)
    kinds = [l["kind"] for l in lines]
    assert kinds == ["run_start", "tool", "run_end"]
    assert lines[1]["tool"] == "forecast_electricity_prices"
    assert lines[1]["ok"] is True
    assert lines[2]["tool_count"] == 1
    assert lines[0]["query"] == "预测明天的电价"


def test_trace_never_breaks_on_missing_dir(tmp_path):
    """Tracing must be best-effort: writing to an impossible path is a no-op."""
    backend = JsonlTraceBackend(str(tmp_path / "nope" / "x.jsonl"))
    backend.record({"kind": "run_start", "run_id": "x"})  # should not raise
