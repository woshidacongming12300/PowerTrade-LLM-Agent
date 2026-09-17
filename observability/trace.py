"""Observability layer for agent runs — pluggable, zero-dependency by default.

Design:
- `TraceRecorder` captures per-run events: model calls, tool calls, latency,
  token usage (when available), errors.
- Backend is pluggable:
    * `JsonlTraceBackend` (default): append structured events to trace.jsonl —
      works offline, no services needed.
    * `LangfuseTraceBackend`: forwards events to a Langfuse instance when
      LANGFUSE_* env vars are configured (self-hosted via docker-compose or
      Langfuse Cloud).

Usage (inside agent code):
    from observability.trace import get_recorder
    rec = get_recorder()
    with rec.span("agent_run", mode="langgraph", query=q):
        ... rec.event("tool_call", tool="get_texas_weather", ok=True, latency_ms=120)
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_JSONL = os.path.join(_PROJECT_ROOT, "trace.jsonl")


# ============================================================
# Backends
# ============================================================
class TraceBackend:
    def record(self, entry: dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        pass


class JsonlTraceBackend(TraceBackend):
    """Append trace entries to a JSONL file. Safe for concurrent processes
    (each write is a single append line)."""

    def __init__(self, path: str = _DEFAULT_JSONL) -> None:
        self.path = path

    def record(self, entry: dict[str, Any]) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass  # tracing must never break the agent

    def close(self) -> None:
        pass


class LangfuseTraceBackend(TraceBackend):
    """Forward entries to a Langfuse instance (self-hosted or cloud).

    Enabled only when LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
    are set. Events are buffered per-run id into one Langfuse trace with
    spans per tool call.
    """

    def __init__(self) -> None:
        import langfuse  # deferred import — only needed when enabled

        self._langfuse = langfuse.Langfuse(
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
            host=os.environ.get("LANGFUSE_HOST", "http://localhost:3000"),
        )
        self._runs: dict[str, Any] = {}

    def record(self, entry: dict[str, Any]) -> None:
        try:
            run_id = entry.get("run_id", "")
            if entry.get("kind") == "run_start":
                trace = self._langfuse.trace(
                    id=run_id,
                    name=entry.get("mode", "agent"),
                    input=entry.get("query", ""),
                    metadata={"lang": entry.get("lang", "")},
                )
                self._runs[run_id] = trace
            elif entry.get("kind") == "tool":
                trace = self._runs.get(run_id)
                if trace is None:
                    return
                trace.span(
                    name=entry.get("tool", "tool"),
                    input=entry.get("input", ""),
                    output=entry.get("output", ""),
                    start_time=entry.get("ts"),
                    metadata={"ok": entry.get("ok"), "latency_ms": entry.get("latency_ms")},
                )
            elif entry.get("kind") == "run_end":
                trace = self._runs.get(run_id)
                if trace is not None:
                    trace.update(
                        output=entry.get("response", ""),
                        end_time=entry.get("ts"),
                        metadata={
                            "latency_s": entry.get("latency_s"),
                            "tool_count": entry.get("tool_count"),
                        },
                    )
        except Exception:  # noqa: BLE001
            pass  # never break the agent because of observability

    def close(self) -> None:
        try:
            self._langfuse.flush()
        except Exception:  # noqa: BLE001
            pass


# ============================================================
# Recorder
# ============================================================
@dataclass
class TraceEntry:
    kind: str  # run_start | tool | run_end
    run_id: str
    mode: str = ""
    query: str = ""
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tool: str = ""
    input: str = ""
    output: str = ""
    ok: bool = True
    latency_ms: int = 0
    latency_s: float = 0.0
    tool_count: int = 0
    response: str = ""
    lang: str = "zh"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in ("", 0, False) or k in ("kind", "run_id")}


class TraceRecorder:
    def __init__(self, backend: TraceBackend | None = None) -> None:
        self.backend = backend or _default_backend()
        self._current_run: Optional[str] = None

    def start_run(self, mode: str, query: str, lang: str = "zh") -> str:
        run_id = uuid.uuid4().hex[:12]
        self._current_run = run_id
        self._tool_count = 0
        self.backend.record({
            "kind": "run_start", "run_id": run_id, "mode": mode,
            "query": query, "lang": lang,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        return run_id

    def tool(self, name: str, ok: bool, latency_ms: int, input_: str = "", output: str = "") -> None:
        self._tool_count += 1
        self.backend.record({
            "kind": "tool", "run_id": self._current_run or "", "tool": name,
            "ok": ok, "latency_ms": latency_ms, "input": input_[:500], "output": str(output)[:1000],
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    def end_run(self, response: str = "", latency_s: float = 0.0) -> None:
        self.backend.record({
            "kind": "run_end", "run_id": self._current_run or "",
            "response": response[:2000], "latency_s": round(latency_s, 2),
            "tool_count": getattr(self, "_tool_count", 0),
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        self._current_run = None

    @contextmanager
    def run(self, mode: str, query: str, lang: str = "zh") -> Iterator[str]:
        run_id = self.start_run(mode, query, lang)
        t0 = time.time()
        try:
            yield run_id
        finally:
            self.end_run(latency_s=time.time() - t0)

    def close(self) -> None:
        self.backend.close()


_default_recorder: Optional[TraceRecorder] = None


def _default_backend() -> TraceBackend:
    if all(os.environ.get(k) for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")):
        try:
            return LangfuseTraceBackend()
        except Exception:  # noqa: BLE001
            pass
    return JsonlTraceBackend()


def get_recorder() -> TraceRecorder:
    """Process-wide singleton recorder (lazy init)."""
    global _default_recorder
    if _default_recorder is None:
        _default_recorder = TraceRecorder()
    return _default_recorder
