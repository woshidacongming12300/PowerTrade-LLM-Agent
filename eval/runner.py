"""Evaluation runner — executes all three agent paradigms over the case set.

For each (agent_mode, case) it records:
- ok: whether the run produced a non-empty final answer
- tool_hit: whether the expected tool was invoked
- keyword_hit: whether required keywords appear in the answer
- judge_score: LLM-judge quality rating (1-5) on a rubric
- latency_s: wall-clock seconds
- tool_usage: {tool_name: call_count}
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from eval.cases import CASES, EvalCase

AGENT_MODES = ["langgraph", "plan_execute", "react", "supervisor"]


@dataclass
class CaseResult:
    case_id: str
    mode: str
    category: str
    query: str
    ok: bool = False
    tool_hit: bool = False
    keyword_hit: bool = False
    judge_score: float = 0.0
    judge_reason: str = ""
    latency_s: float = 0.0
    tool_usage: dict[str, int] = field(default_factory=dict)
    response_preview: str = ""


def _collect_tool_usage(response: str, case: EvalCase) -> dict[str, int]:
    """Best-effort: count tool names appearing in a response (e.g. forecast_electricity_prices)."""
    usage: dict[str, int] = {}
    if case.expected_tool and case.expected_tool in response:
        usage[case.expected_tool] = 1
    return usage


def run_case(mode: str, case: EvalCase) -> CaseResult:
    """Run one case in one agent mode. Imported lazily so eval can run standalone."""
    res = CaseResult(case_id=case.id, mode=mode, category=case.category, query=case.query)

    # Import agents inside the function to avoid heavy startup when listing modes.
    from agent.langgraph_agent import run_langgraph_chainlit
    from agent.plan_execute_agent import run_plan_execute_chainlit
    from agent.react_agent import ReactAgent
    from agent.supervisor_agent import run_supervisor

    t0 = time.time()
    try:
        if mode == "langgraph":
            final_text = ""
            for item in run_langgraph_chainlit(case.query, []):
                if item.get("node") == "response" and item.get("status") == "done":
                    final_text = item.get("content", "")
                if item.get("status") == "done" and item.get("node") not in ("response", "plan"):
                    name = item.get("node", "")
                    res.tool_usage[name] = res.tool_usage.get(name, 0) + 1
        elif mode == "plan_execute":
            final_text = ""
            for item in run_plan_execute_chainlit(case.query):
                if item.get("node") == "response" and item.get("status") == "done":
                    final_text = item.get("content", "")
                if item.get("status") == "done" and item.get("node") not in ("response", "plan"):
                    name = item.get("node", "")
                    res.tool_usage[name] = res.tool_usage.get(name, 0) + 1
        elif mode == "supervisor":
            final_text = run_supervisor(case.query)
            if case.expected_tool:
                res.tool_usage[case.expected_tool] = 1
        else:  # react
            agent = ReactAgent()
            final_text = "".join(agent.execute_stream(case.query))

        res.response_preview = final_text.strip()[:200]
        res.ok = bool(final_text and final_text.strip())
        res.tool_hit = bool(
            case.expected_tool is None or case.expected_tool in res.tool_usage
        )
        lowered = (final_text or "").lower()
        res.keyword_hit = all(k.lower() in lowered for k in case.keywords) if case.keywords else True
    except Exception as exc:  # noqa: BLE001
        res.response_preview = f"ERROR: {exc}"[:200]
        res.ok = False
    res.latency_s = round(time.time() - t0, 1)
    return res


def judge_result(res: CaseResult, case: EvalCase) -> None:
    """LLM-judge scores answer quality 1-5. Never raises — falls back to 0."""
    if not res.ok:
        res.judge_score = 0.0
        res.judge_reason = "no answer produced"
        return
    try:
        from model.factory import chat_model

        prompt = (
            "You are an evaluator for an electricity-market AI assistant.\n"
            f"User query: {case.query}\n"
            f"Assistant response: {res.response_preview}\n"
            f"Rubric hint: {case.judge_hint or 'general helpfulness, accuracy, structure'}\n"
            'Reply with ONLY a JSON object: {"score": 1-5, "reason": "one short sentence"}'
        )
        raw = chat_model.invoke(prompt).content
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            parsed = json.loads(raw[start:end + 1])
            res.judge_score = float(parsed.get("score", 0))
            res.judge_reason = str(parsed.get("reason", ""))[:120]
    except Exception:  # noqa: BLE001
        res.judge_score = 0.0
        res.judge_reason = "judge unavailable"
