"""Unit tests for the eval framework (no live agents/LLM needed)."""

from __future__ import annotations

import os

import pytest

from eval.cases import CASES, EvalCase
from eval.report import build_report


def test_cases_are_well_formed():
    assert len(CASES) >= 10
    ids = [c.id for c in CASES]
    assert len(ids) == len(set(ids)), "case ids must be unique"
    categories = {c.category for c in CASES}
    assert {"chat", "weather", "forecast", "rag"} <= categories
    for c in CASES:
        assert isinstance(c, EvalCase)
        assert c.query and c.id and c.category
        assert isinstance(c.keywords, (list, tuple)) and len(c.keywords) > 0


def test_cases_cover_the_bug_fix_scenario():
    """The exact user-reported query must be an eval case."""
    queries = [c.query for c in CASES]
    assert any("未来一周" in q or "一周" in q for q in queries)


def test_build_report_produces_markdown(tmp_path):
    from eval.runner import CaseResult

    res = CaseResult(case_id="fc_01", mode="supervisor", category="forecast", query="预测明天电价")
    res.ok = True
    res.tool_hit = True
    res.keyword_hit = True
    res.judge_score = 4.0
    res.latency_s = 20.5
    res.response_preview = "明天电价为 45.56 $/MWh"

    out = tmp_path / "report.md"
    text = build_report(
        {"supervisor": {"fc_01": res}, "langgraph": {}}
    )
    out.write_text(text, encoding="utf-8")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert "supervisor" in text
    assert "fc_01" in text
