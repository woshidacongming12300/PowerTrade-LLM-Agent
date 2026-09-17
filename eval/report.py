"""Report generation for agent evaluation runs."""

from __future__ import annotations

from datetime import datetime

from eval.cases import CASES, get_cases_by_category
from eval.runner import AGENT_MODES, CaseResult


def build_report(results: dict[str, dict[str, CaseResult]]) -> str:
    """results[mode][case_id] -> CaseResult. Returns a markdown report."""
    lines: list[str] = []
    lines.append("# Agent Evaluation Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Cases: {len(CASES)}  |  Modes: {', '.join(AGENT_MODES)}")
    lines.append("")

    # ---- Summary table per mode ----
    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| Mode | OK% | Tool-hit% | Keyword% | Avg Judge | Avg Latency (s) |")
    lines.append("|------|-----|-----------|----------|-----------|-----------------|")
    for mode in AGENT_MODES:
        mode_results = results.get(mode, {})
        n = max(len(mode_results), 1)
        ok = sum(1 for r in mode_results.values() if r.ok)
        tool_hit = sum(1 for r in mode_results.values() if r.tool_hit)
        kw = sum(1 for r in mode_results.values() if r.keyword_hit)
        judge = sum(r.judge_score for r in mode_results.values()) / n
        lat = sum(r.latency_s for r in mode_results.values()) / n
        lines.append(
            f"| {mode} | {ok}/{len(mode_results)} ({100*ok/max(len(mode_results),1):.0f}%) "
            f"| {100*tool_hit/max(len(mode_results),1):.0f}% "
            f"| {100*kw/max(len(mode_results),1):.0f}% "
            f"| {judge:.2f} | {lat:.1f} |"
        )
    lines.append("")

    # ---- Per-category breakdown ----
    lines.append("## Per-Category (tool-hit % / judge)")
    lines.append("")
    lines.append("| Category | Cases | langgraph | plan_execute | react |")
    lines.append("|----------|-------|-----------|--------------|-------|")
    for category, cases in get_cases_by_category().items():
        row = [f"| {category} | {len(cases)}"]
        for mode in AGENT_MODES:
            mode_results = results.get(mode, {})
            cat_res = [mode_results[c.id] for c in cases if c.id in mode_results]
            if not cat_res:
                row.append(" - / -")
                continue
            hit = sum(1 for r in cat_res if r.tool_hit)
            judge = sum(r.judge_score for r in cat_res) / len(cat_res)
            row.append(f" {100*hit/len(cat_res):.0f}% / {judge:.1f}")
        lines.append(" |".join(row) + " |")
    lines.append("")

    # ---- Per-case detail ----
    lines.append("## Per-Case Detail")
    lines.append("")
    for mode in AGENT_MODES:
        lines.append(f"### {mode}")
        lines.append("")
        lines.append("| Case | Cat | OK | Tool | Kw | Judge | Lat(s) | Preview |")
        lines.append("|------|-----|----|------|----|-------|--------|---------|")
        mode_results = results.get(mode, {})
        for c in CASES:
            r = mode_results.get(c.id)
            if not r:
                continue
            preview = (r.response_preview or "")[:40].replace("|", "/").replace("\n", " ")
            lines.append(
                f"| {r.case_id} | {r.category} | {'Y' if r.ok else 'N'} "
                f"| {'Y' if r.tool_hit else 'N'} | {'Y' if r.keyword_hit else 'N'} "
                f"| {r.judge_score:.1f} | {r.latency_s:.1f} | {preview} |"
            )
        lines.append("")
    return "\n".join(lines)


def print_console_summary(results: dict[str, dict[str, CaseResult]]) -> None:
    """Compact console summary."""
    print("=" * 64)
    print("AGENT EVALUATION SUMMARY")
    print("=" * 64)
    for mode in AGENT_MODES:
        mode_results = results.get(mode, {})
        n = max(len(mode_results), 1)
        ok = sum(1 for r in mode_results.values() if r.ok)
        tool_hit = sum(1 for r in mode_results.values() if r.tool_hit)
        judge = sum(r.judge_score for r in mode_results.values()) / n
        lat = sum(r.latency_s for r in mode_results.values()) / n
        print(f"  {mode:<13} OK={ok}/{len(mode_results)}  tool_hit={100*tool_hit/n:.0f}%  "
              f"judge={judge:.2f}  avg_latency={lat:.1f}s")
    print("=" * 64)
