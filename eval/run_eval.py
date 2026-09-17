"""Entry point: run the full agent evaluation and write a report.

Usage:
    python eval/run_eval.py                 # all modes, all cases
    python eval/run_eval.py --mode react    # one mode
    python eval/run_eval.py --quick         # fewer cases (first 6 per category)
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from eval.cases import CASES
from eval.report import build_report, print_console_summary
from eval.runner import AGENT_MODES, CaseResult, judge_result, run_case

# Hard per-case timeout: a hung LLM call must never block the whole run.
CASE_TIMEOUT_S = 90


def _run_case_with_timeout(mode: str, case) -> CaseResult:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_case, mode, case)
        try:
            return future.result(timeout=CASE_TIMEOUT_S)
        except FutureTimeout:
            future.cancel()
            res = CaseResult(case_id=case.id, mode=mode, category=case.category, query=case.query)
            res.ok = False
            res.response_preview = f"TIMEOUT after {CASE_TIMEOUT_S}s"
            return res


def main() -> None:
    global CASE_TIMEOUT_S
    parser = argparse.ArgumentParser(description="Run agent evaluation")
    parser.add_argument("--mode", choices=AGENT_MODES, default=None, help="Only run this mode")
    parser.add_argument("--quick", action="store_true", help="Run a small subset for a fast check")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge (faster)")
    parser.add_argument("--timeout", type=int, default=CASE_TIMEOUT_S,
                        help=f"Per-case timeout in seconds (default {CASE_TIMEOUT_S})")
    parser.add_argument("--out", default="eval_report.md", help="Output report path")
    args = parser.parse_args()

    CASE_TIMEOUT_S = args.timeout

    modes = [args.mode] if args.mode else AGENT_MODES
    cases = CASES
    if args.quick:
        seen: set[str] = set()
        cases = []
        for c in CASES:
            if c.category not in seen:
                cases.append(c)
                seen.add(c.category)
            elif len([x for x in cases if x.category == c.category]) < 2:
                cases.append(c)

    print(f"Running {len(modes)} mode(s) x {len(cases)} cases (timeout={CASE_TIMEOUT_S}s)...\n", flush=True)
    results: dict[str, dict[str, object]] = {m: {} for m in modes}

    total = len(modes) * len(cases)
    done = 0
    for mode in modes:
        for case in cases:
            res = _run_case_with_timeout(mode, case)
            if not args.no_judge and res.ok:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    judge_future = pool.submit(judge_result, res, case)
                    try:
                        judge_future.result(timeout=30)
                    except FutureTimeout:
                        judge_future.cancel()
                        res.judge_score = 0.0
                        res.judge_reason = "judge timeout"
            results[mode][case.id] = res
            done += 1
            status = "OK" if res.ok else "FAIL"
            print(f"[{done}/{total}] {mode:<12} {case.id:<8} {status} "
                  f"tool={res.tool_hit} kw={res.keyword_hit} judge={res.judge_score:.1f} "
                  f"lat={res.latency_s:.1f}s", flush=True)

    print_console_summary(results)  # type: ignore[arg-type]

    report = build_report(results)  # type: ignore[arg-type]
    out_path = os.path.join(_project_root, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
