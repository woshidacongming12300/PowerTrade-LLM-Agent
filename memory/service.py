"""High-level memory service: preference extraction, prompt injection, history.

Usage:
    from memory.service import MemoryService
    svc = MemoryService("user_001")

    svc.remember_preference("region", "ERCOT")          # extract/store a preference
    prompt = svc.inject_memory("user query here")       # augment a prompt with memory
    svc.remember_prediction("2026-08-28", 41.15, lower=35.85, upper=46.45, mode="langgraph")
    past = svc.recent_predictions()
"""

from __future__ import annotations

import re
from typing import Any, Optional

from memory.store import MemoryStore, get_store

# Simple heuristic preference extraction — keeps the memory layer dependency-free.
_PREFERENCE_PATTERNS: list[tuple[str, re.Pattern, int]] = [
    ("region", re.compile(r"(ERCOT|MISO|德州|休斯顿|印第安纳|Indiana|Texas)"), 1),
    ("lang", re.compile(r"(中文|英文|英语|English|Chinese|use\s+zh|use\s+en)"), 1),
    ("currency", re.compile(r"(美元|人民币|元|USD|CNY)"), 1),
    ("horizon", re.compile(r"(未来\s*\d+\s*天|next\s+\d+\s+days?|一周|一个月)"), 1),
]


class MemoryService:
    def __init__(self, user_id: str, store: Optional[MemoryStore] = None) -> None:
        self.user_id = user_id
        self.store = store or get_store()

    # ---------- preferences ----------
    def remember_preference(self, key: str, value: str) -> None:
        self.store.set_preference(self.user_id, key, value)

    def extract_and_store(self, text: str) -> dict[str, str]:
        """Scan user text for known preference cues and persist them."""
        found: dict[str, str] = {}
        for key, pattern, group in _PREFERENCE_PATTERNS:
            m = pattern.search(text)
            if m:
                value = m.group(group).lower()
                self.store.set_preference(self.user_id, key, value)
                found[key] = value
        return found

    def preferences(self) -> dict[str, str]:
        return self.store.get_preferences(self.user_id)

    # ---------- predictions ----------
    def remember_prediction(
        self,
        target_date: str,
        price: float,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        mode: str = "langgraph",
        model_track: str = "lstm",
    ) -> int:
        return self.store.save_prediction(
            self.user_id, mode, target_date, float(price), lower, upper, model_track
        )

    def recent_predictions(self, limit: int = 5) -> list[dict[str, Any]]:
        return self.store.recent_predictions(self.user_id, limit)

    # ---------- prompt injection ----------
    def inject_memory(self) -> str:
        """Build a memory block to prepend to the agent system prompt."""
        parts: list[str] = []
        prefs = self.preferences()
        if prefs:
            pref_line = ", ".join(f"{k}={v}" for k, v in prefs.items())
            parts.append(f"[Long-term memory] User preferences: {pref_line}")

        recent = self.recent_predictions(3)
        if recent:
            pred_lines = []
            for p in recent:
                pred_lines.append(
                    f"{p['target_date']}: {p['price']} $/MWh "
                    f"({p['model_track']}, {p['mode']})"
                )
            parts.append(
                "[Long-term memory] Recent forecasts: " + "; ".join(pred_lines)
            )
        return "\n".join(parts) if parts else ""
