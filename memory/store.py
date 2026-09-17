"""SQLite-backed long-term memory store (thread-safe)."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DB = os.path.join(_PROJECT_ROOT, "agent_memory.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id   TEXT NOT NULL,
    key       TEXT NOT NULL,
    value     TEXT NOT NULL,
    updated_at TEXT,
    PRIMARY KEY (user_id, key)
);
CREATE TABLE IF NOT EXISTS prediction_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    mode        TEXT,
    target_date TEXT,
    price       REAL,
    lower       REAL,
    upper       REAL,
    model_track TEXT,
    created_at  TEXT
);
CREATE TABLE IF NOT EXISTS conversation_memory (
    user_id   TEXT NOT NULL,
    session_id TEXT NOT NULL,
    summary   TEXT,
    updated_at TEXT,
    PRIMARY KEY (user_id, session_id)
);
"""


class MemoryStore:
    def __init__(self, db_path: str = _DEFAULT_DB) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ---------- user preferences ----------
    def set_preference(self, user_id: str, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO user_preferences (user_id, key, value, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id, key) DO UPDATE SET
                     value = excluded.value, updated_at = excluded.updated_at""",
                (user_id, key, value, _now()),
            )
            self._conn.commit()

    def get_preferences(self, user_id: str) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value FROM user_preferences WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        return {k: v for k, v in rows}

    # ---------- prediction history ----------
    def save_prediction(
        self,
        user_id: str,
        mode: str,
        target_date: str,
        price: float,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        model_track: str = "lstm",
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO prediction_history
                   (user_id, mode, target_date, price, lower, upper, model_track, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, mode, target_date, price, lower, upper, model_track, _now()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def recent_predictions(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT mode, target_date, price, lower, upper, model_track, created_at
                   FROM prediction_history WHERE user_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [
            {
                "mode": r[0], "target_date": r[1], "price": r[2],
                "lower": r[3], "upper": r[4], "model_track": r[5], "created_at": r[6],
            }
            for r in rows
        ]

    # ---------- conversation memory ----------
    def save_conversation_summary(self, user_id: str, session_id: str, summary: str) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO conversation_memory (user_id, session_id, summary, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id, session_id) DO UPDATE SET
                     summary = excluded.summary, updated_at = excluded.updated_at""",
                (user_id, session_id, summary, _now()),
            )
            self._conn.commit()

    def get_conversation_summaries(self, user_id: str, limit: int = 5) -> list[dict[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT session_id, summary, updated_at FROM conversation_memory
                   WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [{"session_id": r[0], "summary": r[1], "updated_at": r[2]} for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Process-wide singleton
_store: Optional[MemoryStore] = None


def get_store() -> MemoryStore:
    global _store
    if _store is None:
        _store = MemoryStore()
    return _store
