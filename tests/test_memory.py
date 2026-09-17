"""Unit tests for the SQLite long-term memory store & service."""

from __future__ import annotations

from memory.service import MemoryService
from memory.store import MemoryStore


def test_memory_store_preferences(tmp_db_path):
    store = MemoryStore(tmp_db_path)
    store.set_preference("u1", "region", "ERCOT")
    store.set_preference("u1", "lang", "zh")
    prefs = store.get_preferences("u1")
    assert prefs == {"region": "ERCOT", "lang": "zh"}
    # Update overwrites, isolation by user.
    store.set_preference("u1", "region", "MISO")
    assert store.get_preferences("u1")["region"] == "MISO"
    assert store.get_preferences("other") == {}
    store.close()


def test_memory_store_prediction_history(tmp_db_path):
    store = MemoryStore(tmp_db_path)
    store.save_prediction("u1", "langgraph", "2026-09-02", 45.56, 40.0, 51.0)
    store.save_prediction("u1", "supervisor", "2026-09-03", 47.0)
    rows = store.recent_predictions("u1", limit=5)
    assert len(rows) == 2
    assert rows[0]["target_date"] == "2026-09-03"  # newest first
    assert rows[0]["price"] == 47.0
    assert rows[1]["mode"] == "langgraph"
    assert rows[1]["lower"] == 40.0
    store.close()


def test_memory_service_extract_and_inject(tmp_db_path):
    svc = MemoryService("u2", store=MemoryStore(tmp_db_path))
    found = svc.extract_and_store("我喜欢关注ERCOT的电价，用中文回答")
    assert found.get("region") == "ercot"
    assert "region=ercot" in svc.inject_memory()

    svc.remember_prediction("2026-09-02", 45.56, mode="supervisor")
    block = svc.inject_memory()
    assert "2026-09-02" in block
    assert "45.56" in block


def test_memory_persists_across_service_instances(tmp_db_path):
    """Data must survive reopening (real persistence, not in-memory)."""
    svc1 = MemoryService("u3", store=MemoryStore(tmp_db_path))
    svc1.remember_preference("horizon", "一周")
    svc1.store.close()

    svc2 = MemoryService("u3", store=MemoryStore(tmp_db_path))
    assert svc2.preferences().get("horizon") == "一周"
    svc2.store.close()
