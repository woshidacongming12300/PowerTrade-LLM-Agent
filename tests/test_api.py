"""Integration tests against the FastAPI app.

Endpoints that need no external credentials run everywhere; live-model
endpoints are marked `integration` and auto-skipped when DASHSCOPE_API_KEY
is absent (i.e. in CI).
"""

from __future__ import annotations

import pytest


def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "supervisor" in body["agents"]
    assert isinstance(body["rag_ready"], bool)


def test_tools_registry_endpoint(client):
    r = client.get("/api/tools")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 10
    names = {t["name"] for t in body["tools"]}
    assert "forecast_electricity_prices" in names
    for t in body["tools"]:
        assert t["name"] and t["description"]


def test_memory_endpoint_roundtrip(client):
    # Empty user -> empty memory.
    r = client.post("/api/memory", json={"user_id": "test_api_user"})
    assert r.status_code == 200
    body = r.json()
    assert body["user_id"] == "test_api_user"
    assert isinstance(body["preferences"], dict)
    assert isinstance(body["recent_predictions"], list)


def test_chat_validation_error(client):
    """Empty message must be rejected by the contract (no LLM call)."""
    r = client.post("/api/chat", json={"message": ""})
    assert r.status_code == 422


@pytest.mark.integration
def test_rag_search_live(client):
    """Requires embedding model + local Chroma index (skipped without key)."""
    r = client.post("/api/rag/search", json={"query": "LMP由哪几个分量构成", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body.get("chunks"), list) and len(body["chunks"]) > 0


@pytest.mark.integration
def test_chat_supervisor_live(client):
    """End-to-end supervisor chat (skipped without key)."""
    r = client.post("/api/chat", json={"message": "预测明天的电价", "mode": "supervisor"})
    assert r.status_code == 200
    body = r.json()
    assert body["response"]
    assert body["mode"] == "supervisor"
