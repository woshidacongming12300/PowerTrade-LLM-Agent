"""Shared pytest fixtures — project-root imports, temp files, API test client."""

from __future__ import annotations

import os
import sys

import pytest

# Make the project root importable regardless of CWD.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _has_dashscope_key() -> bool:
    return bool(os.environ.get("DASHSCOPE_API_KEY"))


def pytest_collection_modifyitems(config, items):
    """Mark tests that need live model credentials so they can be skipped in CI."""
    if _has_dashscope_key():
        return
    skip = pytest.mark.skip(reason="DASHSCOPE_API_KEY not set — skipping live-model tests")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def tmp_db_path(tmp_path):
    """A fresh SQLite memory-store path per test."""
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def tmp_trace_path(tmp_path):
    """A fresh JSONL trace path per test."""
    return str(tmp_path / "test_trace.jsonl")


@pytest.fixture(scope="session")
def client():
    """FastAPI TestClient against api.main:app (no external calls for /health, /tools, /memory)."""
    from fastapi.testclient import TestClient

    from api.main import app

    with TestClient(app) as c:
        yield c
