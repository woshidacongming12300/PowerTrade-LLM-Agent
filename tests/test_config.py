"""Unit tests for config loading (no external services)."""

from __future__ import annotations

from utils.config_handler import (
    load_agent_config,
    load_chroma_config,
    load_prompts_config,
    load_rag_config,
)
from utils.prompt_loader import load_react_prompts, load_system_prompts


def test_rag_config_has_model_fields():
    cfg = load_rag_config()
    assert cfg["chat_model_name"]
    assert cfg["embedding_model_name"]


def test_chroma_config_has_collection():
    cfg = load_chroma_config()
    assert cfg["collection_name"]
    assert cfg["persist_directory"]
    assert cfg["k"] >= 1


def test_prompts_config_has_react_path():
    cfg = load_prompts_config()
    assert cfg["react_prompt_path"]


def test_agent_config_has_modes():
    cfg = load_agent_config()
    # Must at least exist as a dict we can extend.
    assert isinstance(cfg, dict)


def test_react_prompt_loads_with_tools_mention():
    text = load_react_prompts()
    assert isinstance(text, str) and len(text) > 100
    assert "工具" in text or "tool" in text.lower()


def test_system_prompt_loads():
    text = load_system_prompts()
    assert isinstance(text, str) and len(text) > 50
