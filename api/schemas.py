"""Pydantic request/response contracts for the Agent API.

These models are the API's contract — every endpoint validates input and
shapes output through them, so consumers get stable, documented schemas
(generating the OpenAPI spec automatically).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ============================================================
# Chat
# ============================================================
class ChatRequest(BaseModel):
    """A single-turn chat request against one of the three agent modes."""

    message: str = Field(..., min_length=1, max_length=8000, description="User query text")
    user_id: str = Field(default="default", max_length=128, description="User id for long-term memory")
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description='Optional prior turns as [{"role": "user"|"assistant", "content": "..."}]',
    )
    mode: Literal["langgraph", "plan_execute", "react", "supervisor"] = Field(
        default="langgraph", description="Which agent paradigm to run"
    )
    lang: Literal["zh", "en"] = Field(default="zh", description="Response language hint")


class ToolStep(BaseModel):
    """One tool-execution event observed during an agent run."""

    node: str = Field(..., description="Tool / node name")
    title: str = Field(default="", description="Human-friendly step title")
    status: Literal["running", "done"] = Field(..., description="Step lifecycle state")
    content: str = Field(default="", description="Step output / progress text")


class ChatResponse(BaseModel):
    """Full (non-streaming) chat result."""

    response: str = Field(..., description="Final agent answer")
    steps: list[ToolStep] = Field(default_factory=list, description="Tool steps executed")
    forecast_series: list[dict[str, Any]] = Field(
        default_factory=list, description="Extracted forecast series for charting (if any)"
    )
    mode: str = Field(..., description="Agent mode that produced this response")
    tool_usage: dict[str, int] = Field(
        default_factory=dict, description="Map of tool name -> call count for observability"
    )


class ChatStreamEvent(BaseModel):
    """One SSE event payload for the streaming chat endpoint."""

    type: Literal["step_running", "step_done", "response_chunk", "response_done", "error"] = ...
    node: str = Field(default="", description="Step node for step events")
    title: str = Field(default="", description="Step title for step events")
    content: str = Field(default="", description="Chunk / step / error content")


# ============================================================
# Forecast
# ============================================================
class PredictRequest(BaseModel):
    """Request a single-day or multi-day electricity price forecast."""

    target_date: str = Field(
        default="tomorrow",
        description='Target: "tomorrow", "3天后", a date like "2026-09-01", or "N天后"',
    )
    horizon_days: int | None = Field(
        default=None, ge=1, le=30, description="For multi-day: number of days to forecast"
    )


class ForecastPoint(BaseModel):
    """One predicted price point."""

    forecast_date: str = Field(..., description="YYYY-MM-DD")
    price: float = Field(..., description="Predicted price in $/MWh")
    low: float | None = Field(default=None, description="Lower confidence bound")
    high: float | None = Field(default=None, description="Upper confidence bound")


class PredictResponse(BaseModel):
    """Prediction result with optional Chronos cross-validation."""

    mode: Literal["single", "multi"] = ...
    series: list[ForecastPoint] = Field(..., description="Forecast series")
    chronos: list[ForecastPoint] | None = Field(
        default=None, description="Chronos-T5 cross-validation series (if enabled)"
    )
    meta: dict[str, Any] = Field(default_factory=dict, description="Extra metadata")


# ============================================================
# RAG
# ============================================================
class RagSearchRequest(BaseModel):
    """Hybrid RAG retrieval + summarization request."""

    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of chunks to retrieve")


class RagChunk(BaseModel):
    """One retrieved knowledge chunk."""

    content: str = Field(..., description="Chunk text")
    source: str = Field(default="", description="Source document")


class RagSearchResponse(BaseModel):
    """RAG result: summary plus retrieved chunks (for citation tracing)."""

    summary: str = Field(default="", description="LLM summary of retrieved context")
    chunks: list[RagChunk] = Field(default_factory=list, description="Raw retrieved chunks")


# ============================================================
# Tools registry
# ============================================================
class ToolInfo(BaseModel):
    """Metadata for one registered tool."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description / docstring")
    args_schema: dict[str, Any] = Field(
        default_factory=dict, description="JSON-schema of tool arguments"
    )


class ToolsResponse(BaseModel):
    """Registry dump of all tools the agents can call."""

    count: int = Field(..., description="Number of registered tools")
    tools: list[ToolInfo] = Field(..., description="Tool metadata list")


# ============================================================
# Common
# ============================================================
class ErrorResponse(BaseModel):
    """Uniform error envelope."""

    error: dict[str, str] = Field(
        ..., description='Stable shape: {"code": "...", "message": "..."}'
    )


class HealthResponse(BaseModel):
    """Service health + capability probe."""

    status: Literal["ok", "degraded"] = ...
    service: str = "electricity-price-intelligence-api"
    version: str = "1.0.0"
    agents: list[str] = Field(default_factory=list, description="Available agent modes")
    rag_ready: bool = Field(default=False, description="Whether the RAG knowledge base is loaded")
