"""Health / capability probe endpoint."""

from __future__ import annotations

import os

from fastapi import APIRouter

from api.schemas import HealthResponse

router = APIRouter(prefix="/api/health", tags=["health"])


@router.get("", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report service status, available agent modes, and RAG readiness."""
    rag_ready = os.path.exists(os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "chroma_db_v2",
    ))
    return HealthResponse(
        status="ok",
        agents=["langgraph", "plan_execute", "react", "supervisor"],
        rag_ready=rag_ready,
    )
