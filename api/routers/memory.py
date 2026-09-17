"""Memory endpoints — inspect the long-term memory store (prefs / forecast history)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.deps import require_api_key
from memory.service import MemoryService

router = APIRouter(prefix="/api/memory", tags=["memory"], dependencies=[Depends(require_api_key)])


class MemoryRequest(BaseModel):
    user_id: str = Field(default="default", max_length=128)


class MemoryResponse(BaseModel):
    user_id: str
    preferences: dict[str, str] = {}
    recent_predictions: list[dict] = []
    conversation_summaries: list[dict] = []


@router.post("", response_model=MemoryResponse)
def get_memory(req: MemoryRequest) -> MemoryResponse:
    """Return stored preferences, recent predictions, and conversation summaries."""
    svc = MemoryService(req.user_id)
    return MemoryResponse(
        user_id=req.user_id,
        preferences=svc.preferences(),
        recent_predictions=svc.recent_predictions(10),
        conversation_summaries=svc.store.get_conversation_summaries(req.user_id, 5),
    )
