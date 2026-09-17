"""RAG endpoints: hybrid retrieval + summarization with citation chunks."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.deps import require_api_key
from api.schemas import RagChunk, RagSearchRequest, RagSearchResponse

from rag.rag_service import RagSummarizeService

router = APIRouter(prefix="/api/rag", tags=["rag"], dependencies=[Depends(require_api_key)])

# Lazy singleton — building RagSummarizeService loads embedding/chat models,
# which must not happen at import time in credential-less environments (CI).
_rag: "RagSummarizeService | None" = None


def _get_rag() -> RagSummarizeService:
    global _rag
    if _rag is None:
        _rag = RagSummarizeService()
    return _rag


@router.post("/search", response_model=RagSearchResponse)
def rag_search(req: RagSearchRequest) -> RagSearchResponse:
    """Run hybrid RAG (ChromaDB + BM25 + RRF) and summarize the retrieved context."""
    rag_svc = _get_rag()
    try:
        summary = rag_svc.rag_summarize(req.query)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail={
            "code": "rag_error",
            "message": str(exc),
        }) from exc

    # Retrieve raw chunks for citation tracing (best-effort; may be empty).
    chunks: list[RagChunk] = []
    try:
        retriever = rag_svc.retriever if hasattr(rag_svc, "retriever") else None
        if retriever is None:
            from rag.vector_store import VectorStoreService
            retriever = VectorStoreService().get_retriever()
        for doc in retriever.invoke(req.query)[: req.top_k]:
            src = ""
            meta = getattr(doc, "metadata", {}) or {}
            if isinstance(meta, dict):
                src = str(meta.get("source", ""))
            chunks.append(RagChunk(content=getattr(doc, "page_content", "")[:500], source=src))
    except Exception:  # noqa: BLE001 — chunk tracing is best-effort
        pass

    return RagSearchResponse(summary=summary, chunks=chunks)
