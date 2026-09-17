"""FastAPI entry point for the Electricity Price Intelligence agent service.

Run:
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8001 --reload
or:
    python api/main.py

Interactive docs (auto-generated from the Pydantic contracts):
    http://127.0.0.1:8001/docs
"""

from __future__ import annotations

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat, forecast, health, memory, rag, tools

app = FastAPI(
    title="Electricity Price Intelligence — Agent API",
    description=(
        "REST + SSE service exposing the multi-agent electricity price forecasting "
        "system: three agent paradigms, hybrid RAG, and LSTM/Chronos dual-track prediction."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow the Chainlit/Streamlit frontends and local dev tools to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to specific origins in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Uniform error envelope: {"error": {"code": ..., "message": ...}}
@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # noqa: ANN001
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_error", "message": str(exc)}},
    )


app.include_router(health.router)
app.include_router(tools.router)
app.include_router(chat.router)
app.include_router(forecast.router)
app.include_router(rag.router)
app.include_router(memory.router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "electricity-price-intelligence-api",
        "docs": "/docs",
        "endpoints": [
            "GET  /api/health",
            "GET  /api/tools",
            "POST /api/chat",
            "POST /api/chat/stream (SSE)",
            "POST /api/predict",
            "POST /api/predict/chronos",
            "POST /api/rag/search",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("API_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
