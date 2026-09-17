"""Shared dependencies: API-key auth, agent/tool registries.

Keeps auth and capability wiring in one place so routers stay thin.
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException, status

from utils.config_handler import get_abs_path

# Optional API key auth. If API_KEY is set in .env, every request must carry
# `X-API-Key: <key>`. Leave unset for local development.
_API_KEY = os.environ.get("API_KEY", "").strip()


def get_api_key_dependency():
    """Return a FastAPI dependency that enforces API-key auth when configured."""

    if not _API_KEY:
        # Auth disabled — return a dependency that always passes.
        async def _noop() -> None:
            return None

        return _noop

    async def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if x_api_key != _API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"code": "unauthorized", "message": "Invalid or missing API key"},
            )

    return _require_api_key


# Re-export for routers
require_api_key = get_api_key_dependency()
