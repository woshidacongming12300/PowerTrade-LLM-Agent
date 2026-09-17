"""Tools registry endpoint — introspect every tool the agents can call."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from api.deps import require_api_key
from api.schemas import ToolInfo, ToolsResponse

from agent.langgraph_agent import TOOLS

router = APIRouter(prefix="/api/tools", tags=["tools"], dependencies=[Depends(require_api_key)])


def _tool_args_schema(tool) -> dict[str, Any]:
    """Best-effort extraction of a tool's argument JSON schema."""
    try:
        args = getattr(tool, "args", None)
        if args is not None:
            return args.schema()
        fn = getattr(tool, "func", None) or tool
        import inspect

        sig = inspect.signature(fn)
        props = {}
        for name, param in sig.parameters.items():
            if name in ("self", "kwargs", "args"):
                continue
            ann = param.annotation
            props[name] = {
                "type": "string",
                "default": None if param.default is inspect.Parameter.empty else str(param.default),
            }
            if param.default is not inspect.Parameter.empty:
                props[name]["default"] = (
                    param.default if not isinstance(param.default, str) else param.default
                )
        return {"type": "object", "properties": props}
    except Exception:  # noqa: BLE001
        return {}


@router.get("", response_model=ToolsResponse)
def list_tools() -> ToolsResponse:
    """Return the registry of tools available to the agents."""
    tools = [
        ToolInfo(
            name=t.name,
            description=(getattr(t, "description", "") or ""),
            args_schema=_tool_args_schema(t),
        )
        for t in TOOLS
    ]
    return ToolsResponse(count=len(tools), tools=tools)
