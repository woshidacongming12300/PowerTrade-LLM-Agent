"""Chainlit frontend for the Electricity Price Intelligence multi-agent system."""

import asyncio
import json
import os
import sqlite3
import sys

# GRIDSTATUS_ENABLED is set via .env — do NOT force-enable here

import chainlit as cl
import chainlit.data as cl_data
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
import plotly.graph_objects as go
from chainlit.input_widget import Select

_db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".chainlit", "chainlit.db")

def _init_chainlit_db(db_path: str):
    """Create tables required by Chainlit SQLAlchemyDataLayer."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            "id" TEXT PRIMARY KEY,
            "identifier" TEXT NOT NULL,
            "createdAt" TEXT,
            "metadata" TEXT
        );
        CREATE TABLE IF NOT EXISTS threads (
            "id" TEXT PRIMARY KEY,
            "createdAt" TEXT,
            "name" TEXT,
            "userId" TEXT,
            "userIdentifier" TEXT,
            "tags" TEXT,
            "metadata" TEXT
        );
        CREATE TABLE IF NOT EXISTS steps (
            "id" TEXT PRIMARY KEY,
            "name" TEXT NOT NULL,
            "type" TEXT NOT NULL,
            "threadId" TEXT NOT NULL,
            "parentId" TEXT,
            "command" TEXT,
            "modes" TEXT,
            "streaming" INTEGER DEFAULT 0,
            "waitForAnswer" INTEGER DEFAULT 0,
            "isError" INTEGER DEFAULT 0,
            "metadata" TEXT,
            "tags" TEXT,
            "input" TEXT,
            "output" TEXT,
            "createdAt" TEXT,
            "start" TEXT,
            "end" TEXT,
            "generation" TEXT,
            "showInput" TEXT,
            "defaultOpen" INTEGER DEFAULT 0,
            "autoCollapse" INTEGER DEFAULT 0,
            "language" TEXT,
            "icon" TEXT,
            "feedback" TEXT
        );
        CREATE TABLE IF NOT EXISTS feedbacks (
            "id" TEXT PRIMARY KEY,
            "forId" TEXT NOT NULL,
            "value" INTEGER NOT NULL,
            "threadId" TEXT,
            "comment" TEXT
        );
        CREATE TABLE IF NOT EXISTS elements (
            "id" TEXT PRIMARY KEY,
            "threadId" TEXT,
            "type" TEXT,
            "chainlitKey" TEXT,
            "path" TEXT,
            "url" TEXT,
            "objectKey" TEXT,
            "name" TEXT,
            "display" TEXT DEFAULT 'side',
            "size" TEXT,
            "language" TEXT,
            "page" INTEGER,
            "props" TEXT,
            "autoPlay" TEXT,
            "playerConfig" TEXT,
            "forId" TEXT,
            "mime" TEXT
        );
    """)
    conn.commit()
    conn.close()

_init_chainlit_db(_db_path)

@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(
        conninfo=f"sqlite+aiosqlite:///{_db_path}"
    )

# --- Simple password auth for conversation history sidebar ---
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    share_pwd = os.environ.get("CHAINLIT_AUTH_PASSWORD", "epf2026")
    if password == share_pwd:
        return cl.User(identifier=username, metadata={"role": "user"})
    return None

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from agent.react_agent import ReactAgent
from agent.langgraph_agent import run_langgraph_chainlit
from agent.plan_execute_agent import run_plan_execute_chainlit
from utils.i18n import t


# ============================================================
# Chart helper
# ============================================================
def build_chart(forecast_series_json: str, lang: str) -> go.Figure | None:
    try:
        series = json.loads(forecast_series_json)
        if not series:
            return None
        dates = [r["forecast_date"] for r in series]
        prices = [r["price"] for r in series]
        y_title = "电价 ($/MWh)" if lang == "zh" else "Price ($/MWh)"
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=prices, mode="lines+markers",
            line=dict(color="#6366f1", width=2.5),
            marker=dict(size=6, color="#818cf8"),
            fill="tozeroy",
            fillcolor="rgba(99,102,241,0.1)",
            name=y_title,
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=30, b=20),
            height=320,
            xaxis_title="" if lang == "zh" else "Date",
            yaxis_title=y_title,
            font=dict(color="#a1a1aa"),
            hovermode="x unified",
        )
        return fig
    except Exception:
        return None


# ============================================================
# Chat Settings (persistent language & mode toggles)
# ============================================================
@cl.on_chat_start
async def on_chat_start():
    # Init session
    cl.user_session.set("lang", "zh")
    cl.user_session.set("mode", "langgraph")
    cl.user_session.set("agent", ReactAgent())
    cl.user_session.set("history", [])

    lang = cl.user_session.get("lang")
    mode = cl.user_session.get("mode")

    # 1. Send persistent ChatSettings (gear icon in header → settings panel)
    settings = cl.ChatSettings([
        Select(
            id="language",
            label="Language / 语言",
            values=["zh", "en"],
            initial_value="zh",
        ),
        Select(
            id="agent_mode",
            label="Agent Mode / 智能体模式",
            items={
                "LangGraph Agent (Dynamic)": "langgraph",
                "Plan-Execute Agent (Structured)": "plan_execute",
                "ReAct Agent (Quick Chat)": "react",
                "Supervisor Agent (Multi-Agent)": "supervisor",
            },
            initial_value="langgraph",
        ),
    ])
    await settings.send()

    # 2. Set starters (quick action suggestions at the bottom)
    pass

    # 3. Welcome message (compact)
    mode_labels = {
        "langgraph": "LangGraph Agent (Dynamic + Reflection)",
        "plan_execute": "Plan-Execute Agent (Structured)",
        "react": "ReAct Agent (Quick Chat)",
        "supervisor": "Supervisor Agent (Multi-Agent)",
    }
    mode_label = mode_labels.get(mode, "LangGraph Agent (Dynamic)")
    welcome = (
        f"### ⚡ {t('app_title', lang)}\n"
        f"{t('app_subtitle', lang)}\n\n"
        f"> **Mode:** {mode_label}  |  **Language:** {'中文' if lang == 'zh' else 'English'}\n"
        f"> **Tools:** 9 tools (Price Forecast + Trading Advice + Alerts + Briefing + Weather + RAG + ...)\n\n"
        f"{t('welcome_text', lang)}"
    )
    await cl.Message(content=welcome).send()


# ============================================================
# Chat resume handler — restore session when clicking past conversations
# ============================================================
@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    meta = thread.get("metadata", {}) or {}
    lang = meta.get("lang", "zh")
    mode = meta.get("mode", "langgraph")

    cl.user_session.set("lang", lang)
    cl.user_session.set("mode", mode)
    cl.user_session.set("agent", ReactAgent())

    # Rebuild conversation history from stored steps
    history = []
    for step in thread.get("steps", []):
        stype = step.get("type", "")
        output = step.get("output", "") or ""
        if not output:
            continue
        if stype == "user_message":
            history.append({"role": "user", "content": output})
        elif stype in ("assistant_message", "run"):
            history.append({"role": "assistant", "content": output[:300]})
    if len(history) > 20:
        history = history[-20:]
    cl.user_session.set("history", history)


# ============================================================
# Settings update handler
# ============================================================
@cl.on_settings_update
async def on_settings_update(settings):
    new_lang = settings.get("language", "zh")
    new_mode = settings.get("agent_mode", "langgraph")

    old_lang = cl.user_session.get("lang", "zh")
    old_mode = cl.user_session.get("mode", "langgraph")

    changed = False

    if new_lang != old_lang:
        cl.user_session.set("lang", new_lang)
        changed = True

    if new_mode != old_mode:
        cl.user_session.set("mode", new_mode)
        changed = True

    if changed:
        lang = new_lang
        mode = new_mode

        mode_labels = {
            "langgraph": "LangGraph Agent (Dynamic)",
            "plan_execute": "Plan-Execute Agent (Structured)",
            "react": "ReAct Agent (Quick Chat)",
            "supervisor": "Supervisor Agent (Multi-Agent)",
        }
        mode_label = mode_labels.get(mode, "LangGraph Agent (Dynamic)")
        msg = (
            f"Language: **{'中文' if lang == 'zh' else 'English'}**  |  "
            f"Mode: **{mode_label}**"
        )
        await cl.Message(content=msg).send()


# ============================================================
# ReAct Agent
# ============================================================
async def run_react(message: cl.Message, lang: str):
    agent: ReactAgent = cl.user_session.get("agent")
    query = message.content

    msg = cl.Message(content="")
    await msg.send()

    full_text = ""
    try:
        loop = asyncio.get_event_loop()
        gen = agent.execute_stream(query)

        def _next():
            try:
                return next(gen)
            except StopIteration:
                return None

        while True:
            chunk = await loop.run_in_executor(None, _next)
            if chunk is None:
                break
            full_text += chunk
            await msg.stream_token(chunk)
    except Exception as e:
        await msg.stream_token(f"\n\nError: {e}")

    await msg.update()

    # Try extract chart
    try:
        import re as _re
        json_match = _re.search(r'\{.*"series".*\}', full_text, _re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            if "series" in parsed and parsed["series"]:
                fig = build_chart(json.dumps(parsed["series"], ensure_ascii=False), lang)
                if fig:
                    await cl.Message(content="", elements=[cl.Plotly(name="chart", figure=fig, display="inline")]).send()
    except Exception:
        pass

    return full_text


# ============================================================
# LangGraph Dynamic Agent
# ============================================================
async def run_langgraph(message: cl.Message, lang: str):
    query = message.content
    history = cl.user_session.get("history", [])

    loop = asyncio.get_event_loop()
    gen = run_langgraph_chainlit(query, history)

    def _next():
        try:
            return next(gen)
        except StopIteration:
            return None

    final_response = ""
    forecast_series_json = "[]"
    tool_steps: dict[str, cl.Step] = {}
    pending_tools: set[str] = set()

    async with cl.Step(name="Agent", type="tool") as parent:
        parent.output = "Thinking..."

        while True:
            result = await loop.run_in_executor(None, _next)
            if result is None:
                break

            node = result.get("node", "")
            content = result.get("content", "") or ""
            status = result.get("status", "done")

            if node == "response":
                final_response = content
                parent.output = "Done"
            elif status == "running":
                tool_name = result.get("title", node)
                pending_tools.add(node)
                step = cl.Step(
                    name=f"🔧 {tool_name}",
                    type="tool",
                    parent_id=parent.id,
                )
                step.output = content
                await step.send()
                tool_steps[node] = step
            elif status == "done" and node in pending_tools:
                pending_tools.discard(node)
                if node in tool_steps:
                    s = tool_steps[node]
                    s.output = content[:500] + ("..." if len(content) > 500 else "")
                    await s.update()

            # Try to extract forecast series for chart
            if content and "forecast_date" in content:
                try:
                    import re as _re
                    matches = _re.findall(r'"forecast_date":\s*"([^"]+)"\s*,\s*"price":\s*([\d.]+)', content)
                    if matches:
                        series = [{"forecast_date": d, "price": float(p)} for d, p in matches]
                        if series:
                            forecast_series_json = json.dumps(series)
                except Exception:
                    pass

    if final_response:
        await cl.Message(content=final_response).send()

    if forecast_series_json and forecast_series_json != "[]":
        fig = build_chart(forecast_series_json, lang)
        if fig:
            await cl.Message(
                content="",
                elements=[cl.Plotly(name="forecast_chart", figure=fig, display="inline")],
            ).send()

    return final_response


# ============================================================
# Plan-Execute Agent
# ============================================================
async def run_plan_execute(message: cl.Message, lang: str):
    query = message.content

    loop = asyncio.get_event_loop()
    gen = run_plan_execute_chainlit(query)

    def _next():
        try:
            return next(gen)
        except StopIteration:
            return None

    final_response = ""
    forecast_series_json = "[]"
    tool_steps: dict[str, cl.Step] = {}

    async with cl.Step(name="Plan-Execute", type="tool") as parent:
        parent.output = "Planning..."

        while True:
            result = await loop.run_in_executor(None, _next)
            if result is None:
                break

            node = result.get("node", "")
            content = result.get("content", "") or ""
            status = result.get("status", "done")
            title = result.get("title", node)

            if node == "response":
                final_response = content
                parent.output = "Done"
            elif node == "plan" and status == "running":
                parent.output = content
            elif status == "done" and node != "plan":
                step = cl.Step(
                    name=f"📋 {title}",
                    type="tool",
                    parent_id=parent.id,
                )
                step.output = content[:400] + ("..." if len(content) > 400 else "")
                await step.send()

            # Extract forecast series
            if content and "forecast_date" in content:
                try:
                    import re as _re
                    matches = _re.findall(r'"forecast_date":\s*"([^"]+)"\s*,\s*"price":\s*([\d.]+)', content)
                    if matches:
                        series = [{"forecast_date": d, "price": float(p)} for d, p in matches]
                        if series:
                            forecast_series_json = json.dumps(series)
                except Exception:
                    pass

    if final_response:
        await cl.Message(content=final_response).send()

    if forecast_series_json and forecast_series_json != "[]":
        fig = build_chart(forecast_series_json, lang)
        if fig:
            await cl.Message(
                content="",
                elements=[cl.Plotly(name="forecast_chart", figure=fig, display="inline")],
            ).send()

    return final_response


# ============================================================
# Supervisor Multi-Agent
# ============================================================
async def run_supervisor_chainlit(message: cl.Message, lang: str):
    """Run the supervisor multi-agent flow with step-by-step rendering.

    Shows: Coordinator routing decision -> specialist agent -> final synthesis.
    """
    from agent.supervisor_agent import run_supervisor
    from memory.service import MemoryService

    query = message.content
    user_id = message.author if message.author else "default"

    # Extract & persist user preferences from this message (long-term memory).
    memory = MemoryService(user_id)
    prefs_found = memory.extract_and_store(query)

    async with cl.Step(name="🧠 Supervisor", type="tool") as parent:
        parent.output = "Coordinator 正在路由..."
        await parent.update()

        final_response = await asyncio.get_event_loop().run_in_executor(
            None, run_supervisor, query
        )

        parent.output = "✅ 综合完成"
        await parent.update()

    if final_response:
        await cl.Message(content=final_response).send()

    # Show what long-term memory learned (observability touch).
    prefs = memory.preferences()
    if prefs:
        pref_lines = ", ".join(f"{k}={v}" for k, v in prefs.items())
        await cl.Message(
            content=f"🧠 **长期记忆已更新**: {pref_lines}\n"
                    f"（跨会话记住你的偏好，切换模式后依然生效）",
        ).send()

    return final_response


# ============================================================
# Main message handler
# ============================================================
@cl.on_message
async def on_message(message: cl.Message):
    lang = cl.user_session.get("lang", "zh")
    mode = cl.user_session.get("mode", "langgraph")

    # Save user message to conversation history
    history = cl.user_session.get("history", [])
    history.append({"role": "user", "content": message.content})

    if mode == "langgraph":
        assistant_content = await run_langgraph(message, lang)
        if assistant_content:
            history.append({"role": "assistant", "content": assistant_content[:300]})
    elif mode == "plan_execute":
        assistant_content = await run_plan_execute(message, lang)
        if assistant_content:
            history.append({"role": "assistant", "content": assistant_content[:300]})
    elif mode == "supervisor":
        assistant_content = await run_supervisor_chainlit(message, lang)
        if assistant_content:
            history.append({"role": "assistant", "content": assistant_content[:300]})
    else:
        assistant_content = await run_react(message, lang)
        if assistant_content:
            history.append({"role": "assistant", "content": assistant_content[:300]})

    # Keep only last 10 turns
    if len(history) > 20:
        history = history[-20:]
    cl.user_session.set("history", history)

    # Persist lang/mode to thread metadata for chat resume
    dl = cl_data.get_data_layer()
    if dl:
        await dl.update_thread(
            thread_id=message.thread_id,
            metadata={"lang": lang, "mode": mode},
        )
