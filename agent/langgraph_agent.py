"""LangGraph Agent — LLM dynamically decides which tools to call and when to respond.
This is a true agent graph, NOT a fixed pipeline."""

import json
import os
import re
import sys
from typing import TypedDict, Annotated, Sequence, Literal
import operator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END

from agent.tools.agent_tools import (
    rag_summarize,
    get_texas_weather,
    predict_electricity_price,
    forecast_electricity_prices,
    get_chronos_prediction,
    get_data_freshness,
    get_trading_advice,
    check_price_alerts,
    generate_daily_briefing,
    get_current_date,
)
from datetime import date as _date_today
from model.factory import chat_model

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from algorithms.epf_predictor import predict_future_prices  # noqa: E402

# ============================================================
# 0. Tools & Tool Map
# ============================================================
TOOLS = [
    get_current_date,
    get_texas_weather,
    predict_electricity_price,
    forecast_electricity_prices,
    get_chronos_prediction,
    rag_summarize,
    get_data_freshness,
    get_trading_advice,
    check_price_alerts,
    generate_daily_briefing,
]

TOOL_BY_NAME = {t.name: t for t in TOOLS}

def _system_prompt() -> str:
    return f"""You are an intelligent electricity market assistant. Today is {_date_today.today().isoformat()}.

You have access to these tools:

- get_current_date: Get today's real date. Call when user asks "what date is it today".
- get_texas_weather: 7-day weather forecast for MISO/ERCOT regions. Input 'indiana 7天预报' or 'texas forecast'.
- predict_electricity_price: Single-day electricity price prediction (LSTM multivariate model).
- forecast_electricity_prices: Multi-day price forecast with confidence intervals and Chronos-T5 cross-validation. Returns JSON.
- get_chronos_prediction: Alternative model (Chronos-T5) for cross-validation.
- rag_summarize: Search knowledge base for electricity market rules, policies, regulations.
- get_data_freshness: Check training data and live data source status.
- get_trading_advice: Generate BUY/SELL/HOLD trading recommendation based on price trend. Input: target_date like "3天后" or "7天后".
- check_price_alerts: Scan next 7 days for price anomaly risks (red/orange/yellow alerts).
- generate_daily_briefing: Generate a full structured daily market briefing (weather + forecast + alerts + trading advice).

Rules:
1. For casual chat (greetings, thanks, "who are you", etc.), reply directly WITHOUT calling any tools.
2. For weather questions, call get_texas_weather.
3. For price forecast questions, call forecast_electricity_prices (preferred, gives confidence intervals) or predict_electricity_price (single day).
4. For trading/investment advice, call get_trading_advice.
5. For risk/alert questions, call check_price_alerts.
6. For "daily briefing" or "market report" requests, call generate_daily_briefing.
7. For policy/rule questions, call rag_summarize.
8. Use the SAME language as the user (Chinese or English).
9. When you have enough information, synthesize a clear, concise answer. Present multi-day forecasts as tables."""

REFLECTION_PROMPT = """You are a quality reviewer for an electricity market AI assistant. Critically evaluate the assistant's response below.

Check for:
1. Factual accuracy: Are the numbers/tool outputs correctly cited? Any contradictions?
2. Completeness: Did the response address ALL parts of the user's question?
3. Clarity: Is the response well-structured and easy to understand?
4. Actionability: Does the response give the user useful, concrete information?

User question: {user_query}
Assistant response: {assistant_response}

Reply with ONLY one word:
- "PASS" if the response is good enough (minor issues are OK)
- "REVISE: <specific instruction>" if the response has a significant issue that must be fixed

Examples:
- PASS
- REVISE: The price forecast is missing — you called the tool but didn't include the numbers in your response.
- REVISE: You answered about weather but the user also asked about price impact. Address both."""


# ============================================================
# 1. Agent State
# ============================================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ============================================================
# 2. Agent Node — LLM decides what to do
# ============================================================
def agent_node(state: AgentState):
    """The LLM decides: call a tool, or respond directly."""
    messages = state["messages"]

    # Ensure system prompt is present
    has_system = any(getattr(m, "type", "") == "system" for m in messages)
    if not has_system:
        full_messages = [SystemMessage(content=_system_prompt())] + list(messages)
    else:
        full_messages = list(messages)

    # Bind tools to model
    try:
        model_with_tools = chat_model.bind_tools(TOOLS)
        response = model_with_tools.invoke(full_messages)
    except Exception:
        # Fallback: model doesn't support native tool calling, use JSON prompt
        response = _fallback_agent(full_messages)

    return {"messages": [response]}


def _fallback_agent(messages):
    """Fallback when model doesn't support bind_tools natively."""
    history_parts = []
    for msg in messages:
        c = getattr(msg, "content", None)
        if isinstance(c, str) and c.strip():
            t = getattr(msg, "type", "human")
            role = "User" if t == "human" else "Assistant" if t == "ai" else t
            history_parts.append(f"{role}: {c[:500]}")

    tool_descs = "\n".join(f"- {t.name}: {t.description}" for t in TOOLS)

    prompt = f"""You are an electricity market AI assistant.

Available tools:
{tool_descs}

Conversation:
{chr(10).join(history_parts)}

Decide your next action. Reply with JSON:
- To respond directly: {{"action": "respond", "content": "your reply here"}}
- To call a tool: {{"action": "tool", "tool": "tool_name", "input": "the input"}}

Choose ONE action. For casual chat, use "respond". For domain queries, use the appropriate tool."""

    response = chat_model.invoke(prompt)
    raw = response.content.strip()

    # Parse JSON
    try:
        import re as _re
        json_match = _re.search(r'\{[^{}]*\}', raw)
        if json_match:
            parsed = json.loads(json_match.group(0))
            action = parsed.get("action", "respond")

            if action == "respond":
                return AIMessage(content=parsed.get("content", raw))

            if action == "tool":
                tool_name = parsed.get("tool", "")
                tool_input = parsed.get("input", "")
                if tool_name in TOOL_BY_NAME:
                    # Create a fake tool call message
                    result = AIMessage(
                        content=f"Calling {tool_name}...",
                        tool_calls=[{
                            "name": tool_name,
                            "args": {"query" if "weather" in tool_name or "summarize" in tool_name or "freshness" in tool_name else "target_date": tool_input},
                            "id": "fallback_1",
                        }]
                    )
                    # Mark it with additional_kwargs for routing
                    result.additional_kwargs = {"tool_calls": [{
                        "function": {"name": tool_name, "arguments": json.dumps({"query" if "weather" in tool_name or "summarize" in tool_name or "freshness" in tool_name else "target_date": tool_input})},
                        "id": "fallback_1",
                    }]}
                    return result
    except Exception:
        pass

    return AIMessage(content=raw)


# ============================================================
# 3. Tool Execution Node
# ============================================================
def tool_node(state: AgentState):
    """Execute the tool calls from the last AI message."""
    last_message = state["messages"][-1]

    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls:
        # Check additional_kwargs (fallback)
        extra = getattr(last_message, "additional_kwargs", {}) or {}
        tool_calls = extra.get("tool_calls", [])

    results = []
    for tc in tool_calls:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})

        if tool_name not in TOOL_BY_NAME:
            results.append(ToolMessage(
                content=f"Tool '{tool_name}' not found. Available: {list(TOOL_BY_NAME.keys())}",
                tool_call_id=tc.get("id", "unknown"),
            ))
            continue

        tool = TOOL_BY_NAME[tool_name]
        print(f"[Agent] Calling tool: {tool_name} with {tool_args}")

        try:
            # Tools expect a string input; extract it from args
            if tool_args:
                # Get the first string value from args
                input_val = list(tool_args.values())[0] if isinstance(tool_args, dict) else str(tool_args)
            else:
                input_val = ""

            result = tool.invoke(input_val)
            results.append(ToolMessage(
                content=str(result),
                tool_call_id=tc.get("id", "unknown"),
            ))
        except Exception as e:
            results.append(ToolMessage(
                content=f"Error executing {tool_name}: {e}",
                tool_call_id=tc.get("id", "unknown"),
            ))

    return {"messages": results}


# ============================================================
# 4. Reflection Node — self-critique for quality assurance
# ============================================================
MAX_REFLECTION_PASSES = 1


def _count_reflection_passes(messages) -> int:
    """Count how many reflection feedback rounds have occurred."""
    count = 0
    for m in messages:
        content = getattr(m, "content", "")
        if isinstance(content, str) and "[REFLECTION FEEDBACK]" in content:
            count += 1
    return count


def reflect_node(state: AgentState):
    """Critique the agent's response and either pass or request revision."""
    messages = state["messages"]

    # Find user query and latest assistant response
    user_query = ""
    assistant_response = ""
    for m in messages:
        if isinstance(m, HumanMessage) and "[REFLECTION FEEDBACK]" not in str(m.content):
            user_query = m.content
        elif isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
            assistant_response = m.content

    critique_prompt = REFLECTION_PROMPT.format(
        user_query=user_query, assistant_response=assistant_response[:2000]
    )

    critique_msg = chat_model.invoke(critique_prompt)
    verdict = critique_msg.content.strip()
    print(f"[Reflection] Verdict: {verdict[:150]}")

    if verdict.upper().startswith("PASS"):
        return {"messages": []}

    # Needs revision — inject feedback
    revision_instruction = verdict.replace("REVISE:", "").strip()
    feedback = HumanMessage(
        content=f"[REFLECTION FEEDBACK] Your previous response needs improvement: {revision_instruction}\n"
                f"Please revise your response to address this issue. Do NOT call any tools again — "
                f"just synthesize a better answer based on the tool results you already have."
    )
    return {"messages": [feedback]}


# ============================================================
# 5. Routers
# ============================================================
def should_continue(state: AgentState) -> Literal["tools", "reflect", "__end__"]:
    """After agent: call tools, reflect on final response, or end."""
    last_message = state["messages"][-1]

    # Agent just processed reflection feedback and produced revised response
    content = getattr(last_message, "content", "")
    if isinstance(content, str) and "[REFLECTION FEEDBACK]" in content:
        # This is the feedback message itself — route to agent
        return "tools"  # We'll add a direct edge "reflect"→"agent" instead

    # Has pending tool calls?
    tc = getattr(last_message, "tool_calls", None)
    if tc:
        return "tools"
    extra = getattr(last_message, "additional_kwargs", {}) or {}
    if extra.get("tool_calls"):
        return "tools"

    # No tool calls — this is a final response. Reflect if not already done.
    if _count_reflection_passes(state["messages"]) >= MAX_REFLECTION_PASSES:
        return END

    return "reflect"


def after_reflect(state: AgentState) -> Literal["agent", "__end__"]:
    """After reflection: go back to agent for revision, or end."""
    last_message = state["messages"][-1]
    content = getattr(last_message, "content", "")
    if isinstance(content, str) and "[REFLECTION FEEDBACK]" in content:
        return "agent"
    return END


# ============================================================
# 6. Build the Graph (with Reflection)
# ============================================================
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("reflect", reflect_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "reflect": "reflect", END: END},
)

workflow.add_conditional_edges(
    "reflect",
    after_reflect,
    {"agent": "agent", END: END},
)

workflow.add_edge("tools", "agent")

langgraph_app = workflow.compile()


# ============================================================
# 7. Streaming wrappers (backward-compatible)
# ============================================================
def run_langgraph_stream(query: str, history: list | None = None):
    """Stream the agent's response. Yields text chunks."""
    messages = _build_messages(query, history)
    initial_state = {"messages": messages}

    for output in langgraph_app.stream(initial_state, stream_mode="values"):
        if "messages" not in output:
            continue
        latest = output["messages"][-1]
        # Only yield AI responses (not tool messages)
        if isinstance(latest, AIMessage) and latest.content:
            # Only yield the final response (no tool calls)
            if not getattr(latest, "tool_calls", None):
                yield latest.content


def run_langgraph_chainlit(query: str, history: list | None = None):
    """Generator for Chainlit: yields structured dicts for step-by-step UI rendering."""
    messages = _build_messages(query, history)
    initial_state = {"messages": messages}

    pending_tool_names: list[str] = []
    step_idx = 0

    for output in langgraph_app.stream(initial_state, stream_mode="values"):
        if "messages" not in output:
            continue
        latest = output["messages"][-1]

        if isinstance(latest, AIMessage):
            tc = getattr(latest, "tool_calls", None)
            extra = getattr(latest, "additional_kwargs", {}) or {}
            if not tc:
                tc = extra.get("tool_calls", [])

            if tc:
                for call in tc:
                    tool_name = call.get("name", "unknown")
                    pending_tool_names.append(tool_name)
                    step_idx += 1
                    yield {
                        "type": "step",
                        "node": tool_name,
                        "title": tool_name.replace("get_", "").replace("_", " ").title(),
                        "status": "running",
                        "content": f"Calling {tool_name}...",
                    }
            else:
                yield {
                    "type": "step",
                    "node": "response",
                    "title": "Response",
                    "status": "done",
                    "content": latest.content,
                }

        elif isinstance(latest, ToolMessage):
            tool_name = pending_tool_names.pop(0) if pending_tool_names else f"tool_{step_idx}"
            yield {
                "type": "step",
                "node": tool_name,
                "title": tool_name.replace("get_", "").replace("_", " ").title(),
                "status": "done",
                "content": str(latest.content)[:500],
            }


def _build_messages(query: str, history: list | None = None):
    messages = []
    if history:
        for h in history:
            role = h.get("role", "user")
            content = h.get("content", "")
            if content.strip():
                messages.append(
                    HumanMessage(content=content) if role == "user"
                    else AIMessage(content=content)
                )
    messages.append(HumanMessage(content=query))
    return messages


# ============================================================
# 8. Test
# ============================================================
if __name__ == "__main__":
    print("LangGraph Dynamic Agent Starting...\n" + "=" * 50)

    tests = [
        "你好",
        "今天天气怎么样",
        "预测一下明天的电价",
    ]

    for q in tests:
        print(f"\n>>> User: {q}")
        for chunk in run_langgraph_stream(q):
            print(chunk, end="", flush=True)
        print("\n" + "-" * 40)
