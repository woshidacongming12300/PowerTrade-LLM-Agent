"""Plan-and-Execute Agent — decomposes complex queries into subtasks,
executes them step by step, then synthesizes a final response.

This is the "Plan-and-Solve" pattern (Wang et al., 2023) adapted for
the electricity market domain with tool-augmented execution.
"""

import json
import os
import sys
from typing import TypedDict, Annotated, Sequence, Literal
import operator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

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

PLANNER_PROMPT = f"""You are an electricity market AI planner. Today is {_date_today.today().isoformat()}. Given a user query, decompose it into a sequence of concrete subtasks.

Available tools:
- get_current_date: Get today's date.
- get_texas_weather: Real-time or 7-day forecast weather for MISO/ERCOT regions.
- predict_electricity_price: Predict electricity price for a single date (LSTM multivariate).
- forecast_electricity_prices: Multi-day electricity price forecast with confidence intervals.
- get_chronos_prediction: Alternative model (Chronos-T5) for cross-validation.
- rag_summarize: Search knowledge base for market rules, policies, regulations.
- get_data_freshness: Check training data and live data freshness.
- get_trading_advice: BUY/SELL/HOLD trading recommendation based on price trend.
- check_price_alerts: Scan next 7 days for price anomaly risks (red/orange/yellow).
- generate_daily_briefing: Full structured daily market briefing.

Rules:
1. Output a JSON array of subtasks. Each subtask has "step" (number), "description" (what to do), "tool" (tool name or "synthesize"), "input" (tool input string).
2. Keep subtasks concrete and executable — one tool call per subtask.
3. For simple queries (single tool needed), output exactly ONE subtask.
4. For "briefing" or "report" queries, use generate_daily_briefing if available; otherwise combine rag_summarize, forecast_electricity_prices, and get_texas_weather.
5. The final subtask should be "synthesize" to combine all results.

Example for "生成未来一周的电价简报":
[
  {{{{step": 1, "description": "获取ERCOT天气和7日预报", "tool": "get_texas_weather", "input": "forecast 7天 ERCOT"}},
  {{{{step": 2, "description": "预测未来一周每日电价", "tool": "forecast_electricity_prices", "input": "7天后"}},
  {{{{step": 3, "description": "检索电力市场政策背景", "tool": "rag_summarize", "input": "ERCOT电力市场定价机制和近期政策"}},
  {{{{step": 4, "description": "综合所有结果生成简报", "tool": "synthesize", "input": ""}}
]

User query: {{user_query}}

Output ONLY the JSON array, nothing else."""

SYNTHESIZE_PROMPT = f"""You are an electricity market intelligence analyst. Today is {_date_today.today().isoformat()}. Synthesize the following execution results into a comprehensive, well-structured response.

User query: {{user_query}}

Execution results:
{{execution_log}}

Rules:
1. Use the SAME language as the user query.
2. Present price forecasts as a table where applicable.
3. Highlight key risks (extreme weather, price spikes, policy changes).
4. Be concise but complete — cover all subtasks.
5. End with actionable takeaways."""


class PlanExecuteState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    plan: list | None
    current_step: int
    execution_results: Annotated[list, operator.add]


def _parse_plan(raw: str) -> list:
    """Parse JSON plan from LLM output, handling markdown code fences."""
    text = raw.strip()
    # Remove code fences
    for fence in ["```json", "```"]:
        text = text.replace(fence, "")
    text = text.strip()

    # Find the outermost JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            plan = json.loads(text[start:end + 1])
            if isinstance(plan, list):
                return plan
        except json.JSONDecodeError:
            pass
    return []


def plan_node(state: PlanExecuteState):
    """Decompose user query into subtasks."""
    messages = state["messages"]
    user_query = ""
    for m in messages:
        if isinstance(m, HumanMessage):
            user_query = m.content
            break

    prompt = PLANNER_PROMPT.format(user_query=user_query)
    response = chat_model.invoke(prompt)
    plan = _parse_plan(response.content)

    if not plan:
        # Fallback: single-step plan
        plan = [{"step": 1, "description": "Direct response", "tool": "synthesize", "input": user_query}]

    print(f"[Planner] Generated {len(plan)} subtasks:")
    for s in plan:
        print(f"  Step {s.get('step', '?')}: {s.get('tool', '?')} — {s.get('description', '?')[:80]}")

    return {"plan": plan, "current_step": 0, "execution_results": []}


def execute_node(state: PlanExecuteState):
    """Execute the current subtask by calling the assigned tool."""
    plan = state["plan"]
    step_idx = state.get("current_step", 0)

    if step_idx >= len(plan):
        return {"messages": [], "current_step": step_idx + 1}

    subtask = plan[step_idx]
    tool_name = subtask.get("tool", "")
    tool_input = subtask.get("input", "")

    print(f"[Executor] Step {step_idx + 1}/{len(plan)}: {tool_name}('{tool_input[:60]}')")

    if tool_name == "synthesize" or tool_name not in TOOL_BY_NAME:
        # Synthesis step — just advance, handled by synthesize_node
        return {"current_step": step_idx + 1, "execution_results": [
            {"step": subtask.get("step"), "tool": tool_name, "result": ""}
        ]}

    tool = TOOL_BY_NAME[tool_name]
    try:
        result = tool.invoke(tool_input)
        result_str = str(result)
    except Exception as e:
        result_str = f"Error: {e}"

    print(f"  => Result: {result_str[:100]}...")

    return {
        "current_step": step_idx + 1,
        "execution_results": [{
            "step": subtask.get("step"),
            "description": subtask.get("description"),
            "tool": tool_name,
            "result": result_str,
        }],
    }


def synthesize_node(state: PlanExecuteState):
    """Combine all execution results into a final response."""
    user_query = ""
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            user_query = m.content
            break

    execution_log_parts = []
    for r in state.get("execution_results", []):
        execution_log_parts.append(
            f"Step {r.get('step')} [{r.get('tool')}]: {r.get('result')[:800]}"
        )
    execution_log = "\n\n".join(execution_log_parts)

    prompt = SYNTHESIZE_PROMPT.format(user_query=user_query, execution_log=execution_log)
    response = chat_model.invoke(prompt)

    return {"messages": [AIMessage(content=response.content)]}


def should_continue_execute(state: PlanExecuteState) -> Literal["execute", "synthesize"]:
    """Continue executing or move to synthesis?"""
    plan = state.get("plan", [])
    current = state.get("current_step", 0)
    if current < len(plan):
        return "execute"
    return "synthesize"


# Build Plan-Execute graph
pe_workflow = StateGraph(PlanExecuteState)

pe_workflow.add_node("plan", plan_node)
pe_workflow.add_node("execute", execute_node)
pe_workflow.add_node("synthesize", synthesize_node)

pe_workflow.set_entry_point("plan")

pe_workflow.add_edge("plan", "execute")
pe_workflow.add_conditional_edges(
    "execute",
    should_continue_execute,
    {"execute": "execute", "synthesize": "synthesize"},
)
pe_workflow.add_edge("synthesize", END)

plan_execute_app = pe_workflow.compile()


# ============================================================
# Streaming wrapper
# ============================================================
def run_plan_execute_stream(query: str):
    """Stream the plan-execute agent's response."""
    initial_state: PlanExecuteState = {
        "messages": [HumanMessage(content=query)],
        "plan": None,
        "current_step": 0,
        "execution_results": [],
    }

    for output in plan_execute_app.stream(initial_state, stream_mode="values"):
        if "messages" not in output:
            continue
        latest = output["messages"][-1] if output["messages"] else None
        if latest and isinstance(latest, AIMessage) and latest.content:
            yield latest.content


def run_plan_execute_chainlit(query: str):
    """Chainlit-compatible generator with step-by-step rendering."""
    initial_state: PlanExecuteState = {
        "messages": [HumanMessage(content=query)],
        "plan": None,
        "current_step": 0,
        "execution_results": [],
    }

    step_idx = 0
    for output in plan_execute_app.stream(initial_state, stream_mode="values"):
        plan = output.get("plan")
        current_step = output.get("current_step", 0)

        # Yield plan steps
        if plan and current_step == 0:
            yield {
                "type": "step",
                "node": "plan",
                "title": "Planning",
                "status": "running",
                "content": f"Decomposed into {len(plan)} subtasks:\n" + "\n".join(
                    f"  {s.get('step')}. {s.get('description')}" for s in plan
                ),
            }

        # Yield execution progress
        if plan and 0 < current_step <= len(plan):
            subtask = plan[current_step - 1]
            yield {
                "type": "step",
                "node": subtask.get("tool", "execute"),
                "title": f"Step {current_step}/{len(plan)}: {subtask.get('description', '')[:60]}",
                "status": "done",
                "content": "",
            }

        # Yield final response
        if "messages" in output:
            latest = output["messages"][-1] if output["messages"] else None
            if latest and isinstance(latest, AIMessage) and latest.content:
                yield {
                    "type": "step",
                    "node": "response",
                    "title": "Synthesis",
                    "status": "done",
                    "content": latest.content,
                }


# ============================================================
# Test
# ============================================================
if __name__ == "__main__":
    tests = [
        "生成未来一周的电价简报",
        "预测明天的电价",
    ]
    for q in tests:
        print(f"\n{'='*50}")
        print(f"Query: {q}")
        print(f"{'='*50}")
        for chunk in run_plan_execute_stream(q):
            print(chunk, end="", flush=True)
        print()
