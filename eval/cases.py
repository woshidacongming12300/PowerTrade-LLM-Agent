"""Agent evaluation cases.

Each case defines a query plus what a *correct* agent run should do:
- expected_tool: which tool must be invoked (or None for pure chat)
- keywords: substrings the final answer should contain
- category: scenario bucket for reporting
- lang: response language hint

The evaluation runner scores tool-hit, keyword-hit, LLM-judge quality (1-5)
and latency for all three agent paradigms.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalCase:
    id: str
    query: str
    category: str
    expected_tool: str | None = None
    keywords: tuple[str, ...] = ()
    lang: str = "zh"
    judge_hint: str = ""


CASES: list[EvalCase] = [
    # ========== 1. Casual chat (no tools) ==========
    EvalCase(
        id="chat_01", query="你好，你是谁？", category="chat",
        expected_tool=None, keywords=("助手", "电力"),
        judge_hint="回答应友好地自我介绍为电力市场 AI 助手",
    ),
    EvalCase(
        id="chat_02", query="谢谢你的帮助", category="chat",
        expected_tool=None, keywords=("不客气", "随时"),
    ),
    EvalCase(
        id="chat_03", query="你能做什么？", category="chat",
        expected_tool=None, keywords=("预测", "电价"),
        judge_hint="应列出主要能力：电价预测、天气、RAG 知识库、交易建议、日报",
    ),

    # ========== 2. Weather (tool: get_texas_weather) ==========
    EvalCase(
        id="wx_01", query="休斯顿今天天气怎么样？", category="weather",
        expected_tool="get_texas_weather", keywords=("天气",),
    ),
    EvalCase(
        id="wx_02", query="印第安纳波利斯未来7天天气如何？", category="weather",
        expected_tool="get_texas_weather", keywords=("天气", "7"),
    ),
    EvalCase(
        id="wx_03", query="德克萨斯最近有极端天气吗？", category="weather",
        expected_tool="get_texas_weather", keywords=("天气",),
    ),

    # ========== 3. Price forecast (tools: predict / forecast) ==========
    EvalCase(
        id="fc_01", query="预测明天的电价", category="forecast",
        expected_tool="forecast_electricity_prices", keywords=("41", "MWh"),
    ),
    EvalCase(
        id="fc_02", query="未来一周的电价走势怎么样？", category="forecast",
        expected_tool="forecast_electricity_prices", keywords=("MWh",),
    ),
    EvalCase(
        id="fc_03", query="3天后电价大概多少？", category="forecast",
        expected_tool="forecast_electricity_prices", keywords=("MWh",),
    ),
    EvalCase(
        id="fc_04", query="明天单日电价是多少？", category="forecast",
        expected_tool="forecast_electricity_prices", keywords=("MWh",),
    ),

    # ========== 4. RAG / policy (tool: rag_summarize) ==========
    EvalCase(
        id="rag_01", query="LMP电价由哪几个分量构成？", category="rag",
        expected_tool="rag_summarize", keywords=("能量", "阻塞", "损耗"),
    ),
    EvalCase(
        id="rag_02", query="MISO是美国什么机构？", category="rag",
        expected_tool="rag_summarize", keywords=("独立系统运营商", "RTO"),
    ),
    EvalCase(
        id="rag_03", query="日前市场和实时市场有什么区别？", category="rag",
        expected_tool="rag_summarize", keywords=("日前", "实时"),
    ),
    EvalCase(
        id="rag_04", query="电价预测有哪些方法？", category="rag",
        expected_tool="rag_summarize", keywords=("LSTM", "统计"),
    ),

    # ========== 5. Trading / alerts / briefing ==========
    EvalCase(
        id="tr_01", query="给我一个3天后的交易建议", category="trading",
        expected_tool="get_trading_advice", keywords=("买入", "卖出", "持有"),
    ),
    EvalCase(
        id="tr_02", query="最近7天有价格风险吗？", category="alerts",
        expected_tool="check_price_alerts", keywords=("告警", "风险", "红色", "橙色"),
    ),
    EvalCase(
        id="tr_03", query="生成今天的市场日报", category="briefing",
        expected_tool="generate_daily_briefing", keywords=("日报", "简报"),
    ),
]


def get_cases_by_category() -> dict[str, list[EvalCase]]:
    """Group cases by category for the report."""
    groups: dict[str, list[EvalCase]] = {}
    for c in CASES:
        groups.setdefault(c.category, []).append(c)
    return groups
