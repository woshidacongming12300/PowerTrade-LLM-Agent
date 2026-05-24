"""Internationalization (i18n) module for the Electricity Price Forecast Agent.
Simple dict-based approach — no external dependencies needed."""

T = {
    "app_title": {
        "zh": "电力现货智能预测系统",
        "en": "Electricity Price Intelligence",
    },
    "app_subtitle": {
        "zh": "多智能体协同 · 多变量 LSTM + Chronos-T5 双轨预测 · 混合 RAG 检索 · 交易决策引擎",
        "en": "Multi-Agent System · Multivariate LSTM + Chronos-T5 · Hybrid RAG · Trading Engine",
    },
    "sidebar_header": {
        "zh": "系统配置",
        "en": "Configuration",
    },
    "sidebar_mode": {
        "zh": "智能体模式",
        "en": "Agent Mode",
    },
    "mode_react": {
        "zh": "ReAct 智能体 (快速问答)",
        "en": "ReAct Agent (Quick Chat)",
    },
    "mode_langgraph": {
        "zh": "LangGraph Agent (动态反思)",
        "en": "LangGraph Agent (Dynamic + Reflection)",
    },
    "mode_plan_execute": {
        "zh": "Plan-Execute Agent (结构化规划)",
        "en": "Plan-Execute Agent (Structured)",
    },
    "mode_langgraph_help": {
        "zh": "LangGraph：LLM 自主决策调用 9 个工具 + Reflection 反思质检",
        "en": "LangGraph: LLM-decided tool calls (9 tools) + Reflection self-critique",
    },
    "mode_plan_execute_help": {
        "zh": "Plan-Execute：规划子任务 → 逐步执行 → 综合生成研报/日报",
        "en": "Plan-Execute: Plan → Execute step-by-step → Synthesize report/briefing",
    },
    "mode_react_help": {
        "zh": "ReAct：单轮快速问答，9 个工具自由调用",
        "en": "ReAct: fast single-turn Q&A with 9 tools available",
    },
    "lang_label": {
        "zh": "界面语言",
        "en": "Language",
    },
    "tools_header": {
        "zh": "可用工具",
        "en": "Tools Available",
    },
    "tool_lstm_single": {
        "zh": "LSTM 单日电价预测",
        "en": "LSTM Single-Day Forecast",
    },
    "tool_lstm_multi": {
        "zh": "LSTM 多日滚动预测",
        "en": "LSTM Multi-Day Rolling Forecast",
    },
    "tool_chronos": {
        "zh": "Chronos-T5 交叉验证",
        "en": "Chronos-T5 Cross-Validation",
    },
    "tool_rag": {
        "zh": "混合 RAG 知识检索 (Chroma + BM25 + RRF)",
        "en": "Hybrid RAG Retrieval (Chroma + BM25 + RRF)",
    },
    "tool_weather": {
        "zh": "气象感知 (Open-Meteo API)",
        "en": "Weather Sensing (Open-Meteo)",
    },
    "tool_trading_advice": {
        "zh": "交易策略建议 (BUY/SELL/HOLD)",
        "en": "Trading Advice (BUY/SELL/HOLD)",
    },
    "tool_price_alerts": {
        "zh": "价格异常告警 (红/橙/黄三级)",
        "en": "Price Anomaly Alerts (3-level)",
    },
    "tool_daily_briefing": {
        "zh": "电力市场日报生成",
        "en": "Daily Market Briefing",
    },
    "tool_confidence": {
        "zh": "预测置信区间 (双模型交叉验证)",
        "en": "Prediction Confidence (Dual-Model)",
    },
    "tool_gridstatus": {
        "zh": "MISO 实时电价拉取 (gridstatus)",
        "en": "MISO Live LMP Fetch (gridstatus)",
    },
    "status_header": {
        "zh": "系统状态",
        "en": "System Status",
    },
    "status_model": {
        "zh": "LSTM 模型",
        "en": "LSTM Model",
    },
    "status_model_ready": {
        "zh": "就绪",
        "en": "Ready",
    },
    "status_model_not_ready": {
        "zh": "未训练",
        "en": "Not Trained",
    },
    "status_data_csv": {
        "zh": "CSV 数据",
        "en": "CSV Data",
    },
    "status_data_fresh": {
        "zh": "较新",
        "en": "Fresh",
    },
    "status_data_stale": {
        "zh": "数据滞后",
        "en": "Stale",
    },
    "status_data_stale_days": {
        "zh": "滞后 {days} 天",
        "en": "{days}d behind",
    },
    "status_data_none": {
        "zh": "未找到",
        "en": "Not Found",
    },
    "status_live": {
        "zh": "实时数据源",
        "en": "Live Source",
    },
    "status_live_on": {
        "zh": "已接入",
        "en": "Connected",
    },
    "status_live_off": {
        "zh": "未启用",
        "en": "Disabled",
    },
    "footer_text": {
        "zh": "电力现货智能预测系统 · 基于 LSTM + Chronos-T5 + LangGraph 多智能体架构",
        "en": "Electricity Price Intelligence · LSTM + Chronos-T5 + LangGraph Multi-Agent Architecture",
    },
    "chat_placeholder": {
        "zh": "请输入您的问题，例如：预测未来一周的电价 或 现在应该买入还是卖出 或 生成今日市场日报",
        "en": "Ask: 'Forecast next week's prices' or 'Should I buy or sell?' or 'Generate daily briefing'",
    },
    "spinner_react": {
        "zh": "智能体思考中...",
        "en": "Agent thinking...",
    },
    "spinner_langgraph": {
        "zh": "动态智能体思考中...",
        "en": "Dynamic agent thinking...",
    },
    "spinner_plan_execute": {
        "zh": "智能体规划与执行中...",
        "en": "Agent planning & executing...",
    },
    "quick_actions": {
        "zh": "快捷操作",
        "en": "Quick Actions",
    },
    "quick_forecast_tomorrow": {
        "zh": "预测明日电价",
        "en": "Forecast Tomorrow",
    },
    "quick_forecast_week": {
        "zh": "未来一周预测",
        "en": "Weekly Forecast",
    },
    "quick_trading_advice": {
        "zh": "交易建议",
        "en": "Trading Advice",
    },
    "quick_daily_briefing": {
        "zh": "日报生成",
        "en": "Daily Briefing",
    },
    "quick_weather": {
        "zh": "天气影响",
        "en": "Weather Impact",
    },
    "quick_data_status": {
        "zh": "数据状态",
        "en": "Data Status",
    },
    "kpi_average_price": {
        "zh": "7日均价预测",
        "en": "7-Day Avg Forecast",
    },
    "kpi_trend": {
        "zh": "价格趋势",
        "en": "Price Trend",
    },
    "kpi_weather_risk": {
        "zh": "天气风险等级",
        "en": "Weather Risk",
    },
    "kpi_model_confidence": {
        "zh": "模型置信度",
        "en": "Model Confidence",
    },
    "trend_rising": {
        "zh": "上涨",
        "en": "Rising",
    },
    "trend_falling": {
        "zh": "下跌",
        "en": "Falling",
    },
    "trend_stable": {
        "zh": "平稳",
        "en": "Stable",
    },
    "risk_low": {
        "zh": "低",
        "en": "Low",
    },
    "risk_medium": {
        "zh": "中",
        "en": "Medium",
    },
    "risk_high": {
        "zh": "高",
        "en": "High",
    },
    "welcome_title": {
        "zh": "欢迎使用电力现货智能预测系统",
        "en": "Welcome to Electricity Price Intelligence",
    },
    "welcome_text": {
        "zh": (
            "我是您的电力市场 AI 助手，可以帮您：\n\n"
            "- **预测电价**：多变量 LSTM（价格+天气）+ Chronos-T5 双轨预测，含置信区间\n"
            "- **交易决策**：基于预测趋势生成 BUY / SELL / HOLD 建议\n"
            "- **风险告警**：红/橙/黄三级价格异常检测 + 天气联动\n"
            "- **生成研报**：一键生成结构化电力市场日报\n"
            "- **分析政策**：检索 ERCOT/MISO 交易规则与调度策略\n\n"
            "请在下方输入您的问题，或点击快捷操作开始。"
        ),
        "en": (
            "I'm your electricity market AI assistant. I can help you:\n\n"
            "- **Price Forecast**: Multivariate LSTM (price+weather) + Chronos-T5 with confidence intervals\n"
            "- **Trading Advice**: BUY / SELL / HOLD recommendations based on predicted trends\n"
            "- **Risk Alerts**: 3-level price anomaly detection with weather correlation\n"
            "- **Market Briefing**: one-click structured daily electricity market reports\n"
            "- **Policy Analysis**: retrieve ERCOT/MISO trading rules and dispatch strategies\n\n"
            "Enter your question below or click a quick action to get started."
        ),
    },
    "chart_title": {
        "zh": "价格预测走势",
        "en": "Price Forecast Trend",
    },
    "chart_price_label": {
        "zh": "电价 ($/MWh)",
        "en": "Price ($/MWh)",
    },
    "error_no_response": {
        "zh": "未能获取有效响应，请重试。",
        "en": "No valid response received. Please try again.",
    },
    "refresh_data_hint": {
        "zh": "提示：设置环境变量 GRIDSTATUS_ENABLED=1 可启用 MISO 实时电价数据拉取",
        "en": "Tip: Set GRIDSTATUS_ENABLED=1 to enable live MISO LMP data fetching",
    },
}


def t(key: str, lang: str = "zh") -> str:
    """Look up a translation key. Falls back to key name if not found."""
    entry = T.get(key, {})
    if isinstance(entry, dict):
        return entry.get(lang, entry.get("en", key))
    return str(entry)
