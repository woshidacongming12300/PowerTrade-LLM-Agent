import json
import os
import sys
import time

# GRIDSTATUS_ENABLED is set via .env — do NOT force-enable here

import pandas as pd
import streamlit as st

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from agent.react_agent import ReactAgent
from agent.langgraph_agent import run_langgraph_stream
from agent.plan_execute_agent import run_plan_execute_stream
from utils.i18n import t
from algorithms.price_data_provider import get_last_csv_delivery_date, get_live_price_fetcher

st.set_page_config(
    page_title="Electricity Price Intelligence",
    page_icon="⚡",
    layout="wide",
)


# ============================================================
# Minimal CSS (only accent touches Streamlit can't do natively)
# ============================================================
def inject_minimal_css():
    st.markdown("""<style>
    .stApp { background-color: #0f1117; }
    .block-container { padding-top: 1rem; }
    section[data-testid="stSidebar"] .stMarkdown { font-size: 0.9rem; }
    .stMetric { background: #1a1c23; border-radius: 0.5rem; padding: 0.75rem 1rem; }
    </style>""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================
def init_session():
    defaults = {
        "agent": None,
        "messages": [],
        "lang": "zh",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if st.session_state["agent"] is None:
        st.session_state["agent"] = ReactAgent()


# ============================================================
# Sidebar
# ============================================================
def render_sidebar():
    lang = st.session_state["lang"]

    with st.sidebar:
        # Language
        st.selectbox(
            t("lang_label", lang),
            options=["zh", "en"],
            format_func=lambda x: "中文" if x == "zh" else "English",
            index=0 if lang == "zh" else 1,
            key="lang_selector",
        )
        if st.session_state["lang_selector"] != lang:
            st.session_state["lang"] = st.session_state["lang_selector"]
            st.rerun()

        st.divider()

        # Mode
        st.subheader(t("sidebar_mode", lang))
        mode = st.radio(
            "mode",
            options=["react", "langgraph", "plan_execute"],
            format_func=lambda x: t(f"mode_{x}", lang),
            key="agent_mode",
            label_visibility="collapsed",
        )

        st.divider()

        # Tools list
        st.subheader(t("tools_header", lang))
        tools = [
            ("🔮", "tool_lstm_multi"),
            ("📊", "tool_confidence"),
            ("💰", "tool_trading_advice"),
            ("⚠️", "tool_price_alerts"),
            ("📰", "tool_daily_briefing"),
            ("📚", "tool_rag"),
            ("🌡️", "tool_weather"),
            ("📡", "tool_gridstatus"),
        ]
        for icon, key in tools:
            st.caption(f"{icon}  {t(key, lang)}")

        st.divider()

        # System Status
        st.subheader(t("status_header", lang))

        model_ok = os.path.exists(os.path.join(_project_root, "algorithms", "lstm_epf_multivariate.pth")) or os.path.exists(os.path.join(_project_root, "algorithms", "lstm_epf.pth"))
        csv_last = get_last_csv_delivery_date()
        from datetime import date as _date
        fetcher = get_live_price_fetcher()

        col1, col2 = st.columns([1, 2])
        with col1:
            status_icon = "🟢" if model_ok else "🟡"
        with col2:
            st.caption(f"{status_icon} {t('status_model', lang)}: "
                       f"{t('status_model_ready', lang) if model_ok else t('status_model_not_ready', lang)}")

        col1, col2 = st.columns([1, 2])
        with col1:
            if csv_last:
                gap = (_date.today() - csv_last).days
                icon = "🟢" if gap <= 7 else "🟡" if gap <= 30 else "🔴"
            else:
                icon = "🔴"
        with col2:
            if csv_last:
                gap = (_date.today() - csv_last).days
                st.caption(f"{icon} {t('status_data_csv', lang)}: "
                           f"{t('status_data_stale_days', lang).format(days=gap)}")
            else:
                st.caption(f"{icon} {t('status_data_csv', lang)}: {t('status_data_none', lang)}")

        col1, col2 = st.columns([1, 2])
        with col1:
            icon = "🟢" if fetcher else "🟡"
        with col2:
            st.caption(f"{icon} {t('status_live', lang)}: "
                       f"{t('status_live_on', lang) if fetcher else t('status_live_off', lang)}")

        st.divider()

        # Quick actions
        st.subheader(t("quick_actions", lang))
        q1, q2 = st.columns(2)
        with q1:
            if st.button(f"⚡ {t('quick_forecast_tomorrow', lang)}", use_container_width=True):
                st.session_state["quick_prompt"] = (
                    "预测明天的电价" if lang == "zh" else "Predict tomorrow's electricity price"
                )
        with q2:
            if st.button(f"📊 {t('quick_forecast_week', lang)}", use_container_width=True):
                st.session_state["quick_prompt"] = (
                    "生成未来一周的电价简报" if lang == "zh" else "Generate a weekly electricity price briefing"
                )
        q3, q4 = st.columns(2)
        with q3:
            if st.button(f"💰 {t('quick_trading_advice', lang)}", use_container_width=True):
                st.session_state["quick_prompt"] = (
                    "现在应该买入还是卖出电力？给出交易建议" if lang == "zh" else "Should I buy or sell electricity? Give trading advice"
                )
        with q4:
            if st.button(f"📰 {t('quick_daily_briefing', lang)}", use_container_width=True):
                st.session_state["quick_prompt"] = (
                    "生成今日电力市场日报" if lang == "zh" else "Generate today's electricity market briefing"
                )
        q5, q6 = st.columns(2)
        with q5:
            if st.button(f"🌡️ {t('quick_weather', lang)}", use_container_width=True):
                st.session_state["quick_prompt"] = (
                    "当前天气对电价有什么影响" if lang == "zh" else "How does current weather impact electricity prices"
                )
        with q6:
            if st.button(f"📋 {t('quick_data_status', lang)}", use_container_width=True):
                st.session_state["quick_prompt"] = (
                    "检查数据新鲜度和模型状态" if lang == "zh" else "Check data freshness and model status"
                )

        st.divider()
        st.caption(t("refresh_data_hint", lang))

    return mode


# ============================================================
# Main
# ============================================================
def main():
    inject_minimal_css()
    init_session()
    lang = st.session_state["lang"]

    # ---- Header ----
    st.title(t("app_title", lang))
    st.caption(t("app_subtitle", lang))

    mode = render_sidebar()

    # ---- Chat history ----
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "chart_data" in message and message["chart_data"]:
                df = pd.DataFrame(message["chart_data"])
                st.line_chart(df.set_index("forecast_date"), y="price",
                              y_label=t("chart_price_label", lang))

    # ---- Welcome (when empty) ----
    if not st.session_state["messages"]:
        with st.chat_message("assistant"):
            st.markdown(t("welcome_text", lang))

    # ---- Chat input ----
    prompt = None
    if "quick_prompt" in st.session_state and st.session_state["quick_prompt"]:
        prompt = st.session_state["quick_prompt"]
        st.session_state["quick_prompt"] = None
    else:
        prompt = st.chat_input(t("chat_placeholder", lang))

    if not prompt:
        return

    # User message
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # === LangGraph Dynamic Agent ===
    if mode == "langgraph":
        with st.spinner(t("spinner_langgraph", lang)):
            full_response_parts = []
            for chunk in run_langgraph_stream(prompt):
                full_response_parts.append(chunk)
            full_response = "".join(full_response_parts)

        # Extract chart data
        chart_data = None
        try:
            import re as _re
            matches = _re.findall(r'\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([\d.]+)\s*\|', full_response)
            if matches:
                seen = set()
                chart_data = []
                for d, p in matches:
                    if d not in seen:
                        seen.add(d)
                        chart_data.append({"forecast_date": d, "price": float(p)})
        except Exception:
            pass

        st.chat_message("assistant").write(full_response)
        if chart_data:
            df = pd.DataFrame(chart_data)
            st.line_chart(df.set_index("forecast_date"), y="price",
                          y_label=t("chart_price_label", lang))

        st.session_state["messages"].append({
            "role": "assistant", "content": full_response, "chart_data": chart_data,
        })

    # === Plan-Execute Agent ===
    elif mode == "plan_execute":
        with st.spinner(t("spinner_plan_execute", lang)):
            full_response_parts = []
            for chunk in run_plan_execute_stream(prompt):
                full_response_parts.append(chunk)
            full_response = "".join(full_response_parts)

        chart_data = None
        try:
            import re as _re
            json_match = _re.search(r'\{.*"series".*\}', full_response, _re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if "series" in parsed and parsed["series"]:
                    chart_data = parsed["series"]
            if not chart_data:
                matches = _re.findall(r'\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([\d.]+)\s*\|', full_response)
                if matches:
                    seen = set()
                    chart_data = []
                    for d, p in matches:
                        if d not in seen:
                            seen.add(d)
                            chart_data.append({"forecast_date": d, "price": float(p)})
        except Exception:
            pass

        st.chat_message("assistant").write(full_response)
        if chart_data:
            df = pd.DataFrame(chart_data)
            st.line_chart(df.set_index("forecast_date"), y="price",
                          y_label=t("chart_price_label", lang))

        st.session_state["messages"].append({
            "role": "assistant", "content": full_response, "chart_data": chart_data,
        })

    # === ReAct Agent ===
    else:
        with st.spinner(t("spinner_react", lang)):
            full_chunks = []
            res_stream = st.session_state["agent"].execute_stream(prompt)

            def capture(gen, cache):
                for chunk in gen:
                    cache.append(chunk)
                    for char in chunk:
                        time.sleep(0.01)
                        yield char

            st.chat_message("assistant").write_stream(capture(res_stream, full_chunks))
            full_text = "".join(full_chunks)

        chart_data = None
        try:
            import re as _re
            json_match = _re.search(r'\{.*"series".*\}', full_text, _re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
                if "series" in parsed and parsed["series"]:
                    chart_data = parsed["series"]
            if not chart_data:
                matches = _re.findall(r'\|\s*(\d{4}-\d{2}-\d{2})\s*\|\s*([\d.]+)\s*\|', full_text)
                if matches:
                    seen = set()
                    chart_data = []
                    for d, p in matches:
                        if d not in seen:
                            seen.add(d)
                            chart_data.append({"forecast_date": d, "price": float(p)})
        except Exception:
            pass

        st.session_state["messages"].append({
            "role": "assistant", "content": full_text, "chart_data": chart_data,
        })

    st.rerun()


if __name__ == "__main__":
    main()
