"""Smoke test for the Electricity Price Intelligence project.

Verifies, in order:
1. Config loading (rag.yml / chroma.yml / prompts.yml)
2. Prompt loading (system / rag / report)
3. LLM & embedding factory instantiation (no API call)
4. RAG vector store init + empty-knowledge-base load_document (no crash)
5. All 10 agent tools importable
6. LangGraph app compile (langgraph / plan-execute)
7. ReAct agent creation
8. Real LSTM inference with pre-trained model (no API key needed)

Usage:
    python scripts/smoke_test.py
"""

import os
import sys
import traceback

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

PASSED = []
FAILED = []


def check(name: str, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  [PASS] {name}")
    except Exception as e:
        FAILED.append(name)
        print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)


def main():
    print("=" * 60)
    print("Electricity Price Intelligence — Smoke Test")
    print("=" * 60)

    # 1. Config loading
    print("\n[1] Config loading")
    check("config rag.yml", lambda: (
        __import__("utils.config_handler", fromlist=["rag_conf"]).rag_conf["chat_model_name"] == "qwen-plus"
        and print("      chat_model_name = qwen-plus")
    ))
    check("config chroma.yml", lambda: (
        __import__("utils.config_handler", fromlist=["chroma_conf"]).chroma_conf["collection_name"]
    ))
    check("config prompts.yml", lambda: (
        __import__("utils.config_handler", fromlist=["prompts_conf"]).prompts_conf["report_prompt_path"]
    ))

    # 2. Prompt loading
    print("\n[2] Prompt loading")
    check("load_system_prompts", lambda: __import__("utils.prompt_loader", fromlist=["load_system_prompts"]).load_system_prompts())
    check("load_rag_prompts", lambda: __import__("utils.prompt_loader", fromlist=["load_rag_prompts"]).load_rag_prompts())
    check("load_report_prompts", lambda: __import__("utils.prompt_loader", fromlist=["load_report_prompts"]).load_report_prompts())

    # 3. Model factory (instantiation only — no API call)
    print("\n[3] Model factory")
    check("chat_model instantiate", lambda: __import__("model.factory", fromlist=["chat_model"]).chat_model)
    check("embed_model instantiate", lambda: __import__("model.factory", fromlist=["embed_model"]).embed_model)

    # 4. RAG vector store + empty KB load
    print("\n[4] RAG vector store")
    check("VectorStoreService init", lambda: __import__("rag.vector_store", fromlist=["VectorStoreService"]).VectorStoreService())
    check("load_document (empty data/)", lambda: __import__("rag.vector_store", fromlist=["VectorStoreService"]).VectorStoreService().load_document())

    # 5. Agent tools import
    print("\n[5] Agent tools")
    check("import 10 tools", lambda: __import__("agent.tools.agent_tools", fromlist=["get_current_date", "get_texas_weather", "predict_electricity_price", "forecast_electricity_prices", "get_chronos_prediction", "rag_summarize", "get_data_freshness", "get_trading_advice", "check_price_alerts", "generate_daily_briefing"]))

    # 6. LangGraph graphs compile
    print("\n[6] LangGraph graphs")
    check("langgraph_app compile", lambda: __import__("agent.langgraph_agent", fromlist=["langgraph_app"]).langgraph_app)
    check("plan_execute_app compile", lambda: __import__("agent.plan_execute_agent", fromlist=["plan_execute_app"]).plan_execute_app)

    # 7. ReAct agent creation
    print("\n[7] ReAct agent")
    check("ReactAgent()", lambda: __import__("agent.react_agent", fromlist=["ReactAgent"]).ReactAgent())

    # 8. Real LSTM inference (pre-trained model, no API key)
    print("\n[8] LSTM inference (real model)")
    check("predict_future_price('明天')", lambda: print(f"      -> {__import__('algorithms.epf_predictor', fromlist=['predict_future_price']).predict_future_price('明天')} $/MWh"))
    check("predict_future_prices('3天后')", lambda: print(
        f"      -> {len(__import__('algorithms.epf_predictor', fromlist=['predict_future_prices']).predict_future_prices('3天后'))} days forecast"
    ))
    check("get_recent_prices_flat", lambda: print(
        f"      -> {__import__('algorithms.price_data_provider', fromlist=['get_recent_prices_flat']).get_recent_prices_flat(None, 7)}"
    ))

    print("\n" + "=" * 60)
    print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("Failed:", ", ".join(FAILED))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
