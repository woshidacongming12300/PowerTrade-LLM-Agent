import os, sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (
    forecast_electricity_prices,
    predict_electricity_price,
    rag_summarize,
    get_texas_weather,
    get_chronos_prediction,
    get_data_freshness,
    get_trading_advice,
    check_price_alerts,
    generate_daily_briefing,
    get_current_date,
)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch


class ReactAgent:
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[
                get_current_date,
                predict_electricity_price,
                forecast_electricity_prices,
                rag_summarize,
                get_texas_weather,
                get_chronos_prediction,
                get_data_freshness,
                get_trading_advice,
                check_price_alerts,
                generate_daily_briefing,
            ],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }

        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("预测一下明天的电价"):
        print(chunk, end="", flush=True)
