import os

from rag.rag_service import RagSummarizeService
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path

import random
from langchain_core.tools import tool


import sys
import os
from langchain_core.tools import tool
from utils.logger_handler import logger
rag = RagSummarizeService()


# 确保 Python 能找到我们新建的 algorithms 文件夹
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from algorithms.epf_predictor import predict_future_price
@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) -> str:
    return rag.rag_summarize(query)



@tool
def predict_electricity_price(target_date: str) -> str:
    """
    预测未来特定日期的电力市场交易价格。
    当用户询问“预测某天的电价”、“未来电价走势”、“下周电价”时，必须调用此工具。
    参数 target_date 应该是类似于 '2026-04-20' 或 '明天' 的时间描述字符串。

    【强制执行指令】：
    调用本工具获取到预测电价数值后，你绝对不能直接将该数值输出给用户！
    你必须紧接着调用 rag_summarize 工具，检索最新的“分时电价政策”或“交易规则”，
    最后，你必须将获取到的预测电价数值，与检索到的政策文本进行深度逻辑结合，给出一份完整的综合分析回答。
    """
    logger.info(f"💡 Agent 正在调动底层的 LSTM 神经网络！开始计算 {target_date} 的电力张量数据...")

    try:
        # ================= 高光时刻：调用真实的深度学习模型 =================
        predicted_price = predict_future_price(target_date)
        # ====================================================================

        # 注意单位我们改成了你真实数据的 $/MWh
        return f"【预测结果】根据自研 LSTM 时序预测模型推演，{target_date} 的预测加权平均电价为：{predicted_price} $/MWh。建议结合最新电价政策进行交易决策。"

    except Exception as e:
        logger.error(f"底层的时序模型推理失败: {e}")
        return f"【系统提示】底层电力预测模型在计算 {target_date} 的数据时遇到异常，请检查算法库运行状态。"