# ⚡ EPF-Smart-Agent: 基于大模型与深度学习的电力智能交易决策系统

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![LangChain](https://img.shields.io/badge/LangChain-Agent%20Framework-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-orange)

## 📖 项目简介
本项目是一个针对北美洲际交易所（ICE）电力现货市场交易场景开发的**跨模态大模型智能体（Agent）**。
传统的 LLM 缺乏高精度数学推演能力，且在垂直金融/能源领域容易产生幻觉。本项目通过底层源码级干预，将 **PyTorch 时序预测网络**封装为大模型原生计算工具，并结合 **RAG（检索增强生成）** 架构，成功打通了从“底层张量预测”到“上层商业决策”的完整技术闭环。

系统能够精准预测未来交割日电价，并自动交叉检索本地《电力交易调度政策》，最终输出具备极强解释性的自动化交易策略研报。

## ✨ 核心技术亮点
- **🧠 深度学习底座 (Time-Series Forecasting)**：基于 PyTorch 独立训练 LSTM 神经网络，完成多变量滑动窗口特征工程，为 Agent 提供真实可靠的数值支撑。
- **🛠️ 工具链调用 (Tool Calling & ReAct)**：基于 LangChain 搭建 ReAct 多步推理 Agent，将深度学习推理过程封装为原生 Tool，打破了大模型的数学计算瓶颈。
- **📚 知识库增强 (RAG)**：接入 ChromaDB 本地向量数据库，对海量电力调度政策进行切块与向量化。
- **⛓️ 强制编排与幻觉抑制**：设计防“工具链断裂”机制，强制 Agent 在获取电价数值后必须进行政策检索，实现定量数据与定性规则的强交叉验证。

## 📂 项目结构

```text
EPF-Smart-Agent/
├── agent/                  # 大模型智能体核心逻辑
│   ├── tools/
│   │   └── agent_tools.py  # 桥接底层算法的预测工具 (Tool Calling)
│   └── react_agent.py      # ReAct Agent 编排与多步推理逻辑
│
├── algorithms/             # 深度学习算力底座
│   ├── epf_predictor.py    # LSTM 数据清洗、训练与推理引擎
│   └── lstm_epf.pth        # 预训练模型权重 (由于体积原因未上传)
│
├── data/                   # RAG 向量知识库源文件 (非结构化数据)
│   └── 2026北美电力交易策略与政策.txt
│
├── datasets/               # 结构化数据集
│   └── clean_electric_data.csv # 供模型训练的清洗后数据 (示例)
│
├── .env                    # 大模型 API 密钥配置 (需自行创建)
├── requirements.txt        # 项目依赖清单
└── app.py                  # 系统主入口
