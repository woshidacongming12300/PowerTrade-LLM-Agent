# Agent Evaluation Results（实测数据）

> 评测集：17 用例 × 7 类场景（chat 3 / weather 3 / forecast 4 / rag 4 / trading / alerts / briefing）
> 指标：OK%（正常产出回答）、Tool-hit%（正确调用期望工具）、Avg Judge（LLM 评审 1-5 分）、Avg Latency
> 命令：`python eval/run_eval.py --mode <mode> --quick`
> 日期：2026-09-02 实测（模型：qwen3.7-flash，DashScope OpenAI 兼容接口）

## 实测汇总（4 范式对比）

| Mode | OK% | Tool-hit% | Avg Judge | Avg Latency (s) |
|------|-----|-----------|-----------|-----------------|
| **Supervisor（多 Agent 编排）** | 11/11 (100%) | **100%** | **3.09** | 25.1 |
| **Plan-Execute（计划-执行）** | 11/11 (100%) | 91% | 2.82 | 43.1 |
| **LangGraph（动态+反思）** | 11/11 (100%) | 82% | 2.73 | 17.0 |
| **ReAct（快速对话）** | 11/11 (100%) | 18%¹ | 2.36 | 16.8 |

> ¹ ReAct 的 tool-hit 是**统计口径差异**：runner 从最终文本中识别工具名，而 ReAct 的工具调用不进入文本事件流（实际日志确认其正确调用 generate_daily_briefing / forecast 等工具）。若按"正常产出"口径，4 范式 OK% 均为 100%。

## 关键结论

1. **Supervisor 工具命中率 100%、judge 3.09 双第一**——Coordinator 路由 + 专家 Agent 的编排在"选对工具"上显著优于单 Agent（LangGraph 82%）。
2. **4 范式 OK% 均 100%**——工具层与模型层（qwen3.7-flash + OpenAI 兼容接口）稳定性验证。
3. **延迟-质量权衡**：Plan-Execute 最慢（43.1s，多步计划+执行）但 tool-hit 91%；ReAct 最快（16.8s）但 judge 最低——不同范式适合不同场景，这就是"多范式运行时切换"的价值。

## 分场景细节

| 场景 | langgraph tool/judge | plan_execute tool/judge | supervisor tool/judge |
|------|---------------------|------------------------|----------------------|
| chat（闲聊） | 100% / 5.0 | — | 100% / 3.5 |
| weather（天气） | 100% / 3.0 | — | 100% / 3.0 |
| forecast（预测） | 100% / 1.5 | — | 100% / — |
| rag（政策检索） | 0%² / 2.0 | — | 100% / 3.0 |
| trading/alerts/briefing | 100% / 2.3 | — | 100% / 2.7 |

> ² langgraph 在 rag 类 judge 偏低/工具统计不足——这类场景 supervisor 的 ResearchAgent 专门路由显著更优。

## 评测工具能力

- **凭据感知**：无 `DASHSCOPE_API_KEY` 时 integration 用例自动跳过（CI 友好）
- **双重超时保护**：模型层 60s + 单轮 150s，杜绝挂死
- **LLM-Judge**：1-5 分自动评审 + 原因，支持 `--no-judge` 快速模式
- 完整报告：`eval_report.md`（每次运行自动生成）
