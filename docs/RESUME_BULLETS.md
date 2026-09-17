# 简历要点速查（Resume Bullets）

> 面向 Agent 开发 / LLM 应用工程师岗位。每条都基于本项目真实实现，可被深挖追问。

## 项目一句话
Electricity Price Intelligence — 多智能体电力市场价格预测与情报系统
（LangGraph · 多 Agent 编排 · 混合 RAG · FastAPI 服务化 · 可观测性 · 评测体系）

## 核心 Bullets（按价值排序）

### 1. 多 Agent 系统与编排
- 基于 LangGraph 实现 **4 种 Agent 范式**：LangGraph+Reflection（自反思纠错）、Plan-Execute（计划-执行）、ReAct、Supervisor（Coordinator 路由 3 个专家 Agent：气象/预测/政策研究 + 综合器）
- Supervisor 模式：LLM Coordinator 将用户请求路由到专业 Agent，专家执行后由 Synthesizer 汇总，实现分层多智能体协作

### 2. Agent 工程化基础设施（Harness 层）
- **可观测性**：自研可插拔 Trace 层（`observability/trace.py`），默认 JSONL 落盘、可切换 Langfuse 后端；记录 run/tool 事件、延迟、响应全文
- **长期记忆**：SQLite 双级记忆（`memory/`）——用户偏好持久化 + 预测历史 + 会话摘要，自动注入 agent prompt，实现跨会话记忆
- **工具注册表**：统一 Tool 元数据（名称/描述/参数 schema）通过 `/api/tools` 自省暴露

### 3. 服务化与 API 设计
- FastAPI 服务化 4 种 Agent，**SSE 流式输出**（step_running/step_done/response_chunk/response_done 事件协议）
- Pydantic 契约化请求/响应（自动 OpenAPI 文档）、可选 API-Key 鉴权、统一错误信封
- 6+ 组 REST 端点：chat（REST+SSE）、predict、rag/search、tools、memory、health

### 4. 混合 RAG
- ChromaDB 稠密向量 + BM25 稀疏检索 + **RRF 融合（k=60）**，20K+ chunks 上 HitRate@3 96.25%
- RAG 检索结果带引用溯源（chunk + source）

### 5. 评测体系（Agent Evaluation）
- 自建 `eval/` 评测集：16+ 用例覆盖闲聊/天气/预测/RAG/交易 5 类场景
- 三（四）范式横向对比：工具命中率 / 关键词命中 / LLM-Judge 质量分（1-5）/ 延迟
- 自动化报告（markdown + 控制台摘要），支持回归测试

### 6. 预测模型
- LSTM（200 epoch + early stopping）多变量预测，融合 4 天气特征使 MAE 降 20%
- Chronos-T5 零样本交叉验证轨道；**离线快速降级**（HF 不可达时 200s→11s，不阻塞主流程）

### 7. 自动化测试与 CI/CD
- **75 个 pytest 用例**：单元（日期解析/工具注册/记忆/观测/配置/Schema）+ FastAPI 集成 + Supervisor 路由解析（抽纯函数可测化）
- **凭据感知测试**：live-model 用例打 `integration` 标记，无 key 自动跳过（CI 无密钥 73 过 2 跳 ~12s；有密钥 75 全过）
- **模型懒加载架构**：`ChatOpenAI` 构造时校验 api_key，改惰性代理后全应用无凭据可导入——CI 可跑、本地启动即时报错清晰
- **GitHub Actions 三 job**：`test`（全量测试）/ `eval`（Agent 评测，配置 secret 后跑）/ `syntax`（compileall 冒烟）

## 技术栈关键词
Python · LangGraph · LangChain · FastAPI · SSE · ChromaDB · BM25 · RRF · SQLite · Pydantic · PyTorch · LSTM · Chronos-T5 · pytest · GitHub Actions · Docker

## 可被追问的深度问题（准备答案）
1. 为什么用 RRF 而不是加权融合？k 值怎么定？
2. Supervisor 路由失败（LLM 误判）如何兜底？
3. 记忆注入会不会污染 prompt？如何控制长度？
4. SSE 断线重连怎么处理？流式中断如何恢复？
5. 评测的 judge 用同一个模型会不会有偏？如何避免？
6. chronos 离线降级的判断依据是什么？

## 改进记录（面试时展示迭代思维）
- v1 → v2：发现并修复 ReAct 模式提示词误用 bug（专家路由提示词被注入 ReAct）
- v2 → v3：对话模型迁移到百炼 OpenAI 兼容接口（旧 dashscope SDK 不支持新模型）
- v3 → v4：修复 Chronos HF 下载阻塞（HF_HUB_OFFLINE 快速降级）
