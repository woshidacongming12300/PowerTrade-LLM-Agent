# 工程化改造记录（CHANGELOG）

> 项目：Electricity Price Intelligence — 多智能体电力市场价格预测与情报系统
> 本次改造目标：把这套「多 Agent + RAG + LSTM 预测」的课程/个人项目，升级为**面向 Agent 开发岗位简历的工程化项目**
> 完成时间：2026-09（P0 → P2 全部落地）

---

## 一、总览：本次新增能力

| 层 | 能力 | 状态 |
|----|------|------|
| **服务化** | FastAPI 暴露 4 种 Agent 范式（REST + SSE 流式） | ✅ 已验证 |
| **多 Agent** | Supervisor 分层编排（Coordinator → 3 专家 → Synthesizer） | ✅ tool-hit 100% |
| **记忆** | SQLite 长期记忆（偏好 + 预测历史 + 会话摘要） | ✅ 跨会话持久化 |
| **可观测** | 可插拔 Trace 层（JSONL 默认 / Langfuse 可选） | ✅ 已记录 |
| **评测** | 4 范式横向对比评测集（17 用例 + LLM Judge） | ✅ 数据齐全 |
| **测试** | 75 个 pytest 用例（凭据感知分层） | ✅ 全通过 |
| **CI** | GitHub Actions 三 job（test / eval / syntax） | ✅ 配置就绪 |
| **前端** | Chainlit 接入 supervisor + 记忆展示 | ✅ 运行中 |

---

## 二、新增模块（文件级明细）

### 1. FastAPI 服务层（`api/`）

| 文件 | 说明 |
|------|------|
| `api/main.py` | FastAPI 应用：CORS、统一错误处理、6 组路由注册 |
| `api/schemas.py` | Pydantic 契约（ChatRequest/Response、ToolStep、SSE 事件、Predict、Rag、Tools、Health） |
| `api/deps.py` | 可选 API-Key 鉴权（`.env` 设 `API_KEY` 时启用 `X-API-Key`） |
| `api/routers/chat.py` | `POST /api/chat`（4 模式）+ `POST /api/chat/stream`（SSE） |
| `api/routers/forecast.py` | `POST /api/predict`、`/api/predict/chronos` |
| `api/routers/rag.py` | `POST /api/rag/search`（混合检索 + 引用溯源） |
| `api/routers/tools.py` | `GET /api/tools`（10 工具注册表自省） |
| `api/routers/memory.py` | `POST /api/memory`（长期记忆查询） |
| `api/routers/health.py` | `GET /api/health`（状态 + 4 范式 + RAG 就绪） |

**关键设计**
- **SSE 事件协议**：`step_running` / `step_done` / `response_chunk` / `response_done` / `error`
- **契约化**：全部走 Pydantic → 自动生成 OpenAPI（`/docs` 可交互）
- **懒加载**：`_IncludedRouter` 与惰性模型（见「Bug 修复 #4」）

### 2. 多 Agent Supervisor（`agent/supervisor_agent.py`）

- **Coordinator**：LLM 路由到 `WeatherAgent` / `ForecastAgent` / `ResearchAgent` / `chat`
- **专家 Agent**：各自持有聚焦工具集（气象 / 预测 / RAG 检索）
- **Synthesizer**：汇总专家输出为结构化回答
- `parse_route(raw)`：**抽成纯函数**（容忍 markdown 包裹、前后文、非法 JSON，兜底 `chat`）—— 便于单测
- 预测结果在**工具层**直接写入长期记忆（前端无关，API/Chainlit 都生效）

### 3. 长期记忆（`memory/`）

| 文件 | 说明 |
|------|------|
| `memory/store.py` | SQLite 三表：`user_preferences` / `prediction_history` / `conversation_memory`，线程安全（uvicorn 多线程） |
| `memory/service.py` | 偏好启发式提取（地区/语言/货币/时间跨度）+ 记忆注入 prompt + 预测历史 |

### 4. 可观测性（`observability/trace.py`）

- `TraceRecorder`：`start_run` / `tool` / `end_run` + `run()` 上下文管理器
- **可插拔后端**：
  - `JsonlTraceBackend`（默认，写 `trace.jsonl`，离线可用）
  - `LangfuseTraceBackend`（设 `LANGFUSE_*` 环境变量 + 安装 langfuse 后自动切换）
- 观测永不阻断主流程（写入失败静默）

### 5. 评测体系（`eval/`）

| 文件 | 说明 |
|------|------|
| `eval/cases.py` | 17 用例 × 7 类场景（chat/weather/forecast/rag/trading/alerts/briefing），含期望工具 + 关键词 + judge 提示 |
| `eval/runner.py` | 4 范式执行器 + `judge_result()`（LLM 1-5 分评审） |
| `eval/report.py` | Markdown 报告 + 控制台摘要 |
| `eval/run_eval.py` | CLI：`--mode` / `--quick` / `--no-judge` / `--timeout` |

### 6. 测试与 CI

| 文件 | 说明 |
|------|------|
| `tests/conftest.py` | 项目根导入 + fixture（临时 DB/trace）+ `pytest_collection_modifyitems` 自动跳过无凭据的 integration 用例 |
| `tests/test_date_parse.py` | 日期解析（24 参数化用例 + 单点/周序列预测） |
| `tests/test_tools_registry.py` | 工具注册表 + 失败语义回归 |
| `tests/test_memory.py` | 记忆三层 + 跨实例持久化 |
| `tests/test_trace.py` | Trace 生命周期 + 容错 |
| `tests/test_config.py` | 配置与提示词加载 |
| `tests/test_schemas.py` | Pydantic 契约校验 |
| `tests/test_eval_framework.py` | 评测集完整性 + 报告生成 |
| `tests/test_supervisor_routing.py` | Supervisor 路由解析（12 参数化） |
| `tests/test_api.py` | FastAPI 集成（health/tools/memory + 2 个 live 用例） |
| `pytest.ini` | 配置 + `integration` 标记 |
| `.github/workflows/ci.yml` | 三 job：`test` / `eval`（配 secret 才跑）/ `syntax` |

---

## 三、Bug 修复记录（根因 → 修复 → 验证）

### #1 Chronos-T5 下载阻塞（工具调用卡 170s）

- **根因**：`agent_tools` 在函数内设 `HF_HUB_OFFLINE`，但 huggingface_hub 在 **import 时**已读取该常量 → 设置无效 → 无缓存时反复重试不可达的 HF 端点
- **修复**：把 `HF_HUB_OFFLINE=1` 提到模块顶部（任何 hf 相关 import 之前）+ `_try_load_chronos()` 离线快速失败
- **效果**：单次预测工具调用 **200s → 11s**

### #2 ReAct 模式只会复读用户输入

- **根因**：`middleware.py` 的 `@dynamic_prompt` 非报告分支返回了**路由用的系统提示词**，覆盖了 ReAct 提示词
- **修复**：新增 `prompts/react_prompt.txt` + `load_react_prompts()`，middleware 返回正确的 ReAct 提示词
- **效果**：ReAct 产出真实回答（预测查询 9.6s）

### #3 Supervisor 路由恒为 chat（提示词占位符未替换）

- **根因**：`COORDINATOR_PROMPT` 里写成 `{{user_input}}`，在 `.format()` 中是 **转义**（输出字面 `{user_input}`），用户输入从未注入 → 模型只能瞎猜
- **修复**：改为 `{user_input}`（JSON 示例保留 `{{...}}` 转义）；并把路由解析抽为 `parse_route()`
- **效果**：4 类查询路由全部正确（预测→ForecastAgent、天气→WeatherAgent、政策→ResearchAgent、闲聊→chat），**tool-hit 100%**

### #4 `ChatOpenAI` 构造时校验 key 导致无法导入（CI 必挂）

- **根因**：`model/factory.py` 在**模块级**创建 `chat_model`，而 `ChatOpenAI` 在**构造时**就校验 `api_key` → 无凭据环境（CI/测试）连 import 都失败；`RagSummarizeService` 也在模块级实例化并访问模型
- **修复**：
  - `model/factory.py`：引入 `_LazyModel` 惰性代理（首次属性访问/调用才真正创建），**调用方零改动**
  - `agent/tools/agent_tools.py` + `api/routers/rag.py`：`RagSummarizeService` 改惰性单例 `_get_rag()`
  - `rag/rag_service.py`：模型不再组合进 LangChain `|` 管道（惰性代理不是 Runnable，会被误判为 callable 包装成 RunnableLambda → `'ChatOpenAI' object is not callable`），改为显式 `self.model.invoke(prompt)`
- **效果**：无凭据环境全部模块可导入；CI 可跑测试

### #5 Plan-Execute 模式完全不可用（双重格式化）

- **根因**：`PLANNER_PROMPT` / `SYNTHESIZE_PROMPT` 是 **f-string**（日期已插入），又被 `.format()` 二次格式化 → f-string 阶段把 `}}` 转为 `}`（单括号）→ `.format()` 遇到单 `}` 抛 `ValueError: Single '}' encountered in format string`
- **修复**：去掉 `f` 前缀，日期改 `{today}` 占位符；JSON 示例统一为 `.format` 单层转义 `{{ }}`；两处 `.format()` 传 `today=...`
- **效果**：Plan-Execute 由 **0/11 → 11/11 OK**（tool-hit 91%）

### #6 日期解析不支持「未来一周」

- **根因**：`parse_forecast_end_date` 只认 ISO / 明天 / 后天 / N天后，不认「未来一周」「下周」「一星期」「next week」
- **修复**：新增周/月语义（周→7 天，`N个月后` 中文数字→30N 天，含 `_cn_number()` 中文数字转换）
- **效果**：「预测一下未来一周电价」正确返回 **7 天预测表**（此前报错）

### #7 预测失败返回硬编码假值 48.88

- **根因**：`predict_electricity_price` 异常分支 `return 48.88` → LLM 把假数据当真实电价回答用户
- **修复**：改为 `raise ValueError(...)`（失败要响亮），由 agent 侧感知并降级到可用工具
- **效果**：假数据从回答中消失

### #8 评测工具会整体挂死

- **根因**：单轮卡在挂起的 LLM 调用上无超时；`ThreadPoolExecutor.future.cancel()` 无法终止已运行线程
- **修复**：**模型层** 加 `timeout=60, max_retries=1`（源头掐断）+ **评测层** 单轮 150s 超时 + judge 30s 超时 + `-u` 无缓冲输出
- **效果**：评测可稳定跑完（此前两次挂死）

---

## 四、4 范式评测实测数据

| Mode | OK% | Tool-hit% | Avg Judge | Avg Latency |
|------|-----|-----------|-----------|-------------|
| **Supervisor** | 11/11 (100%) | **100%** | **3.09** | 25.1s |
| **Plan-Execute** | 11/11 (100%) | 91% | 2.82 | 43.1s |
| **LangGraph** | 11/11 (100%) | 82% | 2.73 | 17.0s |
| **ReAct** | 11/11 (100%) | 18%* | 2.36 | 16.8s |

> *ReAct 的工具命中为统计口径差异（工具调用不进入文本事件流），日志确认其正确调用工具。
> 完整分场景数据见 `docs/EVAL_RESULTS.md`。

---

## 五、测试与 CI 结果

```
有凭据（本地）  : 75 passed            (~50-60s，含 2 个真 LLM 集成用例)
无凭据（CI 场景）: 73 passed, 2 skipped (~12s，integration 自动跳过)
```

CI 三 job：
1. `test`：全量 pytest（无 secret 也能跑）
2. `eval`：Agent 评测（配置 `DASHSCOPE_API_KEY` secret 后触发，产物 `eval_report.md` artifact）
3. `syntax`：`compileall` 冒烟

---

## 六、文档与仓库收尾

| 项 | 说明 |
|----|------|
| `.gitignore` | 补全运行产物（`agent_memory.db` / `trace.jsonl` / `debug.log` / `eval_report.md` / `md5.text`）与个人目录（`.streamlit/` / `.dsh-vision-toolkit/`），`.env` + `chroma_db_v2/` 早已忽略 |
| `requirements.txt` | **重构**：原为 430 行 conda `pip freeze` 快照（含 `file:///C:/...` 本地路径，CI 必挂）→ 改为引用运行时清单 |
| `requirements_deploy.txt` | 补全缺失核心依赖：`langchain-openai` / `fastapi` / `uvicorn` / `torch` / `openai` |
| `requirements-dev.txt` | 新建（pytest 等开发/CI 工具） |
| `.env.example` | 补全 `API_KEY` / `LANGFUSE_*` / `AGENT_CHROMA_PERSIST_DIR` / 各配置说明 |
| `README.md` | 新增 API 端点表、工程层说明、Testing & CI 章节；项目结构图重写（补 api/memory/observability/eval/tests/.github）；Key Innovations 扩为 5 条 |
| `docs/RESUME_BULLETS.md` | 简历话术（7 大 bullet + 面试追问准备 + 技术栈关键词） |
| `docs/EVAL_RESULTS.md` | 4 范式实测对比 + 分场景细节 + 已知改进点 |

---

## 七、如何运行

```bash
# 1) 依赖
pip install -r requirements.txt -r requirements-dev.txt

# 2) 配置
cp .env.example .env      # 填入 DASHSCOPE_API_KEY

# 3) 知识库 + 模型（首次）
python -c "from rag.vector_store import VectorStoreService; VectorStoreService().load_document()"
python -c "from algorithms.epf_predictor import train_model; train_model()"

# 4) 启动
chainlit run chainlit_app.py --port 8000            # 前端（4 范式 + 记忆展示）
python -m uvicorn api.main:app --port 8001          # Agent API（/docs 可交互）
streamlit run app.py --server.port 8501             # 备选前端

# 5) 测试与评测
python -m pytest                                    # 75 用例
python -m pytest -m "not integration"               # 离线快速套件
python eval/run_eval.py --quick                     # 4 范式评测 → eval_report.md
```

---

## 八、简历表述（可直接引用）

- 基于 **LangGraph** 实现 4 种 Agent 范式（动态+反思 / 计划-执行 / ReAct / **Supervisor 多 Agent 分层编排**），运行时切换并横向对比
- **Supervisor 工具命中率 100%**（vs 单 Agent 82%），用数据证明多 Agent 编排的收益
- **FastAPI 服务化**：REST + SSE 流式，Pydantic 契约自动 OpenAPI，可选 API-Key 鉴权
- **长期记忆**：SQLite 双级记忆（用户偏好 + 预测历史）自动注入 prompt，跨会话生效
- **可观测性**：可插拔 Trace 层（本地 JSONL ↔ Langfuse），记录 run/tool/耗时
- **评测体系**：17 用例 × 4 范式对比 + LLM Judge 自动打分，支持回归
- **工程化**：75 个 pytest 用例（凭据感知分层，CI 无密钥可跑），GitHub Actions 三 job，Docker 部署
- **疑难排查**：定位并修复 8 个真实缺陷（HF 离线标志失效、提示词双重格式化、模型构造期校验导致不可导入等）

---

## 九、发布前安全审计（pre-publish audit）

上传 GitHub 前执行完整审计（`git init` + `git check-ignore -v` + 全库检索），**发现并修复 2 个真实风险**：

### 风险 1：`.gitignore` 行内注释导致规则失效（严重）

- **现象**：`agent_memory.db` / `trace.jsonl` / `md5.text` / `eval_report.md` 明明写进了 `.gitignore`，却仍出现在待提交清单
- **根因**：**gitignore 不支持行尾注释**——`agent_memory.db          # SQLite long-term memory` 整行被当成模式，永远匹配不到文件
- **修复**：注释移到独立行并加说明防复发；补充忽略 `.chainlit/translations/`（23 个自动生成 i18n 文件，降噪）
- **验证**：14 个关键路径逐一 `git check-ignore -v` 确认忽略；待提交文件 **120 → 93**

### 风险 2：硬编码的个人信息

| 位置 | 内容 | 修复 |
|------|------|------|
| `scripts/sync_to_server.py` | 默认值硬编码服务器 IP（2 处，含 print） | 改为必须设置 `EPF_SERVER`，未设置则清晰报错退出 |
| `scripts/check_server.py` | 同上（2 处） | 同上 |
| `scripts/weekly_update.bat` | 硬编码个人路径 `d:\python\project\Agent Project` | 改为默认取脚本父目录（`%~dp0..`），可用 `EPF_PROJECT_DIR` 覆盖 |

### 审计结论

| 检查项 | 结果 |
|--------|------|
| `.env`（5 个真实凭据） | ✅ 已忽略 |
| 密钥外泄到其他文件 | ✅ 无（全库检索无匹配） |
| 运行产物（记忆库/日志/追踪/报告） | ✅ 已忽略 |
| `chroma_db_v2/`、`logs/`、`.streamlit/`、`.dsh*` | ✅ 已忽略 |
| 大文件（>1MB） | ✅ 无（最大 `chroma_db_v2` 0.9MB） |
| 硬编码密钥 / 密码 / IP | ✅ 无（服务器 IP 已清除） |
| 模型文件（`lstm_epf*.pth`，各 0.2MB） | ✅ 纳入提交（开箱可运行） |
| 待提交文件总数 | **93 个**，结构干净 |

### 已知取舍（非阻塞）

- `data/*.txt|pdf|docx`（RAG 知识库）不提交——需自行放入，`data/README.md` 已说明构建方式
- `datasets/*.xlsx`、`algorithms/merged_price_weather.csv` 不提交（数据授权 / 可重新生成）
- `Dockerfile` 的 `LABEL maintainer="zhushuyu"` 保留（如需匿名可自行改）
