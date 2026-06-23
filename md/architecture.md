# Agent Backend 系统架构

## 一、项目定位

基于 **LangGraph StateGraph** 的多 Agent 工作流后端。采用 **Planner → Execute → Review → Synthesize** 四阶段模式，通过 SSE 实时推送执行过程给前端。

**从单 Agent 到多 Agent 的演进**：原始版本是一个简单的 ReAct 循环（单 LLM + 工具调用），重构后变成 8 个节点的 LangGraph 状态机，支持并行执行、人工审批、回滚、流式可观测性。

---

## 二、整体架构

```
┌──────────────────────────────────────────────────┐
│  api/                 薄转发层                    │
│  agent.py / chat.py  → 只做参数解析和路由转发       │
├──────────────────────────────────────────────────┤
│  agents/service.py    业务编排层                  │
│  SSE 轮询 + 状态快照 + 格式转换                    │
├──────────────────────────────────────────────────┤
│  agents/runtime.py    公共入口 + 生命周期           │
│  agents/graph.py      StateGraph 编译 + 条件路由   │
│  agents/nodes/        8 个图节点                   │
│  agents/agent_impl/   4 种 Agent 实现              │
├──────────────────────────────────────────────────┤
│  agents/core/         工具基础设施（9 模块）         │
├──────────────────────────────────────────────────┤
│  services/ + tools/ + prompts/   底层能力层        │
│  文档解析 / 向量检索 / RAG / 天气查询（不变）       │
└──────────────────────────────────────────────────┘
```

---

## 三、核心：LangGraph 状态机

### 图结构

```
START → entry → planner → approval_gate → execute_group → reviewer
                   ↑            ↑              ↑   ↓(retry)    ↓(continue)
                   │            │              │ rollback    advance_group
                   │            │              └──────────────┘
                   │            │ (awaiting/rejected)
                   │            └→ END
                   │
                   └→ (resume 状态：跳过 planner 直达 approval_gate)
                                    ↓
                              synthesizer → END
```

### 8 个节点职责

| 节点 | 职责 | 调 LLM？ |
|------|------|---------|
| `entry` | 路由：新会话→planner，resume→approval_gate | 否 |
| `planner` | LLM 将用户意图拆解为 step list，指定分组和依赖 | **是（流式）** |
| `approval_gate` | 检查是否需要人工审批，否则放行 | 否 |
| `execute_group` | `asyncio.gather` 并行执行同 group 的所有 step | 通过 agent 调 |
| `reviewer` | LLM 审查本组结果 → continue/retry/finish | **是** |
| `advance_group` | current_group_index += 1 | 否 |
| `rollback_group` | 回滚到指定 group，截断后续 step_results | 否 |
| `synthesizer` | 汇总所有 artifacts → 流式生成最终回复 | **是（流式）** |

### 4 种 Agent

| Agent | 能力 | Tool |
|-------|------|------|
| `research_agent` | 有文档时直接检索，无文档时 LLM 分析 | search_documents |
| `tool_agent` | ReAct 循环（最多 5 轮） | get_weather, search_documents |
| `rag_agent` | Supabase 向量检索 + LLM 回答 | 无（内部调 rag_chat） |
| `general_agent` | 纯 LLM 推理/写作/转换 | 无 |

---

## 四、工作流生命周期

### 一次请求的完整路径

```
用户输入
  → entry: 路由判断
  → planner: LLM 生成 workflow_plan (step list)
    → prompt 指令 + JSON 输出格式约束
    → _normalize_plan 清洗（字段缺失→默认值、agent 非法→general、group 压缩）
    → 兜底：关键词检测 fallback
  → approval_gate: 检查 approva_required
  → execute_group: asyncio.gather 并行执行
    → dispatch_step: 按 agent 类型分发
  → reviewer: LLM 审查 → continue/retry/finish
    → continue: advance_group → 下一组
    → retry: 重试或回滚
    → finish: synthesizer → 最终回复
```

### SSE 流式推送（service.py）

```
asyncio.create_task(run_agent_graph)    # 后台跑 LangGraph
     │
while not task.done():                  # 轮询循环
    snapshot = graph.aget_state()       # 从 checkpointer 拉状态
    yield "event: trace"                # 增量推送 trace 事件
    yield "event: thinking"             # 流式 thinking delta
    yield "event: reply_delta"          # 流式 reply delta
    yield "event: state"                # 状态变化时推送完整快照
    sleep(60ms 活跃 / 400ms 空闲)
     │
yield "event: final"                    # 最终结果
```

---

## 五、关键设计决策

### 1. Planner + 关键词双层兜底
- **主路径**：LLM Planner 生成 step list（准确但可能出错）
- **兜底**：关键词检测 fallback（`routing.py`，快但不准）
- **修正**：`reconcile_plan_steps` 修正明显错误（如天气路由到 rag_agent）
- **清洗**：`_normalize_plan` 处理 LLM 输出的各种格式问题

### 2. 结构化输出
每个 step 输出 JSON：`{summary, artifact_type, artifact_data, confidence}`
- 在 agent 间通过 `artifacts` dict 传递
- `output_key` 作为 key 存入 artifacts

### 3. 流式双通道
- `thinking` 通道：Planner 的思考过程
- `reply` 通道：Synthesizer 的最终回复
- 线程安全的内存缓冲（`stream_buffer.py`）

### 4. 状态持久化
- `memory` 模式：InMemorySaver（重启丢失）
- `postgres` 模式：AsyncPostgresSaver（持久化）
- 通过 `LANGGRAPH_CHECKPOINTER` 环境变量切换

### 5. 超时保护
- 300s 全局超时（`_invoke_with_timeout`）
- 超时后写 `timed_out` 状态并推送 event

---

## 六、文件导航

| 要做什么 | 改哪个文件 |
|---------|-----------|
| 加新 Agent 类型 | `agent_impl/` 新增 + `prompts.py` 加 prompt + `executor.py` 加分发 |
| 加新图节点 | `nodes/` 新增 + `graph.py` 加 add_node/add_edge |
| 改 LLM 模型 | `core/llm.py` 的 `get_model_name()` |
| 调 prompt | `core/prompts.py` |
| 加新 tool | `tools/` 新增 + `agent_impl/tool_agent.py` 的 TOOLS 列表 |
| 改审批逻辑 | `nodes/approval.py` |
| 加 trace 事件 | `core/trace_utils.py` + `trace_labels.py` |
| 改 SSE 推送频率 | `service.py` 的 `asyncio.sleep()` |

---

## 七、核心概念速查

```
Step            最小任务单元：{id, agent, goal, parallel_group, depends_on, output_key}
Group           并行执行单元：相同 parallel_group 的 step 同时跑
Artifact        Step 的结构化产出，通过 output_key 在 agent 间传递
Trace Event     工作流执行事件，SSE 推送给前端展示进度
Turn            一轮对话，包含一次完整的 Planner→Synthesize 周期
Workflow Plan   Planner 生成的 step 列表，驱动整个执行流程
Reviewer        每 group 执行后的质量闸门：continue/retry/finish/rollback
Checkpointer    LangGraph 的状态快照存储（memory 或 postgres）
```
