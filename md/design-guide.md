# Agent 设计原则与改进方向

## 一、前端思考过程展示的时机

**对，就是在 Synthesizer 之前。** 前端看到的"思考过程"来自两个阶段：

### 阶段 1：Planner 思考流（thinking 通道）

```
用户输入 → Planner LLM 开始思考
  → SSE 流式推送 "event: thinking"   ← 前端实时显示 Planner 的思考过程
  → 内容来自 prompt 里的 <thinking>...</thinking> 标签
```

代码路径：[core/plan_parser.py](agents/core/plan_parser.py) → `call_planner_model()` → `stream_text_model(on_delta=...)` → `append_thinking()` 写入缓冲区 → `service.py` 轮询消费 → SSE 推送

### 阶段 2：Step 执行 trace（trace 通道）

```
Planner 生成 plan → 开始执行 step
  → 每个 step_started 事件 → SSE "event: trace" → 前端显示 "Document Q&A · 查询迪迦资料"
  → 每个 tool_called 事件 → SSE "event: trace" → 前端显示 "Get weather"
  → 每个 step_completed 事件 → SSE "event: trace" → 前端显示完成状态
  → reviewer_completed → group_advanced → 下一组...
```

### 阶段 3：Synthesizer 回复流（reply 通道）

```
所有 step 跑完 → Synthesizer 开始
  → SSE 流式推送 "event: reply_delta"  ← 前端显示最终回复
```

**总结**：用户看到的过程是 `Planner 思考 → Step 执行进度 → Synthesizer 最终回复`。思考在前，结果在后。

---

## 二、Agent 应按领域组织，而非按 Tool 组织

### 当前问题

```python
# ❌ 当前：一个 Agent 绑定 1-2 个 tool
research_agent: [search_documents]          # 研究=搜文档？太窄
tool_agent:     [get_weather, search]        # "工具 agent" 语义模糊
rag_agent:      []                           # 内部调 rag_chat，tool 不可见
general_agent:  []                           # 纯 LLM，无 tool
```

**问题**：
- `research_agent` 只能搜文档，不能搜网络、查 API、读外部数据源
- `tool_agent` 名字太泛——"工具"是什么工具？
- 新增能力时不知道归到哪个 agent

### 正确做法：按业务领域划分

```
agent_impl/
├── weather_agent.py      # 气象领域
│   tools: get_current_weather, get_forecast, get_air_quality,
│          get_humidity, get_wind_speed
│
├── document_agent.py     # 文档/知识库领域
│   tools: search_documents, list_chunks, get_document_meta,
│          summarize_document, compare_documents
│
├── code_agent.py         # 代码/开发领域
│   tools: read_file, edit_file, grep_code, run_test,
│          git_diff, list_directory
│
├── data_agent.py         # 数据分析领域
│   tools: query_db, run_sql, generate_chart, export_csv
│
├── web_agent.py          # 网络/外部数据领域
│   tools: web_search, fetch_url, check_status
│
└── reasoning_agent.py    # 通用推理/写作（无需 tool）
    tools: []
```

### 设计原则

| 原则 | 说明 |
|------|------|
| **一个 Agent = 一个领域** | Agent 的名字应该描述它的业务能力，而非技术实现 |
| **一个领域 = 多个 Tool** | 同一领域的不同操作封装为不同 tool |
| **Agent 内部做 ReAct 循环** | 跟当前 `tool_agent` 一样：LLM 决定调哪个 tool，循环直到产出结果 |
| **Planner 按领域选 Agent** | prompt 里描述各 Agent 的领域能力，LLM 自动选择合适的 Agent |
| **Tool 是 Agent 的私有财产** | 不同 Agent 可以有自己的同名 tool，不共享全局 tool 列表 |

### Planner prompt 的对应变化

```python
PLANNER_PROMPT = """
Available agents:
- weather_agent: 气象数据查询（当前天气、预报、空气质量等）
- document_agent: 文档知识库检索与分析
- code_agent: 代码读取、编辑、搜索、测试
- data_agent: 数据库查询与数据分析
- web_agent: 网络搜索与外部数据获取
- reasoning_agent: 通用推理、写作、转换（无工具）

Rules:
- 根据用户意图分配合适的领域 agent
- 同一领域的多个操作放在同一个 step 中，由 agent 内部决定调用哪些 tool
- 不同领域的操作拆成不同 step，可设置相同 parallel_group 并行执行
"""
```

### 当前项目的迁移路径

1. 重命名现有 agent 文件，按领域命名
2. 将相关 tool 归入对应 agent
3. 每个 agent 内部实现 ReAct 循环（复用 `tool_agent` 的模式）
4. 更新 Planner prompt 中的 agent 列表
5. 更新 `executor.py` 的 `dispatch_step` 分支

---

## 三、Synthesizer 的本质

Synthesizer 就是一个**汇总器**——将所有 step 的产出一次性喂给 LLM，生成最终回复。

```python
# 输入
- 用户原始问题
- 完整的 workflow_plan
- 所有被 reviewer 接受的 step_results
- 所有 artifact（结构化数据）

# 输出
- 流式生成的最终回复（给用户看）
```

它不是"再次推理"，而是"翻译"——把结构化的执行结果翻译成用户友好的自然语言回复。如果 step 结果已经足够完整，Synthesizer 只是整理语言；如果不够，它会诚实告知。

---

## 四、文件精准修改的实现原理（Cursor 为例）

### 核心机制

不是让 LLM 输出整个文件，而是输出**精确的文本替换指令**：

```json
{
  "file_path": "src/foo.ts",
  "old_string": "要替换的精确文本片段",
  "new_string": "替换后的新文本"
}
```

### 工作流程

1. LLM 读取目标文件，看到需要修改的代码段
2. LLM 输出 `old_string`（文件中必须精确存在的片段）+ `new_string`（替换内容）
3. 系统在文件中查找 `old_string`，找到则替换
4. 找不到则报错，让 LLM 重试

### 关键点

- **old_string 必须唯一**：如果匹配到多处，编辑失败
- **包含足够的上下文**：LLM 需要输出周围足够多的代码来保证唯一性
- **失败重试机制**：匹配失败时把错误返回给 LLM，让它重新读文件再试

### 在 Agent 系统中的实现

如果要在当前项目中加入代码编辑能力，就是加一个 `code_agent`：

```python
# agent_impl/code_agent.py
TOOLS = [
    read_file,       # 读取文件内容
    edit_file,       # old_string → new_string 替换
    grep_code,       # 搜索代码
    run_command,     # 执行命令
]
```

LLM 先 `read_file` 定位要改的代码 → 输出 `edit_file(old_string, new_string)` → 系统执行替换。
