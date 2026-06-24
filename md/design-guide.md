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

---

## 五、三种 Agent 模式的本质

### 表面是三种，本质是两个维度

| 模式 | 代表产品 | 有无 Graph？ | 下一步怎么定？ |
|------|---------|------------|--------------|
| A（Plan-then-Execute） | LangGraph、CrewAI | ✅ 启动前画完整 DAG | Planner 一次性决定全流程 |
| B（单流 ReAct） | ChatGPT、Claude.ai | ❌ 无 | LLM 边想边决定何时调工具 |
| C（动态多 Agent） | Cursor、Devin | ❌ 无 | 每步执行完，LLM 看结果再决定下一步 |

### 本质分类

```
          有无 Graph
          │
    ┌─────┴─────┐
    │           │
  有 Graph    无 Graph
  （A 模式）   │
         ┌────┴────┐
         │         │
      单 Agent   多 Agent
      （B 模式） （C 模式）
      一气呵成   改一步看一步
```

**三种模式的差异只取决于两个决策：**

1. **要不要提前画好执行图？** 要 → A；不要 → B 或 C
2. **Agent 数量？** 1 个 → B；多个 → C

### A vs C 的核心区别

A 和 C 都是多 Agent，但：

| | A（你的项目） | C（Cursor） |
|---|---|---|
| **后续步骤** | 提前确定，写死在 plan 里 | 不知道，靠上一步结果动态决定 |
| **适用场景** | 流程固定、步骤可枚举（RAG→审核→导出） | 步骤依赖上一步结果（改代码→发现新 bug→再改） |
| **优势** | 可审计、可审批、可提前优化并行 | 灵活，能应对未知情况 |
| **劣势** | 计划可能跟不上实际情况 | 不可预测，难以审计 |

**一句话**：A 是画好地图再走，C 是走一步看一步。

### 选型指南

| 场景 | 选哪个 | 原因 |
|------|--------|------|
| 流程固定、步骤可枚举 | A | RAG + 审核 + 导出报告，步骤事先就知道 |
| 步骤依赖上一步结果 | C | 改代码、修 bug，改完才知道有没有新问题 |
| 简单问答 | B | 不需要拆任务，单 LLM 直接回答 |
| 企业内部 Agent 平台 | A 为主 | 需要审计、审批、可观测 |
| 代码/IDE Agent | C 为主 | 需要灵活应对未知的代码问题 |

### 面试时怎么说

面试官不会问"ABC 三种模式的定义"，会问实际决策：

> "为什么你们选了预规划而不是动态执行？"
>
> → 因为我们的场景是 RAG + 审核工作流，步骤可枚举，需要人工审批节点。预规划可以提前展示计划给用户确认，也能优化并行执行。
>
> "什么时候该用动态规划？"
>
> → 当步骤依赖上一步结果才能决定时，比如代码修复——改完一处可能发现新问题，无法事先规划全流程。

---

## 六、A 与 C 是同一种能力，只是调用节奏不同

### 核心洞察

A 的 plan 和 C 的每一步决策，**本质上都是 LLM 在做规划**。区别只在一个维度：

```
A: LLM 一次性输出 [step_A, step_B, step_C] → 系统按图执行
C: LLM 输出 step_A → 执行 → 看结果 → 输出 step_B → 执行 → ...
```

| | LLM 什么时候输出后续步骤 |
|---|---|
| A | 一开始就全部输出 |
| C | 做完一步再输出下一步 |

**A 模式并没有更"智能"，C 模式也没有更"动态"——只是把 N 次 LLM 调用合并成了一开始的一次 Planner 调用。**

### 所以三种模式其实是一种

```
所有 Agent 系统 = LLM 做决策 + 工具执行

  B: 1 次 LLM 调用，LLM 自己决定何时调工具、何时结束
  A: N 次 LLM 调用，但 N 个 step 的规划在第一次就全部做完（Planner）
  C: N 次 LLM 调用，每次只决定下一步（ReAct 循环）

区别只是：后续 step 的规划是在第一次调用时预产出，还是分散到每次调用中临时产出。
```

### Cursor 的本地 vs 云端分工

Cursor 安装包不大，因为本地不跑 LLM：

```
本地（IDE 扩展）              云端（Cursor 后端 / API）
─────────────────────        ─────────────────────────
执行 tool：                  决策：
  grep → 返回匹配             看到结果 → 决定 read_file
  read_file → 返回代码        看到代码 → 决定 edit_file
  edit_file → 执行替换        输出 old_string + new_string
```

**本地只做执行，云端做所有决策。** 跟你项目里的 `tool_agent` 一样——LLM 远程调（`AsyncOpenAI()`），tool 本地跑。区别只是 Cursor 的 tool 更多、prompt 更精妙。

### 工程启示

不要纠结"选 A 还是选 C"。实际生产中可以**混用**：

- 顶层用 A：Planner 拆大任务（"修复这个模块的 bug"→ 定位、分析、修复、测试）
- 每个 step 内部用 C：`code_agent` 在"修复"这个 step 里自己 grep → read → edit

这就是**分层 Agent**：上层预规划，下层动态执行。大方向可控，细节灵活应对。

---

## 七、ReAct 才是纯种循环，Planner 不是

### ReAct 的本质

```
ReAct = Reasoning（推理） + Acting（执行工具）

一个死循环：
  LLM 思考 → 调工具 → 拿到结果 → 再思考 → 再调工具 → ...
  直到 LLM 觉得够了，不再调工具，输出最终答案
```

```python
# 纯 ReAct（B/C 模式）
while True:
    response = llm.call(messages, tools=TOOLS)
    if not response.tool_calls:
        return response.content          # LLM 觉得够了，停止
    for tool in response.tool_calls:
        result = execute(tool)
        messages.append(result)
    # 循环...直到 LLM 不再调工具
```

### 三种模式里谁才是 ReAct

| 模式 | 是不是 ReAct？ | 原因 |
|------|--------------|------|
| **B（ChatGPT）** | ✅ 纯 ReAct | LLM 想一步 → 调工具 → 再想一步 |
| **C（Cursor）** | ✅ 纯 ReAct | 每一步决定下一步，标准循环 |
| **A 的 Planner** | ❌ 不是 | 一次调用输出完整计划，没有"观察→再决策" |
| **A 的 tool_agent** | ✅ 是 ReAct | `for _ in range(MAX_TOOL_ITERATIONS)` 就是循环 |

### A 模式其实是 ReAct 外面包了一层

```
A 模式 = Planner（非 ReAct，一次性规划）
       + tool_agent（ReAct 循环，最多 5 轮）
       + Reviewer（非 ReAct，一次判断）
       + Synthesizer（非 ReAct，一次汇总）
```

**Planner 没有循环**——它不会"执行一步→看结果→修正计划"。如果计划错了，靠 Reviewer 的 retry/rollback 兜底，而不是 Planner 自己修正。这跟 ReAct 的"边做边调整"本质不同。

### 你项目早期的单 Agent 就是纯 ReAct

最早的 `api/agent.py` 就是一个 `for _ in range(5)` 循环——LLM 调工具、拿结果、再调、直到不再调或超限。那才是纯 ReAct。后来加了 LangGraph 的 Planner/Reviewer/Synthesizer，就变成了**有监督的多 Agent 系统**。
