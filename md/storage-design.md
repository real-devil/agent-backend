# 存储设计：会话记忆与对话历史

## 一、两张表的分工

```
conversations（业务表）              checkpoints（运行状态表）
────────────────────────            ──────────────────────────
给前端看                            给 LangGraph 恢复执行
  id                                  thread_id (=session_id)
  user_id                             checkpoint_id
  session_id                          当前 AgentState 全量 JSONB
  title                              上次节点切换时的快照
  last_message                       审批卡在哪一步
  created_at / updated_at

前端查这表 → 展示对话列表             LangGraph 查这表 → 恢复执行 / resume
```

**职责不同，互不替代。** conversations 给用户看"我聊过什么"，checkpoints 给系统恢复"上次跑到哪了"。session_id 是两张表共同的外键。

---

## 二、checkpoints 表（LangGraph 自动管理）

### 不用手写建表 SQL

LangGraph 的 `AsyncPostgresSaver.setup()` 自动建表：

- `checkpoints` — 主表，thread_id + checkpoint_id + state（JSONB）
- `checkpoint_writes` — 写入中间态
- `checkpoint_blobs` — 大字段拆片存储

### 什么时候写入

```
每次节点切换时 LangGraph 自动序列化当前 AgentState → 写一行
  不需要手动 save，不需要手动 load
```

### 什么时候查询

```
graph.ainvoke(input, config)
  → 按 thread_id 从 checkpoints 读最近一次 checkpoint
  → 反序列化 → 继续执行

graph.aget_state(config)
  → 按 thread_id 读当前 state（轮询用）
```

### 内存模式 vs Postgres 模式

```
memory:  InMemorySaver → Python dict，单进程有效
postgres: AsyncPostgresSaver → PG 表，集群共享，重启不丢
```

切换方式：`.env` 设 `LANGGRAPH_CHECKPOINTER=postgres` + 配数据库连接串。

---

## 三、conversations 表（业务表，需手动建）

### 表结构

```sql
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    session_id TEXT NOT NULL UNIQUE,
    title TEXT,                              -- "北京的天气怎么样"
    last_message TEXT,                        -- 回复摘要
    workflow_status TEXT DEFAULT 'active',    -- active / completed
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_updated_at ON conversations(updated_at DESC);
```

### 什么时候写入

每次 `run_agent_graph` 或 `resume_agent_graph` 结束时：

```python
# runtime.py 的 run_agent_graph() 返回后
await upsert_conversation(
    user_id=user_id,
    session_id=session_id,
    title=extract_title(user_input),
    last_message=final_reply[:200],
)
```

### 什么时候查询

前端对话列表页：

```sql
SELECT * FROM conversations
WHERE user_id = $1
ORDER BY updated_at DESC
LIMIT 20;
```

---

## 四、完整记忆体系

```
┌─────────────────────────────────────────────────────┐
│ 短期记忆（单轮上下文）                                 │
│   AgentState.messages — 当前 turn 的原始对话消息        │
│   状态：✅ 已实现，Annotated + add 自动追加             │
├─────────────────────────────────────────────────────┤
│ 中期记忆（跨轮对话）                                    │
│   AgentState.turn_history — 每轮完成后的快照            │
│   状态：✅ 已实现，存完整快照                            │
│   规划：滑动窗口摘要 — 当 messages 或 turn_history       │
│   超过阈值时，调 LLM 将旧轮压缩为摘要，清掉原始消息        │
│   保留摘要 + 最近的 N 轮完整内容，防止上下文窗口爆掉       │
├─────────────────────────────────────────────────────┤
│ 长期记忆（跨会话）                                      │
│   conversations 表 — 对话列表（标题 + 摘要）             │
│   checkpoints 表 — 每个会话的完整状态快照               │
│   状态：❗ checkpoints 代码就绪；conversations 待建表     │
└─────────────────────────────────────────────────────┘
```

### 滑动窗口摘要（已规划，待实现）

当前 `turn_history` 存的是每轮完整快照（含 prompt、plan、metrics），不做压缩。商业级需要加入自动摘要：

```
触发条件：messages 超过 N 条 或 turn_history 超过 M 轮
    ↓
调 LLM（用独立的 summary prompt）：
  "将以下历史对话压缩为 200 字摘要，保留关键信息和用户偏好"
    ↓
存入 turn_history_summary
    ↓
清掉 messages 里的旧消息（只留最近 K 条）
    ↓
后续 Planner/Reviewer 的 context 里塞入摘要而非全量历史
```

当前 demo 阶段 gpt-4o-mini 的 128K 窗口足够，未触发压缩需求。

---

## 五、当前项目状态

| 功能 | 状态 | 备注 |
|------|------|------|
| checkpoints 持久化 | ✅ 代码就绪 | 设 postgres 即可 |
| conversations 表 | ❌ 未建 | 需手动建表 + 后端写入逻辑 |
| 用户体系 | ❌ 未建 | 需登录/注册/鉴权 |
| 前端对话列表 | ❌ 未建 | 依赖 conversations 表 |
| memory 模式多轮对话 | ✅ 可用 | 单进程内有效 |

---

## 六、面试常见追问

**Q: 为什么 checkpoints 和 conversations 不合并成一张表？**

> checkpoints 表存的是序列化后的完整 AgentState（JSONB），动辄几十 KB，不适合给前端分页查询。conversations 只存标题、摘要，轻量级列表查询。职责分离——状态恢复跟展示列表是两个不同的读写模式。

**Q: 对话历史怎么在服务重启后恢复？**

> 用 postgres checkpointer。每个 thread_id 的 state 快照存在 PG 的 checkpoints 表里，`graph.ainvoke` 时自动从 PG 读取最近一次快照恢复。重启不丢，集群共享。

**Q: 用户换了设备怎么看到之前的对话？**

> 依赖用户体系 + conversations 表。用户登录后按 user_id 查对话列表 → 点进去 → 前端拿 session_id 调 `/agent/state/{session_id}` → 后端从 checkpoints 恢复完整 state。
