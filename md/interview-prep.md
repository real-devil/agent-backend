# 面试准备：高频追问与应对

> **核心原则：先贴标签，再讲细节。** 每个项目第一句话就要让面试官知道——线上产品、公司能力、还是个人探索。

---

## 〇、给自己看的真相表（别背给面试官，心里要有数）

| 项目 | 性质 | 你做的 | 做到哪 | 有用户吗 |
|------|------|--------|--------|---------|
| **SoulLove 情感陪伴** | 公司线上产品 | 全栈，Dify 编排对话/记忆 | **真上线**，有流量 | ✅ 有 |
| **多模态 SD 生图** | 公司线上能力（SoulLove 产品内） | **我搭建的** RunPod + ComfyUI + Dify 联动 | **已接入**，用户在用 | ✅ 有 |
| **多 Agent 编排引擎** | 个人技术探索（当前 repo） | 独立写的 FastAPI + LangGraph | 测试环境跑通，能 demo | ❌ 没有 |

> 多模态简历单列是为了突出技术栈，**实际就是 SoulLove 产品里的出图能力**，不是第三个孤立项目。面试时可以说清楚。

### 多模态：简历功能 vs 真实状态

| 功能 | 状态 | 面试怎么说 |
|------|------|-----------|
| RunPod GPU 实例 + ComfyUI 部署 | ✅ 做了 | "我部署的，容器化跑在 RunPod" |
| SD 生图工作流（文生图） | ✅ 做了 | "workflow JSON 编排，能出图" |
| Dify HTTP 节点联调 | ✅ 做了 | "对话里触发，用户不用跳出聊天" |
| LoRA / ControlNet 调参 | 🟡 看实际参与 | 做过就说"调过 CFG、采样器、LoRA 风格一致性" |
| 批量生成脚本 | 🟡 看实际 | 做过就说，没做过别主动提 |
| 前置网关（Token 绑工作流） | 🟡 看有没有 | 被问再说，multimodal.md 里有口径 |

### 自研引擎：简历功能 vs 真实状态

| 功能 | 状态 | 面试怎么说 |
|------|------|-----------|
| LangGraph 14 节点 StateGraph | ✅ 做了 | "图编译跑通，能 demo" |
| 五阶段编排 | ✅ 做了 | "复杂任务走全链路，能演示" |
| 4 种 Agent + 并行执行 | ✅ 做了 | "group 内 gather" |
| SSE + LangSmith | ✅ 做了 | "前端能看执行过程" |
| 安全层 / RAG | ✅ 做了 | "规则拦截 + pgvector 检索" |
| Human-in-the-loop 审批 | 🟡 实验 | "跑通过，没接业务" |
| Checkpointer | 🟡 一半 | "memory 在用，Postgres 代码写好没切" |
| 接 SoulLove 用户流量 | ❌ 没做 | **绝对别说接了** |

---

## 一、30 秒破冰（开场必说）

> 我简历上三个 AI 项目，实际两类：
>
> **第一，SoulLove 情感陪伴**——公司线上产品，我负责全栈，**有真实用户**。对话和记忆走 Dify；里面的**多模态出图也是我搭的**——RunPod 上 ComfyUI 跑 SD，Dify HTTP 节点调用，用户在聊天里就能生图。
>
> **第二，多 Agent 编排引擎**——我**业余时间独立自研**的探索项目，FastAPI + LangGraph，**没有用户**，测试环境跑通，用来深入理解 Agent 编排。和 SoulLove 主链路无关。
>
> 简历把多模态单列，是因为 ComfyUI / SD / RunPod 这套技术栈值得单独写。本质上就是 SoulLove 产品能力的一部分。

---

## 二、速查表（面试前 5 分钟看）

### 三个项目一句话

| 项目 | 一句话 |
|------|--------|
| SoulLove | 线上产品，Dify 驱动，我全栈，**有用户** |
| 多模态 SD 生图 | **我搭建的**，RunPod + ComfyUI，已接入 SoulLove，**有用户在用** |
| 多 Agent 引擎 | 我自研的探索项目，**无用户**，测试环境可 demo |

### 高频坑位

| 面试官问 | 直接答 |
|---------|--------|
| 多模态上线了吗？ | **上了。** RunPod 部署 ComfyUI，Dify 工作流 HTTP 调用，SoulLove 用户在聊天里用。 |
| 多模态谁做的？ | **我搭建的。** 部署、工作流编排、Dify 联调都是我。 |
| 多 Agent 上线了吗？ | **没有。** 测试环境跑通，能 demo。SoulLove 用户走 Dify，不走自研引擎。 |
| 三个项目什么关系？ | SoulLove 是产品（含多模态出图）；多 Agent 是独立探索，和主产品无关。 |
| 五阶段日常聊天用吗？ | **不用。** 闲聊 Dify 直调 LLM。 |
| 审批业务用了吗？ | **没有。** 实验跑通过，架构预埋。 |

### 不要主动提

| 话题 | 应对 |
|------|------|
| 小模型语义审核 | "留了接口，没实现" |
| Postgres checkpointer | "代码支持，当前 memory 模式" |
| 多模态前置网关 | 做了就说，没做被问再答 |

---

## 三、核心口供（Q&A）

### 3.1 SoulLove

**Q：架构？**

> Nuxt3 + Nitro 全栈，Dify 编排对话/记忆/多模态，Supabase 存数据，Redis 去重。PWA 跨端。**线上产品，有用户。**

**Q：多模态在 SoulLove 里怎么接的？**

> Dify 里配了多模态工作流。用户聊天触发 → LLM 生成 prompt → HTTP 节点调 RunPod 上的 ComfyUI API → SD 出图 → 图片 URL 回聊天界面。**整套链路我搭的。**

### 3.2 多模态（我搭建的，可以大胆说）

**Q：你做了什么？**

> 三块：**部署**——RunPod 上容器化跑 ComfyUI，配好 API 鉴权；**工作流**——SD 文生图 workflow JSON 编排，调过 CFG、采样器、LoRA 风格一致性；**联调**——Dify HTTP 节点对接，用户在 SoulLove 聊天里直接出图，不用跳页面。

**Q：Dify 和 ComfyUI / RunPod 怎么联动？**

> Dify 工作流加 HTTP 请求节点，调 RunPod 暴露的 ComfyUI API。请求体传 workflow JSON，Header 带 Bearer Token，ComfyUI 在 GPU 上执行生图，返回图片 URL。

**Q：RunPod 鉴权？和 Dify 调用有什么区别？**

> ComfyUI 启动参数配网页密码和 API Key，两套独立鉴权。调用方 Bearer Token 准入。RunPod 原生 Token 只管"能不能进"，不管"调哪个工作流"——要前置网关做 Token 和工作流绑定。Dify 的 Token 内置匹配工作流，更省事。

**Q：简历为什么单独列？**

> 技术栈和对话产品不一样——ComfyUI、SD、LoRA、GPU 部署。单列让面试官一眼看到多模态能力。实际就是 SoulLove 产品里的出图模块。

### 3.3 多 Agent 引擎（个人探索，别云雾）

**Q：这项目到底是什么？上线了吗？**

> **业余时间独立自研，没有用户。** FastAPI + LangGraph，测试环境跑通，五阶段编排、4 种 Agent、SSE、安全护栏、RAG 都能 demo。目的是搞懂 Agent 编排底层，不是替代 SoulLove 的 Dify。

**Q：和 SoulLove 什么关系？**

> **没关系，零用户流量。** SoulLove 走 Dify。自研引擎是独立 repo。

**Q：什么时候从单 Agent 变多 Agent？**

> **探索过程中遇到的。** 先用 FC 单 Agent 搭原型，后来要 group 并行、审查回滚，单 loop 做不了，才拆状态机。不是 SoulLove 线上逼的。

**Q：是不是炫技？**

> 闲聊不需要。要按 step 并行、按 group 回滚、审查重试才需要状态机。可观测是副产品，不是拆的理由。

**Q：审批呢？**

> step 级，Planner 标 `approval_required` 就暂停。跑通过，没接业务。

### 3.4 简历关键词（多 Agent 项目）

| 关键词 | 说法 |
|--------|------|
| 双图架构 | 逻辑双图，一张图条件路由；安全节点独立目录+开关 |
| 五阶段 | Planner→Approval→Execute→Review→Synthesize，能 demo |
| Human-in-the-loop | 实验验证，架构预埋，无业务场景 |

---

## 四、深挖补充（被追问才用）

### 双图

逻辑双图，非物理。14 节点不需要子图 state 开销。

### 单 Agent → 多 Agent

- **根因**：并行、回滚、审查重试 → 要状态机
- **副产品**：trace 按节点分开

### 多模态技术细节

- ComfyUI = 编排工具，SD = 模型引擎，RunPod = GPU 云租赁
- 工作流本质是 JSON，可版本管理、可复用
- 小体量 <1000 请求/天：单 GPU 实例 + Nginx 反代够用

### 安全层（多 Agent 项目）

关键词 + 限流 + 熔断，纯代码毫秒级。小模型位置预留，未实现。
