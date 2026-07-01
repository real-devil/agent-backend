# AI 多模态生成：概念、工具与部署

## 一、Stable Diffusion 是什么

**Stable Diffusion（SD）是一类基于扩散算法的文本生成图像模型。**

- 核心原理：训练时逐步加噪破坏图片，推理时从噪声逐步去噪还原图片
- Stability AI 公司发布并命名，多个版本：SD 1.5 / 2.0 / SDXL / SD 3
- 跟 GPT 的关系：GPT = 文本进文本出，SD = 文本进图片出。都是训练好的模型文件
- 文件格式：通常用 `.safetensors`（安全张量格式），也可用 `.ckpt`。文件格式 ≠ 模型类别

**相近但不同的概念：**

- SD ≠ 模型规范/协议，就是类似 GPT-4 一样具体的模型文件
- 同类的还有 FLUX、Midjourney、DALL-E
- 生视频的有 Sora、Runway Gen-3、CogVideoX、AnimateDiff

---

## 二、ComfyUI 是什么

**ComfyUI = 带 WebUI 的本地 Python 程序，用于编排 SD 生图工作流。**

```
ComfyUI/
├── python_embeded/       ← 自带 Python 解释器
├── main.py               ← 启动入口
├── nodes.py              ← 节点定义（加载模型、采样、保存...）
└── models/               ← 模型文件放这
    ├── checkpoints/      ← SD 大模型（如 sd_xl_base_1.0.safetensors，6GB+）
    ├── loras/            ← LoRA 微调模型
    └── vae/              ← VAE 模型
```

**启动后做的事：**

① `run_nvidia_gpu.bat` → Python 进程启动 → 加载模型到 GPU
② 浏览器打开 `http://127.0.0.1:8188` → WebUI
③ WebUI 上拖节点、连线 → 本质是定义 JSON 工作流
④ 点 "Queue Prompt" → 把 JSON 发给后台 → 执行生图 → 返回图片

**ComfyUI ≠ SD。** ComfyUI 是方向盘（编排工具），SD 模型是引擎（跑在 GPU 上的神经网络）。

**工作流可保存为 JSON 文件，可分享、可复用、可版本控制。**

---

## 三、RunPod 是什么

**RunPod = 一家 GPU 云租赁公司。** 不是规范、不是协议、不是工作流工具——就是你花钱租一台带 GPU 的远程服务器。

```
RunPod 实例 = 一台云端的 Linux 服务器
  ├── GPU: A100 80GB（租的）
  ├── CPU: 16 核
  ├── 内存: 64GB
  └── 硬盘: 100GB

你可以在这台机器上：
  → 装 ComfyUI（生图）
  → 装 CogVideoX（生视频）
  → 装 vLLM（跑大语言模型）
  → 装任何你能装的东西
```

**跟你租阿里云 ECS 一样，只是这台带 GPU，按时付费。**

---

## 四、部署架构

### 小体量（<1000 请求/天）

```
一台 GPU 云服务器（RunPod / 阿里云 GPU 实例）
  ├── ComfyUI（Python 进程）
  └── Nginx 反代 → 对外提供 HTTP API

前端/Agent 直接调 API → 同步返回图片
```

### 中大体量（>10000 请求/天）

```
API Gateway
    │
    ▼
消息队列（Redis / RabbitMQ）
    │
    ▼
GPU Worker × N（K8s + 多实例 ComfyUI 容器）
    │
    ▼
返回 task_id → 前端轮询 → 拿到图片 URL

RunPod Serverless 自动扩缩容，高峰扩、低谷缩到零
```

### 结合 Agent 平台

```
用户跟 Agent 对话 → Agent 决定需要出图
    → Dify 多模态工作流节点被触发
    → HTTP 请求 → ComfyUI API → 出图 → 返回图片 URL
    → Agent 把图片展示给用户
```

---

## 五、自建 vs API 的决策理由


| 理由       | 说明                                               |
| -------- | ------------------------------------------------ |
| **成本**   | 日请求量大时，API 按张收费太贵。自建 GPU 包月单张图片成本从几毛降到几分         |
| **可控性**  | 自建可用 LoRA 风格模型、调 CFG/采样器参数、拼多阶段工作流，API 给不了这种粒度   |
| **数据安全** | API 调用意味所有 prompt 和图片经过第三方。公司数据不能外传，自建部署在 VPC 内网 |
| **统计埋点** | 自建 Dify/ComfyUI 可做埋点、统计留存、SSO 登录，数据自主可控          |


---

## 六、常见参数速查


| 参数           | 含义             | 常用值                            |
| ------------ | -------------- | ------------------------------ |
| CFG（提示词引导系数） | 模型多听 prompt 的话 | 7，过低自由发挥，过高画面过饱和               |
| Steps        | 去噪步数           | 20-30，过多收益递减                   |
| Sampler（采样器） | 怎么从噪声还原出图      | DPM++ 2M Karras（主流），Euler a（快） |
| LoRA         | 微调模型，控制风格一致性   | 按需下载，加载到已有大模型上                 |


---

## 七、模型选型：checkpoint / LoRA / VAE

### 三类文件各干什么


| 目录             | 作用          | 业务上怎么用                                     |
| -------------- | ----------- | ------------------------------------------ |
| `checkpoints/` | 底模，决定画风大方向  | 写实系或二次元系，常用 SD 1.5 / SDXL                  |
| `loras/`       | 小权重插件，叠在底模上 | **定向控制**：角色脸、姿势、服装、Q 版头像、特定画风等，按训练数据决定 |
| `vae/`         | 解码器，影响颜色和细节 | 底模发灰时换 `vae-ft-mse`（1.5）或 `sdxl_vae`（SDXL） |


### 典型组合

```
Checkpoint（写实或二次元底模）
    + LoRA（按需求叠：角色脸 / 姿势 / 服装 / Q 版头像 / 画风等，可多个叠加）
    + 可选：面部修复、高清放大节点
    + VAE（颜色正常、细节更好）
```

LoRA **不是只能锁角色**——训练数据是什么，就偏向控制什么：有人脸 LoRA 固定长相，有姿势 LoRA 固定构图，有服装 LoRA 固定穿搭，Q 版头像也可以靠专门 LoRA 拉风格。实际项目里常 **多个 LoRA 叠加**，各自带 weight 权重。

情感陪伴类产品的出图需求：**角色立绘、换装、场景氛围、Q 版头像**——工程上是 **底模 + 若干 LoRA（脸/姿势/服装/风格按需叠）+ Prompt 模板 + 工作流批量出图**，不是算法岗从零训基座。

### 选型原则（面试口径）

- **SDXL vs 1.5**：SDXL 画质好、显存大；1.5 快、LoRA 生态多——按显卡成本和速度选
- **为什么用 LoRA**：比全量 fine-tune 成本低、迭代快；训练数据决定控制什么（脸、姿势、服装、Q 版等），可多个叠加
- **上线后**：工作流和 LoRA 组合 **基本冻结**，主要调 CFG、Steps、Prompt 模板，追求 **风格稳定和脸不崩**

### 被问「具体用的什么模型名」

> 具体 checkpoint 名字时间久了记不清了。当时选型是：**底模定画风，LoRA 按需求叠（脸/姿势/服装/Q 版等），VAE 调色**；上线后组合基本固定，主要调参数和 Prompt，没有频繁换底模。

说 **类别**（写实/二次元底模 + 若干 LoRA 控制脸/姿势/服装/画风），不必背具体 `.safetensors` 文件名。

### LoRA 下载平台

| 站点 | 地址 | 说明 |
|------|------|------|
| **Civitai 官方** | https://civitai.com | 国外最大 SD/LoRA 社区；中国大陆等地区 IP 会强制 SFW，设置里可能 **没有** Mature Content 开关 |
| **Civitai Green** | https://civitai.green | 官方纯绿色版，**永远 SFW**，不要和完整版搞混 |
| **Civitai Red** | https://civitai.red | 社区镜像站，**关闭内容过滤后可解锁 R / X / XXX**；国内访问、选型比官方 `.com` 直观 |
| **LiblibAI** | https://www.liblib.art | 国内常用，很多 Civitai 搬运；设置里开敏感内容后下载 |

**Civitai Red 使用要点：**

1. 打开 https://civitai.red（不是 `.com` / `.green`）
2. 登录账号（可与 Civitai 生态通用或单独注册，以站点为准）
3. **关闭内容过滤**，浏览级别勾选 **R、X、XXX**
4. 搜 LoRA → 看 model card、trigger words、示例图 → 下载 `.safetensors` 到 `ComfyUI/models/loras/`

**三个域名别搞混：**

```
civitai.com   → 官方，地区限制时很「绿」
civitai.green → 官方绿色站，永远保守
civitai.red   → 镜像站，关过滤后可看 R/X/XXX（开发选型用）
```

面试口径：

> LoRA 主要从 **Civitai** 选型，开发时也会用 **LiblibAI** 或镜像站；看 model card、trigger words 和底模是否匹配，下载后放 `loras/` 目录，上线后组合固定。

---

## 八、Pod vs Serverless

RunPod 有两种完全不同的租法：


|      | **Pod（单节点）**                | **Serverless**                         |
| ---- | --------------------------- | -------------------------------------- |
| 本质   | 租一台 **一直开着的 GPU 电脑**        | **按请求** 自动起 GPU 容器，用完可缩到 0             |
| 你怎么用 | SSH 上去装 ComfyUI，浏览器 `:8188` | 调 `api.runpod.ai/v2/{endpoint_id}/run` |
| 计费   | **开机时长**，没人用也烧钱             | **请求 / 执行时间**，可缩零                      |
| 冷启动  | 模型常驻，**第二次很快**              | 可能等几十秒起容器 + 加载模型                       |
| 适合   | 稳定流量、开发调试、低延迟               | 流量波动、削峰、不想多租 24h Pod                   |


**RunPod 不会自动帮你做「大模型张量并行」**——单 Pod 就是单机推理；Serverless 帮你做的是 **多 Worker 实例扩容**（应用层分布式），不是把一个超大模型拆多卡。

### 怎么区分（五个维度）

**① 看 URL（最准）**

```
Pod：     https://{podId}-{port}.proxy.runpod.net
           例：https://fcbs686lfr9txf-8000.proxy.runpod.net

Serverless：https://api.runpod.ai/v2/{ENDPOINT_ID}/run
           例：https://api.runpod.ai/v2/y6au5s4n94flua
```


| URL 特征                 | 类型                    |
| ---------------------- | --------------------- |
| `proxy.runpod.net`     | **Pod** 上服务的 HTTP 代理  |
| `api.runpod.ai/v2/...` | **Serverless** 官方 API |


**② 看任务 ID**


| ID 样例                                     | 类型                                          |
| ----------------------------------------- | ------------------------------------------- |
| `99f91c9a-98bc-4ce7-bb07-04ca6b9dbe01-u1` | **Serverless Job**（UUID + `-u1`/`-u2` 分区后缀） |
| `job_deb3c4e4373041ed93e38dfe23880991`    | **业务层任务号**（`job_` + 32 位 hex），需看底层调谁        |
| 纯 UUID、无 `-u1`                            | 多半是 **Pod 上 ComfyUI 的 prompt_id**           |


**③ 看控制台**：Pods 页 = Pod；Serverless → Endpoints 页 = Serverless

**④ 看 API 响应**：Serverless 状态常有 `delayTime`、`executionTime`、`workerId`；Pod + ComfyUI 直连没有这套字段

**⑤ 记忆口诀**

```
看到 proxy.runpod.net  → Pod
看到 api.runpod.ai     → Serverless
看到 id 带 -u1 后缀    → Serverless 异步任务
```

### Serverless API 路径


| 路径                 | 作用               |
| ------------------ | ---------------- |
| `/run`             | 异步提交，立刻返回 job id |
| `/runsync`         | 同步等结果            |
| `/status/{job_id}` | 查任务状态            |
| `/health`          | 健康检查             |


---

## 九、生产路由：TypeB_HD 溢出分流

### 路由规则

- **TypeB_SD / TypeC / TypeD / TypeE**：按配置表 **固定路由**，永远不变
- **TypeB_HD**：动态路由，Pod 主力 + Serverless 溢出

```
TypeB_HD（高清）
    │
    ├─ 常态：Pod 代理（fcbs686lfr9txf-8000.proxy.runpod.net）
    │         → 固定节点、模型常驻、响应快
    │
    └─ 当 Pod 上 TypeB_HD 并发/排队 >= 3：
              → 溢出到 Serverless（api.runpod.ai/v2/y6au5s4n94flua）
              → 多 Worker 扛高峰，避免单 Pod 堵死

TypeB_SD / TypeC / TypeD / TypeE
    → 查表固定路由，不参与动态切换
```

### 设计意图


| 说法                 | 是否准确                                     |
| ------------------ | ---------------------------------------- |
| Pod 负责快响应          | ✅ 常驻、无冷启动                                |
| 并发 ≥3 走 Serverless | ✅ 溢出分流（监控的是 **Pod 侧在途+排队**，不是日总量）        |
| 画质降级               | ❌ 不准确——是 **容量降级/过载兜底**，除非两套工作流本身不同       |
| 降低生产成本             | ⚠️ 削峰省钱：高峰按需付 Serverless，不必为尖峰多租 24h Pod |


这是典型的 **「固定节点 + 弹性溢出」（burst to serverless）**，不是简单砍画质。

### 双层任务 ID

```
用户请求
  → 业务层生成 job_deb3c4e4...（给前端/日志）
    → 溢出时内部调 Serverless
      → 拿到 99f91c9a-...-u1（RunPod 侧 job id）
```

---

## 十、面试速答

### ComfyUI 是什么

> 本地部署 SD 推理环境 + 节点式工作流编排。Checkpoint 定画风，LoRA 按训练目标控制脸/姿势/服装/Q 版等（可多个叠加），VAE 调色；上线后组合基本冻结。

### RunPod Pod vs Serverless

> `xxx-8000.proxy.runpod.net` 是 Pod 的 HTTP 代理，模型常驻、延迟低；`api.runpod.ai/v2/{endpoint_id}` 是 Serverless，异步返回带 `-u1` 的 job id，平台自动扩 Worker。

### TypeB_HD 路由

> TypeB_HD 默认走 Pod；监控 Pod 并发，队列达 3 就溢出到 Serverless，避免单卡排队过长。其他 Type 固定走配置表。这是容量层面的弹性，不是砍画质。

### 分布式部署（应用岗）

> 我们体量不大，Pod 跑 ComfyUI 常驻；高峰用 Serverless 多 Worker 扩容。张量并行拆大模型没上过，SD + LoRA 单卡够。大模型对话侧走 API，分布式是云厂商的事。

### LoRA / 后训练

> 做过应用侧 LoRA 和工作流调优（CFG、采样器、ControlNet），LoRA 按训练数据控制脸、姿势、服装、Q 版头像等，不是算法岗全量 SFT。业务优先 Prompt + 工作流，LoRA 负责定向风格和一致性。

### 被问具体模型名

> 名字记不清了，原则是底模 + 若干 LoRA（脸/姿势/服装等）+ VAE；上线后冻结配置，主要调参数和 Prompt。

### 被问「从哪个网站下的」

> 主要从 **Civitai** 下，开发选型用 **civitai.red** 或 **LiblibAI**；看 model card 和 trigger words，具体文件名记不清。

