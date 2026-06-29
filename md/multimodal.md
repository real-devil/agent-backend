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

| 理由 | 说明 |
|------|------|
| **成本** | 日请求量大时，API 按张收费太贵。自建 GPU 包月单张图片成本从几毛降到几分 |
| **可控性** | 自建可用 LoRA 风格模型、调 CFG/采样器参数、拼多阶段工作流，API 给不了这种粒度 |
| **数据安全** | API 调用意味所有 prompt 和图片经过第三方。公司数据不能外传，自建部署在 VPC 内网 |
| **统计埋点** | 自建 Dify/ComfyUI 可做埋点、统计留存、SSO 登录，数据自主可控 |

---

## 六、常见参数速查

| 参数 | 含义 | 常用值 |
|------|------|--------|
| CFG（提示词引导系数） | 模型多听 prompt 的话 | 7，过低自由发挥，过高画面过饱和 |
| Steps | 去噪步数 | 20-30，过多收益递减 |
| Sampler（采样器） | 怎么从噪声还原出图 | DPM++ 2M Karras（主流），Euler a（快） |
| LoRA | 微调模型，控制风格一致性 | 按需下载，加载到已有大模型上 |
