# 模型私有化部署与高并发处理

## 一、私有化部署是什么

模型跑在自己机器上，数据不出自己的服务器。跟调 OpenAI API 的区别只是**把 base_url 从 openai.com 改成自己服务器 IP**。

```
调 API（数据出境）：                         私有化部署（数据自控）：
  你的后端 → https://api.openai.com              你的后端 → http://10.0.1.1:8000
                                                          ↑ 跑在你自己的 GPU 服务器上
```

---

## 二、工具选型

| 工具 | 适用场景 | 一句话 |
|------|---------|--------|
| **vLLM** | 生产部署 | 加载原始模型，暴露 OpenAI 兼容 API，支持高并发 |
| **Ollama** | 开发/调试/个人使用 | 一键拉量化模型，CPU 也能跑 |
| **TGI** | 生产部署（HuggingFace 系） | 功能跟 vLLM 差不多 |

```bash
# vLLM — 生产环境标准答案
vllm serve Qwen/Qwen2.5-7B-Instruct
# → http://0.0.0.0:8000/v1/chat/completions

# Ollama — 开发调试
ollama pull qwen2.5:1.5b && ollama serve
# → http://localhost:11434/v1/chat/completions
```

---

## 三、量化

量化 = **降低每个参数的存储精度，不砍参数数量**。

| 精度 | 每个参数 | 7B 模型大小 | 效果 |
|------|---------|-----------|------|
| FP16（全精度）| 16 位 | ~14GB | 最好 |
| Q8 | 8 位 | ~7GB | 基本无损 |
| Q4 | 4 位 | ~3.5GB | 精度小降，越大模型越抗压 |

- 4B Q4 ≠ 1.5B 模型。4B 参数一个没少，只是存得省。
- Ollama 下载的已经是量化好的（GGUF 格式内置量化）。
- vLLM 是启动时加 `--quantization awq` 动态量化。

---

## 四、高并发 = 加机器 + 负载均衡

### 对等节点架构

每台 GPU 服务器装完整模型、跑完整推理，负载均衡只转发请求。

```
用户请求
    │
    ▼
Nginx / K8s Service（负载均衡，只转发，不跑模型）
    │
    ├── vLLM-1（GPU 服务器 A，装完整模型）
    ├── vLLM-2（GPU 服务器 B，装完整模型）
    └── vLLM-3（GPU 服务器 C，装完整模型）
```

### Nginx 配置

```nginx
upstream llm_cluster {
    least_conn;                                    # 最少连接策略
    server 10.0.1.1:8000  weight=10;               # A100，权重高
    server 10.0.1.2:8000  weight=5;                # T4，权重低
    server 10.0.1.3:8000  max_fails=3  fail_timeout=30s;
    keepalive 32;
}

server {
    listen 80;
    server_name llm.yourdomain.com;

    location / {
        proxy_pass http://llm_cluster;
        proxy_read_timeout 120s;                    # 推理慢，超时设长
    }
}
```

### 三种负载均衡策略

| 策略 | 逻辑 | 场景 |
|------|------|------|
| **least_conn（最少连接）** | 谁忙谁别接新活 | 推理时长不固定，最常用 |
| **轮询** | 挨个发，一人一次 | 推理时长差不多 |
| **权重** | 好显卡多接点 | 混搭 A100 + T4 |

---

## 五、KV Cache 与会话亲和性

### KV Cache 是什么

LLM 生成每个字都要看之前所有字。KV Cache 把前面算过的注意力结果存起来，后面直接复用——**同样 system prompt 第二个请求比第一个快一倍**。

### vLLM 配置

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 4096              # 上下文窗口 → KV Cache 上限
  --gpu-memory-utilization 0.85      # 显存 85% 给模型 + KV Cache
  --enable-prefix-caching            # 同样前缀自动复用
```

### 会话亲和性

同一用户多次请求发到同一台 GPU，KV Cache 命中率高。

```nginx
hash $http_x_user_id;   # 同一用户粘到同一台 GPU
```

**但生产中有权衡**：粘用户 = 缓存命中率高，但负载可能不均衡；不粘 = 负载均衡好，但每次从头算 KV Cache。实际生产中 `enable-prefix-caching` 已经自动识别相同 system prompt 并复用，手动粘用户的收益有限。除非你的应用场景是超长多轮对话。

### 大厂分级调度

70% 的简单请求（"你好""今天几号"）用小模型（CPU 集群跑 Ollama/llama.cpp）直接处理，只有 30% 的复杂请求才路由到大模型 GPU 集群。

---

## 六、显存不够的处理顺序

```
① 降量化：FP16 → Q8 → Q4，精度优先保
② 减上下文窗口：--max-model-len 从 8192 降到 4096
③ 换小模型：7B → 3B → 1.5B
④ 加显卡：以上全做了还不够
```

---

## 七、私有化 vs API 的决策

| 自建私有化 | 调 API |
|-----------|--------|
| 数据不出门，合规 | 数据经过第三方 |
| 初期投入大（GPU 服务器） | 按量付费，初期便宜 |
| 日请求量大时更划算 | 日请求量小时更划算 |
| 需要运维（装 vLLM、配 Nginx） | 零运维 |

**一句话**：单机验证 → Ollama/vLLM；生产集群 → vLLM + Nginx 负载均衡。技术不难，需要的是"知道怎么拼"。
