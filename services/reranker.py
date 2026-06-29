"""Reranking module — post-retrieval precision filter.

召回阶段（retriever）偏重覆盖率：多捞一些候选片段，宁多勿少。
重排序阶段（本模块）偏重精度：对候选片段重新打分，筛掉不相关的，只留最准的几条。

当前使用 LLM 批量打分方案（免模型下载，适合 demo）。
生产环境可替换为：
  - 本地 Cross-Encoder（sentence-transformers，~90MB，单条 <5ms）
  - BGE Reranker（FlagEmbedding，中文效果好）
  - Cohere Rerank API（商业方案）
"""

import logging
import os

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

RERANK_PROMPT = """You are a search relevance judge.
For the user's query below, score how relevant each document chunk is.

User query: {query}

{chunks_text}

Return ONLY a JSON array of integers, where each integer is the relevance score (0-10).
0 = completely irrelevant, 10 = perfectly answers the query.
Example: [8, 2, 10, 0]

Scores:"""


async def rerank_chunks(
    query: str,
    chunks: list[str],
    top_k: int = 3,
) -> list[str]:
    """对候选片段重新打分，只返回最相关的 top_k 条。

    Args:
        query: 用户问题
        chunks: 召回阶段拉回来的所有候选片段
        top_k: 最终保留几条

    Returns:
        筛选后的片段列表（保持原顺序但只留高分项）
    """
    if len(chunks) <= top_k:
        return chunks  # 候选本来就不多，不用筛

    # 拼成带编号的文本
    chunks_text = "\n".join(
        f"[{i}] {chunk[:300]}" for i, chunk in enumerate(chunks)
    )

    prompt = RERANK_PROMPT.format(query=query, chunks_text=chunks_text)

    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # 打分任务不需要创造性
        )
        raw = response.choices[0].message.content or "[]"

        # 提取 JSON 数组
        import json
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start != -1 and end > start:
            scores = json.loads(raw[start:end])
        else:
            scores = json.loads(raw)

        # 分数 × 片段 → 排序 → 取 top_k
        scored = list(zip(scores, chunks))
        scored.sort(key=lambda x: x[0], reverse=True)

        top = [chunk for _, chunk in scored[:top_k] if _ >= 3]  # 低于 3 分的不要

        if not top:
            # 全部低分 → 降级，至少返回一条
            logger.warning("Reranker: all chunks scored below 3, returning top 1")
            return [scored[0][1]]

        logger.info("Reranker: %d → %d chunks (top score=%d)", len(chunks), len(top), scored[0][0])
        return top

    except Exception as exc:
        logger.warning("Reranker failed, falling back to unre-ranked chunks: %s", exc)
        return chunks[:top_k]  # 挂了就降级：直接取前 top_k 条
