import os

from openai import AsyncOpenAI

from prompts.templates import SYSTEM_PROMPT, build_context_message
from services.retriever import retrieve_chunks

DEFAULT_MODEL = "openai/gpt-4o-mini"


def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def _get_model_name() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


async def rag_chat(
    question: str,
    document_id: str | None = None,
    top_k: int = 5,
    *,
    rerank: bool = False,
    rerank_top_k: int = 3,
) -> str:
    """RAG 问答：召回 → (可选) 重排序 → LLM 生成。

    Args:
        top_k: 召回阶段取多少条（rerank=True 时建议设大，如 15-20）
        rerank: 是否启用重排序
        rerank_top_k: 重排序后保留几条
    """
    chunks = retrieve_chunks(question, top_k=top_k, document_id=document_id)

    # 重排序：广撒网 → 精筛选
    if rerank and len(chunks) > rerank_top_k:
        from services.reranker import rerank_chunks
        chunks = await rerank_chunks(question, chunks, top_k=rerank_top_k)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chunks:
        messages.append({"role": "system", "content": build_context_message(chunks)})
    messages.append({"role": "user", "content": question})

    response = await _get_client().chat.completions.create(
        model=_get_model_name(),
        messages=messages,
    )
    return response.choices[0].message.content
