import os
from openai import AsyncOpenAI
from services.retriever import retrieve_chunks
from prompts.templates import SYSTEM_PROMPT, build_context_message


def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


async def rag_chat(
    question: str,
    document_id: str | None = None,
    top_k: int = 5,
) -> str:
    chunks = retrieve_chunks(question, top_k=top_k, document_id=document_id)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chunks:
        messages.append({"role": "system", "content": build_context_message(chunks)})
    messages.append({"role": "user", "content": question})

    response = await _get_client().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"),
        messages=messages,
    )
    return response.choices[0].message.content
