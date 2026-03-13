from services.embedder import embed_text
from db.supabase_client import get_client


def retrieve_chunks(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
) -> list[str]:
    """将问题向量化后从 Supabase 检索最相关的文档片段"""
    query_embedding = embed_text(query)
    db = get_client()

    params: dict = {"query_embedding": query_embedding, "match_count": top_k}
    if document_id:
        params["filter_document_id"] = document_id

    result = db.rpc("match_chunks", params).execute()
    return [row["content"] for row in result.data]
