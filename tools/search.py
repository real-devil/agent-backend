from services.retriever import retrieve_chunks

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "在用户上传的文档知识库中检索相关内容。"
            "当用户询问文档、资料、合同、报告等具体内容时调用此工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用于检索的关键词或问题",
                },
                "document_id": {
                    "type": "string",
                    "description": "可选，限定检索某个具体文档的 ID",
                },
            },
            "required": ["query"],
        },
    },
}


async def search_documents(query: str, document_id: str | None = None) -> str:
    chunks = retrieve_chunks(query, top_k=5, document_id=document_id)
    if not chunks:
        return "知识库中未找到相关内容。"
    return "\n\n".join(f"[片段 {i+1}]\n{c}" for i, c in enumerate(chunks))
