import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from services.pdf_parser import parse_pdf
from services.word_parser import parse_word
from services.chunker import chunk_text
from services.embedder import embed_batch
from db.supabase_client import get_client

router = APIRouter()

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".doc")


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="只支持 PDF 和 Word 文件")

    file_bytes = await file.read()
    document_id = str(uuid.uuid4())

    # 解析 → 切块 → 向量化
    if file.filename.endswith(".pdf"):
        text = parse_pdf(file_bytes)
    else:
        text = parse_word(file_bytes)
    chunks = chunk_text(text)
    embeddings = embed_batch(chunks)

    # 存入 Supabase
    db = get_client()
    rows = [
        {
            "document_id": document_id,
            "filename": file.filename,
            "content": chunk,
            "embedding": embedding,
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]
    db.table("document_chunks").insert(rows).execute()

    return {
        "document_id": document_id,
        "filename": file.filename,
        "chunks": len(chunks),
    }


@router.get("/list")
def list_documents():
    db = get_client()
    result = db.table("document_chunks").select("document_id, filename").execute()
    seen = {}
    for row in result.data:
        doc_id = row["document_id"]
        if doc_id not in seen:
            seen[doc_id] = row["filename"]
    return [{"document_id": k, "filename": v} for k, v in seen.items()]
