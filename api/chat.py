from fastapi import APIRouter
from pydantic import BaseModel
from services.rag import rag_chat

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    document_id: str | None = None


@router.post("/")
async def chat(request: ChatRequest):
    reply = await rag_chat(request.message, document_id=request.document_id)
    return {"reply": reply, "session_id": request.session_id}
