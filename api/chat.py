from fastapi import APIRouter
from pydantic import BaseModel

from agents.service import run_agent_session

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    document_id: str | None = None


@router.post("/")
async def chat(request: ChatRequest):
    return await run_agent_session(
        user_input=request.message,
        session_id=request.session_id,
        document_id=request.document_id,
    )
