from fastapi import APIRouter
from pydantic import BaseModel
from api.agent import run_agent, AgentRequest

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    document_id: str | None = None


@router.post("/")
async def chat(request: ChatRequest):
    result = await run_agent(AgentRequest(
        input=request.message,
        session_id=request.session_id,
        document_id=request.document_id,
    ))
    return result
