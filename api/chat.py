from fastapi import APIRouter
from pydantic import BaseModel

from agents.service import resume_agent_session, run_agent_session

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    document_id: str | None = None


class ChatResumeRequest(BaseModel):
    session_id: str
    approval_response: str
    message: str | None = None


@router.post("/")
async def chat(request: ChatRequest):
    return await run_agent_session(
        user_input=request.message,
        session_id=request.session_id,
        document_id=request.document_id,
    )


@router.post("/resume")
async def resume_chat(request: ChatResumeRequest):
    return await resume_agent_session(
        session_id=request.session_id,
        approval_response=request.approval_response,
        user_input=request.message,
    )
