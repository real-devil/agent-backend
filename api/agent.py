from fastapi import APIRouter
from pydantic import BaseModel

from agents.service import get_agent_session_state, resume_agent_session, run_agent_session

router = APIRouter()


class AgentRequest(BaseModel):
    input: str
    session_id: str | None = None
    document_id: str | None = None


class AgentResumeRequest(BaseModel):
    session_id: str
    approval_response: str
    input: str | None = None


@router.post("/run")
async def run_agent(request: AgentRequest):
    return await run_agent_session(
        user_input=request.input,
        session_id=request.session_id,
        document_id=request.document_id,
    )


@router.post("/resume")
async def resume_agent(request: AgentResumeRequest):
    return await resume_agent_session(
        session_id=request.session_id,
        approval_response=request.approval_response,
        user_input=request.input,
    )


@router.get("/state/{session_id}")
async def get_agent_state(session_id: str):
    return await get_agent_session_state(session_id)
