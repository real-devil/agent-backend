from fastapi import APIRouter
from pydantic import BaseModel

from agents.service import run_agent_session

router = APIRouter()


class AgentRequest(BaseModel):
    input: str
    session_id: str | None = None
    document_id: str | None = None


@router.post("/run")
async def run_agent(request: AgentRequest):
    return await run_agent_session(
        user_input=request.input,
        session_id=request.session_id,
        document_id=request.document_id,
    )
