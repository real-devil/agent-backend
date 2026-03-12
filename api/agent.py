from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class AgentRequest(BaseModel):
    input: str
    session_id: str | None = None


@router.post("/run")
async def run_agent(request: AgentRequest):
    # TODO: 接入 LLM + Tools
    return {"output": "todo", "session_id": request.session_id}
