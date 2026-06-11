from uuid import uuid4

from agents.runtime import run_agent_graph


async def run_agent_session(
    user_input: str,
    session_id: str | None = None,
    document_id: str | None = None,
) -> dict[str, str | None]:
    active_session_id = session_id or str(uuid4())
    reply = await run_agent_graph(
        user_input=user_input,
        session_id=active_session_id,
        document_id=document_id,
    )
    return {"reply": reply, "session_id": active_session_id}
