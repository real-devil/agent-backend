from uuid import uuid4

from agents.runtime import get_workflow_snapshot, resume_agent_graph, run_agent_graph


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


async def resume_agent_session(
    session_id: str,
    approval_response: str,
    user_input: str | None = None,
) -> dict[str, str | None]:
    reply = await resume_agent_graph(
        session_id=session_id,
        approval_response=approval_response,
        user_input=user_input,
    )
    return {"reply": reply, "session_id": session_id}


async def get_agent_session_state(session_id: str) -> dict[str, str | int | list | None]:
    snapshot = await get_workflow_snapshot(session_id)
    return {
        "session_id": session_id,
        "workflow_status": snapshot.get("workflow_status"),
        "route_reason": snapshot.get("route_reason"),
        "current_group_index": snapshot.get("current_group_index"),
        "pending_approval_group": snapshot.get("pending_approval_group"),
        "review_decision": snapshot.get("review_decision"),
        "review_reason": snapshot.get("review_reason"),
        "workflow_plan": snapshot.get("workflow_plan"),
    }
