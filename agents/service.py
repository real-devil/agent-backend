import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from agents.runtime import get_workflow_snapshot, resume_agent_graph, run_agent_graph


def _snapshot_to_session_state(session_id: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "workflow_status": snapshot.get("workflow_status"),
        "route_reason": snapshot.get("route_reason"),
        "current_group_index": snapshot.get("current_group_index"),
        "pending_approval_group": snapshot.get("pending_approval_group"),
        "review_decision": snapshot.get("review_decision"),
        "review_reason": snapshot.get("review_reason"),
        "artifacts": snapshot.get("artifacts"),
        "metrics_summary": snapshot.get("metrics_summary"),
        "workflow_trace": snapshot.get("workflow_trace"),
        "workflow_plan": snapshot.get("workflow_plan"),
    }


def _format_sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


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


async def stream_agent_session(
    user_input: str,
    session_id: str | None = None,
    document_id: str | None = None,
) -> AsyncIterator[str]:
    active_session_id = session_id or str(uuid4())
    task = asyncio.create_task(
        run_agent_graph(
            user_input=user_input,
            session_id=active_session_id,
            document_id=document_id,
        )
    )

    yield _format_sse_event("session", {"session_id": active_session_id})

    emitted_trace_count = 0
    last_status: str | None = None
    try:
        while not task.done():
            snapshot = await get_workflow_snapshot(active_session_id)
            trace = snapshot.get("workflow_trace") or []
            while emitted_trace_count < len(trace):
                yield _format_sse_event(
                    "trace",
                    {
                        "session_id": active_session_id,
                        "trace": trace[emitted_trace_count],
                    },
                )
                emitted_trace_count += 1

            current_status = snapshot.get("workflow_status")
            if current_status != last_status:
                yield _format_sse_event(
                    "state",
                    _snapshot_to_session_state(active_session_id, snapshot),
                )
                last_status = current_status

            await asyncio.sleep(0.5)

        reply = await task
        snapshot = await get_workflow_snapshot(active_session_id)
        trace = snapshot.get("workflow_trace") or []
        while emitted_trace_count < len(trace):
            yield _format_sse_event(
                "trace",
                {
                    "session_id": active_session_id,
                    "trace": trace[emitted_trace_count],
                },
            )
            emitted_trace_count += 1

        yield _format_sse_event(
            "state",
            _snapshot_to_session_state(active_session_id, snapshot),
        )
        yield _format_sse_event(
            "final",
            {
                "session_id": active_session_id,
                "reply": reply,
            },
        )
    except Exception as exc:
        if not task.done():
            task.cancel()
        yield _format_sse_event(
            "error",
            {
                "session_id": active_session_id,
                "detail": str(exc),
            },
        )


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


async def stream_resume_agent_session(
    session_id: str,
    approval_response: str,
    user_input: str | None = None,
) -> AsyncIterator[str]:
    task = asyncio.create_task(
        resume_agent_graph(
            session_id=session_id,
            approval_response=approval_response,
            user_input=user_input,
        )
    )

    yield _format_sse_event("session", {"session_id": session_id})

    emitted_trace_count = 0
    snapshot = await get_workflow_snapshot(session_id)
    trace = snapshot.get("workflow_trace") or []
    emitted_trace_count = len(trace)
    last_status = snapshot.get("workflow_status")

    try:
        while not task.done():
            snapshot = await get_workflow_snapshot(session_id)
            trace = snapshot.get("workflow_trace") or []
            while emitted_trace_count < len(trace):
                yield _format_sse_event(
                    "trace",
                    {
                        "session_id": session_id,
                        "trace": trace[emitted_trace_count],
                    },
                )
                emitted_trace_count += 1

            current_status = snapshot.get("workflow_status")
            if current_status != last_status:
                yield _format_sse_event(
                    "state",
                    _snapshot_to_session_state(session_id, snapshot),
                )
                last_status = current_status

            await asyncio.sleep(0.5)

        reply = await task
        snapshot = await get_workflow_snapshot(session_id)
        trace = snapshot.get("workflow_trace") or []
        while emitted_trace_count < len(trace):
            yield _format_sse_event(
                "trace",
                {
                    "session_id": session_id,
                    "trace": trace[emitted_trace_count],
                },
            )
            emitted_trace_count += 1

        yield _format_sse_event(
            "state",
            _snapshot_to_session_state(session_id, snapshot),
        )
        yield _format_sse_event(
            "final",
            {
                "session_id": session_id,
                "reply": reply,
            },
        )
    except Exception as exc:
        if not task.done():
            task.cancel()
        yield _format_sse_event(
            "error",
            {
                "session_id": session_id,
                "detail": str(exc),
            },
        )


async def get_agent_session_state(session_id: str) -> dict[str, Any]:
    snapshot = await get_workflow_snapshot(session_id)
    return _snapshot_to_session_state(session_id, snapshot)
