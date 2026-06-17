import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from agents.runtime import get_workflow_snapshot, resume_agent_graph, run_agent_graph


def _trace_for_turn(trace: list[dict[str, Any]], turn_id: str | None) -> list[dict[str, Any]]:
    if not turn_id:
        return trace
    return [event for event in trace if str(event.get("turn_id") or "") == str(turn_id)]


def _current_turn_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    turn_id = snapshot.get("current_turn_id")
    if not turn_id:
        return None

    reply = snapshot.get("final_reply")
    if not reply:
        for message in reversed(snapshot.get("messages") or []):
            if message.get("role") == "assistant":
                reply = message.get("content", "")
                break

    return {
        "turn_id": turn_id,
        "user_message": snapshot.get("current_turn_user_message", ""),
        "started_at": snapshot.get("current_turn_started_at"),
        "status": snapshot.get("workflow_status"),
        "route_reason": snapshot.get("route_reason"),
        "review_decision": snapshot.get("review_decision"),
        "review_reason": snapshot.get("review_reason"),
        "pending_approval_group": snapshot.get("pending_approval_group"),
        "reply": reply or "",
        "workflow_plan": snapshot.get("workflow_plan") or [],
        "workflow_trace": _trace_for_turn(snapshot.get("workflow_trace") or [], str(turn_id)),
        "artifacts": snapshot.get("artifacts") or {},
        "metrics_summary": snapshot.get("metrics_summary") or {},
    }


def _conversation_turns(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    turns = list(snapshot.get("turn_history") or [])
    current_turn = _current_turn_from_snapshot(snapshot)
    if current_turn is None:
        return turns

    current_turn_id = str(current_turn["turn_id"])
    if any(str(turn.get("turn_id")) == current_turn_id for turn in turns):
        return turns

    return turns + [current_turn]


def _snapshot_to_session_state(session_id: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    current_turn_id = snapshot.get("current_turn_id")
    return {
        "session_id": session_id,
        "current_turn_id": current_turn_id,
        "workflow_status": snapshot.get("workflow_status"),
        "route_reason": snapshot.get("route_reason"),
        "current_group_index": snapshot.get("current_group_index"),
        "pending_approval_group": snapshot.get("pending_approval_group"),
        "review_decision": snapshot.get("review_decision"),
        "review_reason": snapshot.get("review_reason"),
        "artifacts": snapshot.get("artifacts"),
        "metrics_summary": snapshot.get("metrics_summary"),
        "workflow_trace": _trace_for_turn(snapshot.get("workflow_trace") or [], current_turn_id),
        "workflow_plan": snapshot.get("workflow_plan"),
        "conversation_turns": _conversation_turns(snapshot),
    }


def _format_sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _format_trace_sse(session_id: str, trace_event: dict[str, Any]) -> str:
    return _format_sse_event(
        "trace",
        {
            "session_id": session_id,
            "turn_id": trace_event.get("turn_id"),
            "trace": trace_event,
        },
    )


def _snapshot_signature(snapshot: dict[str, Any]) -> str:
    artifacts = snapshot.get("artifacts") or {}
    artifact_digest = {
        key: {
            "artifact_type": value.get("artifact_type"),
            "summary": value.get("summary"),
            "confidence": value.get("confidence"),
        }
        for key, value in artifacts.items()
        if isinstance(value, dict)
    }

    return json.dumps(
        {
            "current_turn_id": snapshot.get("current_turn_id"),
            "workflow_status": snapshot.get("workflow_status"),
            "route_reason": snapshot.get("route_reason"),
            "current_group_index": snapshot.get("current_group_index"),
            "pending_approval_group": snapshot.get("pending_approval_group"),
            "review_decision": snapshot.get("review_decision"),
            "review_reason": snapshot.get("review_reason"),
            "artifact_digest": artifact_digest,
            "metrics_summary": snapshot.get("metrics_summary"),
            "workflow_plan": snapshot.get("workflow_plan"),
            "turn_ids": [turn.get("turn_id") for turn in _conversation_turns(snapshot)],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


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
    pre_trace_count = 0
    if session_id:
        pre_snapshot = await get_workflow_snapshot(active_session_id)
        pre_trace_count = len(pre_snapshot.get("workflow_trace") or [])

    task = asyncio.create_task(
        run_agent_graph(
            user_input=user_input,
            session_id=active_session_id,
            document_id=document_id,
        )
    )

    yield _format_sse_event("session", {"session_id": active_session_id})

    emitted_trace_count = pre_trace_count
    last_signature = ""
    try:
        while not task.done():
            snapshot = await get_workflow_snapshot(active_session_id)
            trace = snapshot.get("workflow_trace") or []
            while emitted_trace_count < len(trace):
                yield _format_trace_sse(active_session_id, trace[emitted_trace_count])
                emitted_trace_count += 1

            current_signature = _snapshot_signature(snapshot)
            if current_signature != last_signature:
                yield _format_sse_event(
                    "state",
                    _snapshot_to_session_state(active_session_id, snapshot),
                )
                last_signature = current_signature

            await asyncio.sleep(0.5)

        reply = await task
        snapshot = await get_workflow_snapshot(active_session_id)
        trace = snapshot.get("workflow_trace") or []
        while emitted_trace_count < len(trace):
            yield _format_trace_sse(active_session_id, trace[emitted_trace_count])
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
                "turn_id": snapshot.get("current_turn_id"),
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
    last_signature = _snapshot_signature(snapshot)

    try:
        while not task.done():
            snapshot = await get_workflow_snapshot(session_id)
            trace = snapshot.get("workflow_trace") or []
            while emitted_trace_count < len(trace):
                yield _format_trace_sse(session_id, trace[emitted_trace_count])
                emitted_trace_count += 1

            current_signature = _snapshot_signature(snapshot)
            if current_signature != last_signature:
                yield _format_sse_event(
                    "state",
                    _snapshot_to_session_state(session_id, snapshot),
                )
                last_signature = current_signature

            await asyncio.sleep(0.5)

        reply = await task
        snapshot = await get_workflow_snapshot(session_id)
        trace = snapshot.get("workflow_trace") or []
        while emitted_trace_count < len(trace):
            yield _format_trace_sse(session_id, trace[emitted_trace_count])
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
                "turn_id": snapshot.get("current_turn_id"),
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
