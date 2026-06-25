"""Agent runtime — public API facade.

Orchestrates the LangGraph lifecycle, session management, and workflow execution.
Internal logic is delegated to core/, nodes/, agent_impl/, and graph.py.
"""

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from agents.core.constants import WORKFLOW_TIMEOUT_MS
from agents.core.metrics_utils import default_metrics_summary
from agents.tracing.event_factory import trace_event, trace_for_turn
from agents.graph import build_agent_graph
from agents.tracing.stream_buffer import reset_session_streams

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global runtime state
# ---------------------------------------------------------------------------
_agent_graph: Any | None = None
_checkpointer_cm: AsyncIterator[Any] | None = None
_checkpointer_kind = "memory"
_runtime_last_error: str | None = None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _graph_config(session_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": session_id}}


def _get_checkpointer_mode() -> str:
    return os.getenv("LANGGRAPH_CHECKPOINTER", "memory").strip().lower()


def _get_postgres_url() -> str | None:
    return (
        os.getenv("LANGGRAPH_POSTGRES_URL")
        or os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
    )


# ---------------------------------------------------------------------------
# Health / validation
# ---------------------------------------------------------------------------
def get_checkpointer_kind() -> str:
    return _checkpointer_kind


def validate_runtime_config() -> list[str]:
    errors: list[str] = []

    if not os.getenv("OPENAI_API_KEY"):
        errors.append("Missing OPENAI_API_KEY")

    mode = _get_checkpointer_mode()
    if mode not in {"memory", "postgres"}:
        errors.append("LANGGRAPH_CHECKPOINTER must be either 'memory' or 'postgres'")

    if mode == "postgres" and not _get_postgres_url():
        errors.append(
            "LANGGRAPH_CHECKPOINTER is 'postgres' but LANGGRAPH_POSTGRES_URL, DATABASE_URL, "
            "or POSTGRES_URL is not configured"
        )

    return errors


def get_runtime_health() -> dict[str, Any]:
    return {
        "initialized": _agent_graph is not None,
        "checkpointer": _checkpointer_kind,
        "configured_mode": _get_checkpointer_mode(),
        "has_postgres_url": bool(_get_postgres_url()),
        "config_errors": validate_runtime_config(),
        "last_error": _runtime_last_error,
    }


# ---------------------------------------------------------------------------
# Checkpointer
# ---------------------------------------------------------------------------
async def _build_checkpointer() -> tuple[Any, AsyncIterator[Any] | None, str]:
    mode = _get_checkpointer_mode()
    if mode == "memory":
        return InMemorySaver(), None, "memory"

    if mode != "postgres":
        raise ValueError(f"Unsupported LANGGRAPH_CHECKPOINTER mode: {mode}")

    postgres_url = _get_postgres_url()
    if not postgres_url:
        raise ValueError(
            "LANGGRAPH_CHECKPOINTER is set to postgres but no LANGGRAPH_POSTGRES_URL, "
            "DATABASE_URL, or POSTGRES_URL was provided."
        )

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        raise RuntimeError(
            "Postgres checkpointer dependencies are unavailable. "
            "Install langgraph-checkpoint-postgres and psycopg-binary."
        ) from exc

    checkpointer_cm = AsyncPostgresSaver.from_conn_string(postgres_url)
    checkpointer = await checkpointer_cm.__aenter__()
    await checkpointer.setup()
    return checkpointer, checkpointer_cm, "postgres"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
async def initialize_agent_runtime() -> None:
    global _agent_graph, _checkpointer_cm, _checkpointer_kind, _runtime_last_error
    if _agent_graph is not None:
        return

    config_errors = validate_runtime_config()
    if config_errors:
        _runtime_last_error = "; ".join(config_errors)
        raise RuntimeError(_runtime_last_error)

    checkpointer, checkpointer_cm, checkpointer_kind = await _build_checkpointer()
    _agent_graph = build_agent_graph(checkpointer)
    _checkpointer_cm = checkpointer_cm
    _checkpointer_kind = checkpointer_kind
    _runtime_last_error = None
    logger.info("Initialized agent runtime with %s checkpointer", checkpointer_kind)


async def shutdown_agent_runtime() -> None:
    global _agent_graph, _checkpointer_cm, _checkpointer_kind, _runtime_last_error
    if _checkpointer_cm is not None:
        await _checkpointer_cm.__aexit__(None, None, None)
    _agent_graph = None
    _checkpointer_cm = None
    _checkpointer_kind = "memory"
    _runtime_last_error = None


async def _get_agent_graph():
    if _agent_graph is None:
        await initialize_agent_runtime()
    return _agent_graph


# ---------------------------------------------------------------------------
# Turn management helpers
# ---------------------------------------------------------------------------
def _latest_assistant_message(state: dict[str, Any]) -> str:
    for message in reversed(state.get("messages", [])):
        if message.get("role") == "assistant":
            return str(message.get("content", "") or "")
    return ""


def _current_turn_record(state: dict[str, Any]) -> dict[str, Any] | None:
    turn_id = state.get("current_turn_id")
    if not turn_id:
        return None

    return {
        "turn_id": turn_id,
        "user_message": state.get("current_turn_user_message", ""),
        "started_at": state.get("current_turn_started_at"),
        "status": state.get("workflow_status"),
        "route_reason": state.get("route_reason"),
        "review_decision": state.get("review_decision"),
        "review_reason": state.get("review_reason"),
        "pending_approval_group": state.get("pending_approval_group"),
        "reply": state.get("final_reply") or _latest_assistant_message(state),
        "workflow_plan": state.get("workflow_plan") or [],
        "workflow_trace": trace_for_turn(state.get("workflow_trace") or [], str(turn_id)),
        "artifacts": state.get("artifacts") or {},
        "metrics_summary": state.get("metrics_summary") or default_metrics_summary(),
        "thinking_log": list(state.get("turn_thinking_log") or []),
    }


def _turn_is_terminal(state: dict[str, Any]) -> bool:
    return str(state.get("workflow_status", "")).strip() in {"completed", "rejected", "timed_out"}


async def _archive_current_turn(
    graph: Any,
    config: dict[str, dict[str, str]],
    state: dict[str, Any],
) -> dict[str, Any]:
    turn = _current_turn_record(state)
    if turn is None or not _turn_is_terminal(state):
        return state

    turn_history = list(state.get("turn_history") or [])
    turn_id = str(turn["turn_id"])
    if any(str(existing.get("turn_id")) == turn_id for existing in turn_history):
        return state

    turn_history.append(turn)
    await graph.aupdate_state(config, {"turn_history": turn_history})
    snapshot = await graph.aget_state(config)
    return snapshot.values or {}


def _is_waiting_for_approval(snapshot: dict[str, Any]) -> bool:
    return (
        snapshot.get("workflow_status") == "awaiting_approval"
        and bool(snapshot.get("workflow_plan"))
    )


# ---------------------------------------------------------------------------
# Graph invocation
# ---------------------------------------------------------------------------
async def _invoke_with_timeout(
    graph: Any,
    input_state: dict[str, Any],
    config: dict[str, dict[str, str]],
) -> dict[str, Any]:
    try:
        return await asyncio.wait_for(
            graph.ainvoke(input_state, config),
            timeout=WORKFLOW_TIMEOUT_MS / 1000,
        )
    except asyncio.TimeoutError:
        timeout_message = f"Workflow timed out after {WORKFLOW_TIMEOUT_MS // 1000} seconds."
        snapshot = await graph.aget_state(config)
        existing_state = snapshot.values or {}
        await graph.aupdate_state(
            config,
            {
                "workflow_status": "timed_out",
                "final_reply": timeout_message,
                "messages": [{"role": "assistant", "content": timeout_message}],
                "metrics_summary": existing_state.get("metrics_summary", default_metrics_summary()),
                "workflow_trace": trace_event(
                    "entry",
                    "workflow_timed_out",
                    turn_id=existing_state.get("current_turn_id"),
                    timeout_ms=WORKFLOW_TIMEOUT_MS,
                ),
            },
        )
        return {
            "final_reply": timeout_message,
            "messages": [{"role": "assistant", "content": timeout_message}],
        }


async def get_workflow_snapshot(session_id: str) -> dict[str, Any]:
    graph = await _get_agent_graph()
    snapshot = await graph.aget_state(_graph_config(session_id))
    return snapshot.values or {}


async def run_agent_graph(
    user_input: str,
    session_id: str,
    document_id: str | None = None,
) -> str:
    graph = await _get_agent_graph()
    config = _graph_config(session_id)
    snapshot = await graph.aget_state(config)
    existing_state = snapshot.values or {}

    if _is_waiting_for_approval(existing_state):
        input_state = {
            "messages": [{"role": "user", "content": user_input}],
            "session_id": session_id,
            "document_id": document_id,
            "workflow_status": "awaiting_approval",
            "final_reply": "",
        }
    else:
        existing_state = await _archive_current_turn(graph, config, existing_state)
        new_turn_id = str(uuid4())
        reset_session_streams(session_id)
        input_state = {
            "messages": [{"role": "user", "content": user_input}],
            "session_id": session_id,
            "document_id": document_id,
            "current_turn_id": new_turn_id,
            "current_turn_user_message": user_input,
            "current_turn_started_at": time.time(),
            "turn_history": list(existing_state.get("turn_history") or []),
            "artifacts": {},
            "workflow_trace": trace_event(
                "entry",
                "workflow_started",
                turn_id=new_turn_id,
                user_request=user_input,
            ),
            "workflow_status": "planning",
            "workflow_plan": [],
            "success_criteria": [],
            "current_group_index": 0,
            "current_group_results": [],
            "current_step_result": "",
            "current_step_agent": "",
            "current_step_goal": "",
            "review_decision": "",
            "review_reason": "",
            "review_failure_category": "",
            "review_rollback_target": "",
            "step_results": [],
            "step_retry_count": 0,
            "tool_iterations": 0,
            "approval_response": "",
            "pending_approval_group": "",
            "final_reply": "",
            "turn_thinking_log": [],
            "bypass_input_gates": False,
        }

    final_state = await _invoke_with_timeout(graph, input_state, config)
    return final_state.get("final_reply") or final_state["messages"][-1].get("content") or ""


async def resume_agent_graph(
    session_id: str,
    approval_response: str,
    user_input: str | None = None,
) -> str:
    graph = await _get_agent_graph()
    snapshot = await graph.aget_state(_graph_config(session_id))
    existing_state = snapshot.values or {}
    final_state = await _invoke_with_timeout(
        graph,
        {
            "messages": [{"role": "user", "content": user_input}] if user_input else [],
            "session_id": session_id,
            "current_turn_id": existing_state.get("current_turn_id"),
            "current_turn_user_message": existing_state.get("current_turn_user_message", ""),
            "current_turn_started_at": existing_state.get("current_turn_started_at"),
            "turn_history": list(existing_state.get("turn_history") or []),
            "workflow_trace": trace_event(
                "entry",
                "workflow_resumed",
                turn_id=existing_state.get("current_turn_id"),
                approval_response=approval_response,
            ),
            "workflow_status": "resume",
            "approval_response": approval_response,
            "bypass_input_gates": user_input is None,
            "final_reply": "",
        },
        _graph_config(session_id),
    )
    return final_state.get("final_reply") or final_state["messages"][-1].get("content") or ""
