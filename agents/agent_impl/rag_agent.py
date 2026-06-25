"""RAG agent — answers questions from the uploaded document knowledge base."""

import time
from typing import Any

from agents.core.artifact_utils import build_artifact_record
from agents.tracing.event_factory import state_step_trace_event
from services.rag import rag_chat


async def run_rag_step(step: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    trace_events: list[dict[str, Any]] = []
    trace_events += state_step_trace_event(state, step, "step_started", agent="rag_agent")
    started = time.perf_counter()
    result = await rag_chat(question=step["goal"], document_id=state.get("document_id"))
    duration_ms = int((time.perf_counter() - started) * 1000)
    artifact = build_artifact_record(
        step=step,
        agent="rag_agent",
        summary=f"Answered document question for {step['output_key']}",
        artifact_type="answer",
        artifact_data=result or "",
        confidence="medium",
    )
    trace_events += state_step_trace_event(
        state,
        step,
        "step_completed",
        agent="rag_agent",
        duration_ms=duration_ms,
        summary=(result or "")[:300],
        confidence="medium",
    )
    return {
        "step_id": step["id"],
        "output_key": step["output_key"],
        "agent": "rag_agent",
        "goal": step["goal"],
        "result": result or "",
        "artifact": artifact,
        "trace_events": trace_events,
        "metrics": {
            "duration_ms": duration_ms,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "model_calls": 1,
            "tool_calls": 0,
        },
    }
