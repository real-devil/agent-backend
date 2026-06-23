"""General agent — reasoning, writing, transformation without tools."""

import json
from typing import Any

from agents.core.artifact_utils import artifact_context_text, build_artifact_record, parse_structured_step_output
from agents.core.llm import call_structured_step_model
from agents.core.message_utils import get_latest_user_input
from agents.core.prompts import GENERAL_AGENT_PROMPT
from agents.core.trace_utils import state_step_trace_event


async def run_general_step(step: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    trace_events: list[dict[str, Any]] = []
    trace_events += state_step_trace_event(state, step, "step_started", agent="general_agent")
    structured_payload, meta = await call_structured_step_model(
        GENERAL_AGENT_PROMPT,
        (
            f"Current step goal:\n{step['goal']}\n\n"
            f"Latest user request:\n{get_latest_user_input(state)}\n\n"
            f"Accepted step results so far:\n{json.dumps(state.get('step_results', []), ensure_ascii=False)}\n\n"
            f"Existing artifacts:\n{artifact_context_text(state)}"
        ),
    )
    parsed = parse_structured_step_output(
        json.dumps(structured_payload, ensure_ascii=False),
        step["output_key"],
    )
    artifact = build_artifact_record(
        step=step,
        agent="general_agent",
        summary=str(parsed["summary"]),
        artifact_type=str(parsed["artifact_type"]),
        artifact_data=parsed["artifact_data"],
        confidence=str(parsed["confidence"]),
    )
    trace_events += state_step_trace_event(
        state,
        step,
        "step_completed",
        agent="general_agent",
        duration_ms=meta["duration_ms"],
        summary=str(parsed["summary"]),
        confidence=str(parsed["confidence"]),
        usage=meta["usage"],
    )
    return {
        "step_id": step["id"],
        "output_key": step["output_key"],
        "agent": "general_agent",
        "goal": step["goal"],
        "result": str(parsed["summary"]),
        "artifact": artifact,
        "trace_events": trace_events,
        "metrics": {
            "duration_ms": meta["duration_ms"],
            "usage": meta["usage"],
            "model_calls": 1,
            "tool_calls": 0,
        },
    }
