"""Research agent — gathers context from documents or LLM analysis."""

import json
import time
from typing import Any

from agents.core.artifact_utils import artifact_context_text, build_artifact_record, parse_structured_step_output
from agents.core.llm import call_structured_step_model
from agents.core.message_utils import get_latest_user_input
from agents.core.prompts import RESEARCH_AGENT_PROMPT
from agents.core.trace_utils import state_step_trace_event


async def run_research_step(step: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    trace_events: list[dict[str, Any]] = []
    trace_events += state_step_trace_event(state, step, "step_started", agent="research_agent")
    if state.get("document_id"):
        from tools.search import search_documents

        started = time.perf_counter()
        result = await search_documents(query=step["goal"], document_id=state["document_id"])
        duration_ms = int((time.perf_counter() - started) * 1000)
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        parsed = {
            "summary": f"Retrieved document evidence for {step['output_key']}",
            "artifact_type": "facts",
            "artifact_data": result,
            "confidence": "medium",
        }
    else:
        structured_payload, meta = await call_structured_step_model(
            RESEARCH_AGENT_PROMPT,
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
        result = str(parsed["summary"])
        duration_ms = meta["duration_ms"]
        usage = meta["usage"]
    artifact = build_artifact_record(
        step=step,
        agent="research_agent",
        summary=str(parsed["summary"]),
        artifact_type=str(parsed["artifact_type"]),
        artifact_data=parsed["artifact_data"],
        confidence=str(parsed["confidence"]),
    )
    trace_events += state_step_trace_event(
        state,
        step,
        "step_completed",
        agent="research_agent",
        duration_ms=duration_ms,
        summary=str(parsed["summary"]),
        confidence=str(parsed["confidence"]),
        usage=usage,
    )
    return {
        "step_id": step["id"],
        "output_key": step["output_key"],
        "agent": "research_agent",
        "goal": step["goal"],
        "result": result,
        "artifact": artifact,
        "trace_events": trace_events,
        "metrics": {
            "duration_ms": duration_ms,
            "usage": usage,
            "model_calls": 0 if state.get("document_id") else 1,
            "tool_calls": 1 if state.get("document_id") else 0,
        },
    }
