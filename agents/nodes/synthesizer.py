"""Synthesizer node — composes the final user-facing answer from all step results."""

import json

from agents.core.artifact_utils import artifact_context_text
from agents.core.llm import stream_text_model
from agents.core.message_utils import get_latest_user_input
from agents.core.metrics_utils import merge_metrics
from agents.core.prompts import SYNTHESIZER_PROMPT
from agents.tracing.event_factory import state_trace_event
from agents.state import AgentState
from agents.tracing.stream_buffer import append_reply, finish_reply, start_reply


async def synthesizer(state: AgentState) -> AgentState:
    user_prompt = (
        f"Latest user request:\n{get_latest_user_input(state)}\n\n"
        f"Workflow reason:\n{state.get('route_reason', '')}\n\n"
        f"Workflow plan:\n{json.dumps(state.get('workflow_plan', []), ensure_ascii=False)}\n\n"
        f"Accepted step results:\n{json.dumps(state.get('step_results', []), ensure_ascii=False)}\n\n"
        f"Artifacts:\n{artifact_context_text(state)}"
    )
    session_id = str(state.get("session_id") or "")
    turn_id = str(state.get("current_turn_id") or "")
    start_reply(session_id, turn_id)

    def on_delta(delta: str) -> None:
        append_reply(session_id, delta)

    final_reply, meta = await stream_text_model(
        SYNTHESIZER_PROMPT,
        user_prompt,
        on_delta=on_delta,
    )
    if session_id:
        finish_reply(session_id)
    return {
        "final_reply": final_reply,
        "metrics_summary": merge_metrics(
            state.get("metrics_summary"),
            duration_ms=meta["duration_ms"],
            usage=meta["usage"],
            model_calls=1,
        ),
        "workflow_status": "completed",
        "workflow_trace": state_trace_event(
            state,
            "synthesizer",
            "final_answer_created",
            duration_ms=meta["duration_ms"],
            usage=meta["usage"],
        ),
        "messages": [{"role": "assistant", "content": final_reply}],
    }
