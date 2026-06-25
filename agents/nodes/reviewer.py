"""Reviewer node — LLM inspects group results and decides: continue / retry / finish."""

import json

from agents.core.artifact_utils import artifact_context_text, merge_artifacts
from agents.core.llm import call_text_model
from agents.core.message_utils import get_latest_user_input
from agents.core.metrics_utils import merge_metrics
from agents.core.plan_utils import group_index_by_step_id, has_remaining_groups
from agents.core.message_utils import parse_json_object
from agents.core.prompts import REVIEWER_PROMPT
from agents.tracing.event_factory import state_trace_event
from agents.schemas import ReviewPayload
from agents.state import AgentState

MAX_STEP_RETRIES = 1


async def reviewer(state: AgentState) -> AgentState:
    user_prompt = (
        f"Latest user request:\n{get_latest_user_input(state)}\n\n"
        f"Success criteria:\n{json.dumps(state.get('success_criteria', []), ensure_ascii=False)}\n\n"
        f"Current group index: {state.get('current_group_index', 0)}\n\n"
        f"Current group results:\n{json.dumps(state.get('current_group_results', []), ensure_ascii=False)}\n\n"
        f"Accepted step results so far:\n{json.dumps(state.get('step_results', []), ensure_ascii=False)}\n\n"
        f"Current artifacts:\n{artifact_context_text(state)}\n\n"
        f"Current retry count: {state.get('step_retry_count', 0)}\n"
        f"Has remaining groups after this one: {'yes' if has_remaining_groups(state) else 'no'}"
    )
    raw_content, meta = await call_text_model(REVIEWER_PROMPT, user_prompt)

    payload = parse_json_object(raw_content)
    try:
        review_payload = ReviewPayload(
            decision=str(payload.get("decision", "continue")).lower(),
            reason=str(payload.get("reason", "")).strip() or "Reviewer decision applied.",
            failure_category=str(payload.get("failure_category", "none")).strip().lower() or "none",
            rollback_to_step_id=str(payload.get("rollback_to_step_id", "")).strip(),
        )
    except Exception:
        review_payload = ReviewPayload(
            decision="continue",
            reason="Reviewer decision applied.",
            failure_category="none",
            rollback_to_step_id="",
        )

    decision = review_payload.decision
    reason = review_payload.reason
    failure_category = review_payload.failure_category
    rollback_to_step_id = review_payload.rollback_to_step_id
    if rollback_to_step_id and group_index_by_step_id(state, rollback_to_step_id) is None:
        rollback_to_step_id = ""

    retry_count = state.get("step_retry_count", 0)
    if decision == "retry" and retry_count >= MAX_STEP_RETRIES:
        rollback_group_index = group_index_by_step_id(state, rollback_to_step_id) if rollback_to_step_id else None
        if rollback_group_index is not None and rollback_group_index < state.get("current_group_index", 0):
            reason = "Retry limit reached; rolling workflow back to an earlier group."
        else:
            decision = "continue" if has_remaining_groups(state) else "finish"
            reason = "Retry limit reached; proceeding with the workflow."
    if decision == "continue" and not has_remaining_groups(state):
        decision = "finish"

    step_results = state.get("step_results", [])
    artifacts = state.get("artifacts", {})
    if decision in {"continue", "finish"}:
        step_results = step_results + state.get("current_group_results", [])
        artifacts = merge_artifacts(artifacts, state.get("current_group_results", []))

    return {
        "review_decision": decision,
        "review_reason": reason,
        "review_failure_category": failure_category,
        "review_rollback_target": rollback_to_step_id,
        "artifacts": artifacts,
        "metrics_summary": merge_metrics(
            state.get("metrics_summary"),
            duration_ms=meta["duration_ms"],
            usage=meta["usage"],
            model_calls=1,
            failure_category=failure_category,
        ),
        "step_results": step_results,
        "step_retry_count": retry_count + 1 if decision == "retry" else 0,
        "workflow_trace": state_trace_event(
            state,
            "reviewer",
            "review_completed",
            decision=decision,
            failure_category=failure_category,
            rollback_to_step_id=rollback_to_step_id,
            duration_ms=meta["duration_ms"],
            usage=meta["usage"],
        ),
    }
