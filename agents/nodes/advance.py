"""Advance and rollback nodes — move between workflow groups."""

from agents.core.artifact_utils import rebuild_artifacts_from_step_results
from agents.core.metrics_utils import merge_metrics
from agents.core.plan_utils import group_index_by_step_id, truncate_step_results_before_group
from agents.tracing.event_factory import state_trace_event
from agents.state import AgentState


async def advance_group(state: AgentState) -> AgentState:
    return {
        "current_group_index": state.get("current_group_index", 0) + 1,
        "current_group_results": [],
        "current_step_result": "",
        "current_step_agent": "",
        "current_step_goal": "",
        "review_decision": "",
        "review_reason": "",
        "review_failure_category": "",
        "review_rollback_target": "",
        "step_retry_count": 0,
        "approval_response": "",
        "pending_approval_group": "",
        "workflow_trace": state_trace_event(
            state,
            "advance_group",
            "group_advanced",
            next_group_index=state.get("current_group_index", 0) + 1,
        ),
        "final_reply": "",
    }


async def rollback_group(state: AgentState) -> AgentState:
    rollback_target = state.get("review_rollback_target", "")
    rollback_group_index = group_index_by_step_id(state, rollback_target) or 0
    retained_step_results = truncate_step_results_before_group(state, rollback_group_index)
    return {
        "current_group_index": rollback_group_index,
        "current_group_results": [],
        "current_step_result": "",
        "current_step_agent": "",
        "current_step_goal": "",
        "review_decision": "",
        "review_reason": "",
        "review_failure_category": "",
        "review_rollback_target": "",
        "artifacts": rebuild_artifacts_from_step_results(retained_step_results),
        "metrics_summary": merge_metrics(
            state.get("metrics_summary"),
            rollback_count=1,
        ),
        "step_results": retained_step_results,
        "step_retry_count": 0,
        "approval_response": "",
        "pending_approval_group": "",
        "workflow_trace": state_trace_event(
            state,
            "rollback_group",
            "workflow_rolled_back",
            rollback_target=rollback_target,
            rollback_group_index=rollback_group_index,
        ),
        "final_reply": "",
    }
