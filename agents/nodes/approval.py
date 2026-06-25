"""Approval gate node — pauses workflow if any step in the group requires human approval."""

from agents.core.constants import APPROVAL_APPROVED, APPROVAL_REJECTED
from agents.core.metrics_utils import default_metrics_summary, merge_metrics
from agents.core.plan_utils import get_current_group_id, get_current_group_steps
from agents.tracing.event_factory import state_trace_event
from agents.state import AgentState


async def approval_gate(state: AgentState) -> AgentState:
    current_steps = get_current_group_steps(state)
    requires_approval = any(bool(step.get("approval_required")) for step in current_steps)
    if not requires_approval:
        return {
            "metrics_summary": state.get("metrics_summary", default_metrics_summary()),
            "workflow_trace": state_trace_event(state, "approval_gate", "approval_skipped", group_id=get_current_group_id(state)),
            "workflow_status": "in_progress",
            "pending_approval_group": "",
            "approval_response": "",
            "final_reply": "",
        }

    current_group = str(get_current_group_id(state))
    approval_response = state.get("approval_response", "").strip().lower()
    if approval_response in APPROVAL_REJECTED:
        message = f"Workflow stopped because approval was rejected for group {current_group}."
        return {
            "metrics_summary": merge_metrics(
                state.get("metrics_summary"),
                approval_rejected=1,
            ),
            "workflow_trace": state_trace_event(state, "approval_gate", "approval_rejected", group_id=current_group),
            "workflow_status": "rejected",
            "pending_approval_group": current_group,
            "final_reply": message,
            "messages": [{"role": "assistant", "content": message}],
        }

    if approval_response not in APPROVAL_APPROVED:
        goals = "; ".join(step["goal"] for step in current_steps)
        message = (
            f"Approval required before executing workflow group {current_group}. "
            f"Pending goals: {goals}"
        )
        return {
            "metrics_summary": merge_metrics(
                state.get("metrics_summary"),
                approval_requested=1,
            ),
            "workflow_trace": state_trace_event(state, "approval_gate", "approval_requested", group_id=current_group, goals=goals),
            "workflow_status": "awaiting_approval",
            "pending_approval_group": current_group,
            "final_reply": message,
            "messages": [{"role": "assistant", "content": message}],
        }

    return {
        "metrics_summary": merge_metrics(
            state.get("metrics_summary"),
            approval_granted=1,
        ),
        "workflow_trace": state_trace_event(state, "approval_gate", "approval_granted", group_id=current_group),
        "workflow_status": "in_progress",
        "pending_approval_group": "",
        "approval_response": "",
        "review_failure_category": "",
        "review_rollback_target": "",
        "final_reply": "",
    }
