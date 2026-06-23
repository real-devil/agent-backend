"""Entry node — routes to planner or approval_gate based on workflow state."""

from agents.state import AgentState


async def entry(state: AgentState) -> AgentState:
    return {}


def route_from_entry(state: AgentState) -> str:
    if (
        state.get("workflow_status") in {"resume", "awaiting_approval", "rejected"}
        and state.get("workflow_plan")
    ) or (state.get("pending_approval_group") and state.get("workflow_plan")):
        return "approval_gate"
    return "planner"
