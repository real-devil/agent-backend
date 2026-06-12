from operator import add
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict[str, Any]], add]
    session_id: str | None
    document_id: str | None
    route: str
    route_reason: str
    final_reply: str
    workflow_status: str
    workflow_plan: list[dict[str, Any]]
    success_criteria: list[str]
    current_step_index: int
    current_step_result: str
    current_step_agent: str
    current_step_goal: str
    review_decision: str
    review_reason: str
    step_results: list[dict[str, Any]]
    step_retry_count: int
    tool_iterations: int
