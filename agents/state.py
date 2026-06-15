from operator import add
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict[str, Any]], add]
    workflow_trace: Annotated[list[dict[str, Any]], add]
    session_id: str | None
    document_id: str | None
    route_reason: str
    final_reply: str
    artifacts: dict[str, Any]
    metrics_summary: dict[str, Any]
    workflow_status: str
    workflow_plan: list[dict[str, Any]]
    success_criteria: list[str]
    current_group_index: int
    current_group_results: list[dict[str, Any]]
    current_step_result: str
    current_step_agent: str
    current_step_goal: str
    review_decision: str
    review_reason: str
    review_failure_category: str
    review_rollback_target: str
    step_results: list[dict[str, Any]]
    step_retry_count: int
    tool_iterations: int
    approval_response: str
    pending_approval_group: str
