from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    session_id: str | None
    document_id: str | None
    route: str
    route_reason: str
    final_reply: str
    tool_iterations: int
