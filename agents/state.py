from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    session_id: str | None
    document_id: str | None
    tool_iterations: int
