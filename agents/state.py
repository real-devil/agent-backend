from operator import add
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict[str, Any]], add]
    session_id: str | None
    document_id: str | None
    route: str
    route_reason: str
    final_reply: str
    tool_iterations: int
