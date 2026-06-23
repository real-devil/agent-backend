"""Low-level helpers for extracting and parsing messages from AgentState."""

import json
import logging
from typing import Any

from agents.state import AgentState

logger = logging.getLogger(__name__)


def get_latest_user_input(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def recent_conversation_text(state: AgentState, limit: int = 8) -> str:
    recent = state["messages"][-limit:]
    lines = []
    for message in recent:
        lines.append(f"{message.get('role', 'unknown')}: {message.get('content', '')}")
    return "\n".join(lines)


def parse_json_object(raw_content: str) -> dict[str, Any]:
    if not raw_content:
        return {}
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        start = raw_content.find("{")
        end = raw_content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_content[start : end + 1])
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON object: %s", raw_content)
        else:
            logger.warning("Failed to parse JSON object: %s", raw_content)
    return {}


def serialize_assistant_message(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": message.content or "",
    }
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in message.tool_calls
        ]
    return payload
