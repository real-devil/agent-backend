"""User-facing SSE output — trace/thinking/reply event formatting and filtering."""

import json
from typing import Any


def trace_for_turn(trace: list[dict[str, Any]], turn_id: str | None) -> list[dict[str, Any]]:
    if not turn_id:
        return trace
    return [event for event in trace if str(event.get("turn_id") or "") == str(turn_id)]


def format_sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def format_trace_sse(session_id: str, trace_event: dict[str, Any]) -> str:
    return format_sse_event(
        "trace",
        {
            "session_id": session_id,
            "turn_id": trace_event.get("turn_id"),
            "trace": trace_event,
        },
    )


def format_stream_sse(session_id: str, event: dict[str, Any]) -> str:
    event_type = str(event.pop("type", ""))
    return format_sse_event(event_type, {"session_id": session_id, **event})
