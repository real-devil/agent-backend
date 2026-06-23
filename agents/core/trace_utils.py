"""Trace event factory functions."""

from typing import Any

from agents.schemas import TraceEvent
from agents.trace_labels import resolve_trace_labels


def trace_event(
    node: str,
    event_type: str,
    *,
    turn_id: str | None = None,
    **detail: Any,
) -> list[dict[str, Any]]:
    display_label, activity_kind = resolve_trace_labels(node, event_type, detail)
    return [
        TraceEvent(
            event_type=event_type,
            node=node,
            turn_id=str(turn_id) if turn_id else None,
            display_label=display_label,
            activity_kind=activity_kind,
            detail=detail,
        ).model_dump()
    ]


def trace_for_turn(trace: list[dict[str, Any]], turn_id: str | None) -> list[dict[str, Any]]:
    if not turn_id:
        return trace
    return [event for event in trace if str(event.get("turn_id") or "") == str(turn_id)]


def state_trace_event(state: dict[str, Any], node: str, event_type: str, **detail: Any) -> list[dict[str, Any]]:
    return trace_event(node, event_type, turn_id=state.get("current_turn_id"), **detail)


def state_step_trace_event(
    state: dict[str, Any],
    step: dict[str, Any],
    event_type: str,
    **detail: Any,
) -> list[dict[str, Any]]:
    return state_trace_event(
        state,
        step.get("agent", "step"),
        event_type,
        step_id=step.get("id"),
        goal=step.get("goal"),
        output_key=step.get("output_key"),
        parallel_group=step.get("parallel_group"),
        **detail,
    )
