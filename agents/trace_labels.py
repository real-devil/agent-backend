from typing import Any, Literal

ActivityKind = Literal[
    "session",
    "plan",
    "parallel",
    "step",
    "tool",
    "approval",
    "review",
    "synthesize",
    "system",
]

AGENT_LABELS: dict[str, str] = {
    "research_agent": "Research",
    "tool_agent": "Tools",
    "rag_agent": "Document Q&A",
    "general_agent": "Analysis",
}

TOOL_LABELS: dict[str, str] = {
    "search_documents": "Search documents",
    "get_weather": "Get weather",
}


def _truncate(text: str, max_len: int = 72) -> str:
    cleaned = str(text or "").strip()
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[: max_len - 1]}…"


def _agent_label(agent: str | None) -> str:
    if not agent:
        return "Agent"
    return AGENT_LABELS.get(agent, agent.replace("_", " "))


def _tool_label(tool_name: str | None) -> str:
    if not tool_name:
        return "Run tool"
    return TOOL_LABELS.get(tool_name, tool_name.replace("_", " "))


def format_step_label(agent: str | None, goal: str | None, step_id: str | None = None) -> str:
    goal_text = _truncate(goal or step_id or "Step")
    return f"{_agent_label(agent)} · {goal_text}"


def resolve_trace_labels(
    node: str,
    event_type: str,
    detail: dict[str, Any],
) -> tuple[str, ActivityKind]:
    explicit_label = detail.get("display_label")
    explicit_kind = detail.get("activity_kind")
    if isinstance(explicit_label, str) and explicit_label.strip():
        kind = explicit_kind if isinstance(explicit_kind, str) else _default_kind(event_type)
        return explicit_label.strip(), kind  # type: ignore[return-value]

    match event_type:
        case "workflow_started":
            user_request = detail.get("user_request")
            if user_request:
                return f"Received: {_truncate(str(user_request), 64)}", "session"
            return "Workflow started", "session"
        case "workflow_resumed":
            return "Resumed after approval", "session"
        case "plan_created":
            step_count = int(detail.get("step_count") or len(detail.get("steps") or []))
            suffix = "" if step_count == 1 else "s"
            return f"Planned {step_count} step{suffix}", "plan"
        case "group_started":
            steps = detail.get("steps") or []
            count = len(steps) if isinstance(steps, list) else 0
            if count > 1:
                return f"Running {count} steps in parallel", "parallel"
            if count == 1 and isinstance(steps[0], dict):
                step = steps[0]
                return format_step_label(
                    str(step.get("agent") or ""),
                    str(step.get("goal") or ""),
                    str(step.get("step_id") or step.get("id") or ""),
                ), "step"
            return "Executing workflow group", "parallel"
        case "group_executed":
            group_id = detail.get("group_id", "?")
            return f"Parallel group {group_id} completed", "parallel"
        case "step_started":
            return (
                format_step_label(
                    str(detail.get("agent") or node),
                    str(detail.get("goal") or ""),
                    str(detail.get("step_id") or ""),
                ),
                "step",
            )
        case "step_completed":
            return (
                format_step_label(
                    str(detail.get("agent") or node),
                    str(detail.get("summary") or detail.get("goal") or ""),
                    str(detail.get("step_id") or ""),
                ),
                "step",
            )
        case "tool_called":
            return _tool_label(str(detail.get("tool_name") or "")), "tool"
        case "tool_result":
            return f"{_tool_label(str(detail.get('tool_name') or ''))} completed", "tool"
        case "approval_requested":
            return f"Waiting for approval · group {detail.get('group_id', '?')}", "approval"
        case "approval_granted":
            return f"Approval granted · group {detail.get('group_id', '?')}", "approval"
        case "approval_rejected":
            return f"Approval rejected · group {detail.get('group_id', '?')}", "approval"
        case "approval_skipped":
            return "Approval not required", "approval"
        case "gate_passed":
            return f"Safety check passed · {node}", "system"
        case "gate_rejected":
            reason = detail.get("reason", "")
            suffix = f" ({reason})" if reason else ""
            return f"Input blocked{suffix}", "system"
        case "rate_limited":
            return "Rate limited · too many requests", "system"
        case "circuit_open":
            return "Circuit breaker open · service degraded", "system"
        case "audit_recorded":
            return "Audit log recorded", "system"
        case "review_completed":
            return f"Review · {detail.get('decision', 'continue')}", "review"
        case "workflow_rolled_back":
            return "Rolled back to an earlier step", "system"
        case "group_advanced":
            return f"Advanced to group {detail.get('next_group_index', '?')}", "system"
        case "final_answer_created":
            return "Composing final answer", "synthesize"
        case "workflow_timed_out":
            return "Workflow timed out", "system"
        case _:
            return f"{node} · {event_type.replace('_', ' ')}", _default_kind(event_type)


def _default_kind(event_type: str) -> ActivityKind:
    if event_type in {"workflow_started", "workflow_resumed"}:
        return "session"
    if event_type in {"plan_created"}:
        return "plan"
    if event_type.startswith("approval_"):
        return "approval"
    if event_type in {"gate_passed", "gate_rejected", "rate_limited", "circuit_open", "audit_recorded"}:
        return "system"
    if event_type in {"group_started", "group_executed"}:
        return "parallel"
    if event_type.startswith("step_"):
        return "step"
    if event_type.startswith("tool_"):
        return "tool"
    if event_type == "review_completed":
        return "review"
    if event_type == "final_answer_created":
        return "synthesize"
    return "system"
