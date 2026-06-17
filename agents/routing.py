from dataclasses import dataclass
from typing import Any

WEATHER_KEYWORDS = (
    "weather",
    "temperature",
    "forecast",
    "气温",
    "天气",
    "下雨",
    "降雨",
    "风力",
    "湿度",
    "forecast",
)

DOCUMENT_MARKERS = (
    "文档",
    "document",
    "文件中",
    "根据上传",
    "知识库",
    "pdf",
    "报告",
    "上传的",
)


@dataclass(frozen=True)
class RequestIntents:
    needs_weather_tool: bool
    needs_document_rag: bool


def detect_intents(user_message: str, *, has_document: bool) -> RequestIntents:
    text = user_message.strip()
    lower = text.lower()
    needs_weather = any(keyword in lower or keyword in text for keyword in WEATHER_KEYWORDS)
    explicit_document = any(marker in lower or marker in text for marker in DOCUMENT_MARKERS)

    if not has_document:
        return RequestIntents(needs_weather_tool=needs_weather, needs_document_rag=False)

    if needs_weather and not explicit_document:
        return RequestIntents(needs_weather_tool=True, needs_document_rag=False)

    if explicit_document:
        return RequestIntents(needs_weather_tool=needs_weather, needs_document_rag=True)

    return RequestIntents(needs_weather_tool=False, needs_document_rag=True)


def build_tool_step(
    *,
    step_id: str,
    goal: str,
    parallel_group: int = 0,
    output_key: str = "tool_result",
) -> dict[str, Any]:
    return {
        "id": step_id,
        "agent": "tool_agent",
        "goal": goal,
        "parallel_group": parallel_group,
        "approval_required": False,
        "depends_on": [],
        "output_key": output_key,
    }


def build_rag_step(
    *,
    step_id: str,
    goal: str,
    parallel_group: int = 0,
    output_key: str = "document_answer",
) -> dict[str, Any]:
    return {
        "id": step_id,
        "agent": "rag_agent",
        "goal": goal,
        "parallel_group": parallel_group,
        "approval_required": False,
        "depends_on": [],
        "output_key": output_key,
    }


def fallback_plan_for_intents(user_message: str, *, has_document: bool) -> dict[str, Any]:
    intents = detect_intents(user_message, has_document=has_document)
    steps: list[dict[str, Any]] = []

    if intents.needs_document_rag:
        steps.append(
            build_rag_step(
                step_id="step_doc",
                goal=user_message,
                parallel_group=0,
            )
        )

    if intents.needs_weather_tool:
        steps.append(
            build_tool_step(
                step_id="step_weather" if intents.needs_document_rag else "step_1",
                goal=user_message,
                parallel_group=0,
                output_key="weather_result" if intents.needs_document_rag else "tool_result",
            )
        )

    if not steps:
        steps.append(
            {
                "id": "step_1",
                "agent": "general_agent",
                "goal": user_message,
                "parallel_group": 0,
                "approval_required": False,
                "depends_on": [],
                "output_key": "general_answer",
            }
        )
        reason = "Fallback single-step general workflow."
    elif intents.needs_weather_tool and intents.needs_document_rag:
        reason = "Mixed request routed to rag_agent and tool_agent in parallel."
    elif intents.needs_weather_tool:
        reason = "Weather/live-data request routed to tool_agent."
    elif intents.needs_document_rag:
        reason = "Document-scoped request routed to rag_agent."
    else:
        reason = "Planner generated workflow."

    return {
        "workflow_status": "simple" if len(steps) == 1 else "multi_step",
        "route_reason": reason,
        "success_criteria": ["Produce a direct and accurate answer for the user."],
        "steps": steps,
    }


def reconcile_plan_steps(
    steps: list[dict[str, Any]],
    user_message: str,
    *,
    document_id: str | None,
    route_reason: str = "",
) -> tuple[list[dict[str, Any]], str]:
    if not steps:
        return steps, route_reason

    intents = detect_intents(user_message, has_document=bool(document_id))
    if not intents.needs_weather_tool:
        return steps, route_reason

    agents = {str(step.get("agent", "")) for step in steps}
    if "tool_agent" in agents:
        return steps, route_reason

    if intents.needs_document_rag:
        tool_step = build_tool_step(
            step_id=_next_step_id(steps, "step_weather"),
            goal=user_message,
            parallel_group=min(int(step.get("parallel_group", 0)) for step in steps),
            output_key="weather_result",
        )
        return [*steps, tool_step], (
            route_reason
            or "Added tool_agent for live weather data alongside document workflow."
        )

    tool_step = build_tool_step(step_id="step_1", goal=user_message)
    remaining = [step for step in steps if str(step.get("agent")) != "rag_agent"]
    if remaining:
        return [tool_step, *remaining], (
            "Weather/live-data request corrected to tool_agent; document RAG removed."
        )

    return [tool_step], "Weather/live-data request corrected to tool_agent (not answerable from document RAG)."


def _next_step_id(steps: list[dict[str, Any]], preferred: str) -> str:
    existing = {str(step.get("id", "")) for step in steps}
    if preferred not in existing:
        return preferred
    index = 2
    while f"{preferred}_{index}" in existing:
        index += 1
    return f"{preferred}_{index}"
