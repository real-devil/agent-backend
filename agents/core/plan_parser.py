"""Planner response parsing, plan normalization, and thinking stream extraction."""

import json
from typing import Any

from agents.core.constants import PLAN_JSON_MARKER
from agents.core.llm import stream_text_model
from agents.core.message_utils import parse_json_object
from agents.core.plan_utils import ordered_group_ids
from agents.core.prompts import PLANNER_PROMPT
from agents.routing import fallback_plan_for_intents, reconcile_plan_steps
from agents.schemas import StepSpec
from agents.state import AgentState
from agents.stream_buffer import append_thinking, finish_thinking, start_thinking
from agents.trace_labels import format_step_label


def extract_planner_thinking(raw_content: str) -> str:
    if "<thinking>" in raw_content and "</thinking>" in raw_content:
        return raw_content.split("<thinking>", 1)[1].split("</thinking>", 1)[0].strip()
    if PLAN_JSON_MARKER in raw_content:
        return raw_content.split(PLAN_JSON_MARKER, 1)[0].replace("<thinking>", "").strip()
    return ""


def parse_planner_response(raw_content: str) -> dict[str, Any]:
    if PLAN_JSON_MARKER in raw_content:
        json_part = raw_content.split(PLAN_JSON_MARKER, 1)[1]
        json_part = json_part.replace("</plan_json>", "").strip()
        if json_part.startswith("```"):
            json_part = json_part.strip("`")
            if json_part.startswith("json"):
                json_part = json_part[4:].strip()
        return parse_json_object(json_part)
    return parse_json_object(raw_content)


def emit_planner_thinking_delta(session_id: str, buffer: str, emitted_len: int) -> int:
    if not session_id:
        return emitted_len

    thinking_source = buffer
    if PLAN_JSON_MARKER in thinking_source:
        thinking_source = thinking_source.split(PLAN_JSON_MARKER, 1)[0]

    partial_markers = (
        PLAN_JSON_MARKER,
        "<plan_json",
        "<plan_js",
        "<plan_j",
        "<plan_",
        "<plan",
        "<pla",
        "<pl",
        "<p",
        "<",
    )
    for marker in partial_markers:
        if thinking_source.endswith(marker):
            thinking_source = thinking_source[: -len(marker)]
            break

    thinking_text = (
        thinking_source.replace("<thinking>", "").replace("</thinking>", "").strip()
    )
    if len(thinking_text) <= emitted_len:
        return emitted_len

    append_thinking(session_id, thinking_text[emitted_len:])
    return len(thinking_text)


async def call_planner_model(state: AgentState, user_prompt: str) -> tuple[str, dict[str, Any]]:
    session_id = str(state.get("session_id") or "")
    turn_id = str(state.get("current_turn_id") or "")
    start_thinking(session_id, turn_id, "planner")
    buffer = ""
    emitted_len = 0

    def on_delta(delta: str) -> None:
        nonlocal buffer, emitted_len
        buffer += delta
        emitted_len = emit_planner_thinking_delta(session_id, buffer, emitted_len)

    raw_content, meta = await stream_text_model(PLANNER_PROMPT, user_prompt, on_delta=on_delta)
    if session_id:
        emitted_len = emit_planner_thinking_delta(session_id, raw_content, emitted_len)
        finish_thinking(session_id)
    return raw_content, meta


def fallback_plan(state: AgentState) -> dict[str, Any]:
    from agents.core.message_utils import get_latest_user_input

    return fallback_plan_for_intents(
        get_latest_user_input(state),
        has_document=bool(state.get("document_id")),
    )


def normalize_plan(payload: dict[str, Any], state: AgentState) -> dict[str, Any]:
    from agents.core.message_utils import get_latest_user_input

    if not payload.get("steps"):
        payload = fallback_plan(state)

    steps: list[dict[str, Any]] = []
    known_ids: list[str] = []
    for index, raw_step in enumerate(payload.get("steps", []), start=1):
        step_id = str(raw_step.get("id", f"step_{index}")).strip() or f"step_{index}"
        if step_id in known_ids:
            step_id = f"{step_id}_{index}"
        known_ids.append(step_id)

        agent = str(raw_step.get("agent", "general_agent")).strip()
        if agent not in {"research_agent", "tool_agent", "rag_agent", "general_agent"}:
            agent = "general_agent"
        goal = str(raw_step.get("goal", "")).strip() or get_latest_user_input(state)
        parallel_group = raw_step.get("parallel_group", index - 1)
        try:
            parallel_group = int(parallel_group)
        except (TypeError, ValueError):
            parallel_group = index - 1
        raw_depends_on = raw_step.get("depends_on", [])
        depends_on = []
        if isinstance(raw_depends_on, list):
            depends_on = [str(dep) for dep in raw_depends_on if str(dep) in known_ids[:-1]]
        output_key = str(raw_step.get("output_key", f"{step_id}_output")).strip() or f"{step_id}_output"
        steps.append(
            StepSpec(
                id=step_id,
                agent=agent,
                goal=goal,
                parallel_group=parallel_group,
                approval_required=bool(raw_step.get("approval_required", False)),
                depends_on=depends_on,
                output_key=output_key,
            ).model_dump()
        )

    if not steps:
        fallback = fallback_plan(state)
        steps = fallback["steps"]
        payload = fallback

    group_by_step_id: dict[str, int] = {}
    for step in steps:
        if step["depends_on"]:
            inherited_group = max(group_by_step_id[dep] for dep in step["depends_on"]) + 1
            step["parallel_group"] = max(step["parallel_group"], inherited_group)
        group_by_step_id[step["id"]] = step["parallel_group"]

    ordered_groups = sorted({int(step["parallel_group"]) for step in steps})
    compact_group_map = {group: index for index, group in enumerate(ordered_groups)}
    for step in steps:
        step["parallel_group"] = compact_group_map[int(step["parallel_group"])]

    route_reason = str(payload.get("route_reason", "Planner generated workflow."))
    steps, route_reason = reconcile_plan_steps(
        steps,
        get_latest_user_input(state),
        document_id=state.get("document_id"),
        route_reason=route_reason,
    )

    return {
        "workflow_status": payload.get("workflow_status", "multi_step" if len(steps) > 1 else "simple"),
        "route_reason": route_reason,
        "success_criteria": [
            str(item) for item in payload.get("success_criteria", []) if str(item).strip()
        ] or ["Answer the user's request accurately."],
        "steps": steps,
    }
