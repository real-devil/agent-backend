"""Utility functions for navigating workflow plan groups and steps."""

from typing import Any

from agents.state import AgentState


def ordered_group_ids(state: AgentState) -> list[int]:
    groups: list[int] = []
    for step in state.get("workflow_plan", []):
        group = int(step.get("parallel_group", 0))
        if group not in groups:
            groups.append(group)
    return groups or [0]


def group_index_by_step_id(state: AgentState, step_id: str) -> int | None:
    ordered_groups = ordered_group_ids(state)
    group_id_to_index = {group_id: index for index, group_id in enumerate(ordered_groups)}
    for step in state.get("workflow_plan", []):
        if step["id"] == step_id:
            return group_id_to_index.get(int(step.get("parallel_group", 0)))
    return None


def truncate_step_results_before_group(state: AgentState, target_group_index: int) -> list[dict[str, Any]]:
    allowed_group_ids = set(ordered_group_ids(state)[:target_group_index])
    retained: list[dict[str, Any]] = []
    for result in state.get("step_results", []):
        group_index = group_index_by_step_id(state, str(result.get("step_id", "")))
        if group_index is not None and group_index in allowed_group_ids:
            retained.append(result)
    return retained


def get_current_group_id(state: AgentState) -> int:
    groups = ordered_group_ids(state)
    index = min(state.get("current_group_index", 0), len(groups) - 1)
    return groups[index]


def get_current_group_steps(state: AgentState) -> list[dict[str, Any]]:
    group_id = get_current_group_id(state)
    return [step for step in state.get("workflow_plan", []) if int(step.get("parallel_group", 0)) == group_id]


def has_remaining_groups(state: AgentState) -> bool:
    return state.get("current_group_index", 0) + 1 < len(ordered_group_ids(state))
