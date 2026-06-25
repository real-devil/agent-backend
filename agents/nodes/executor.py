"""Execute group node — runs all steps in the current group in parallel via asyncio.gather."""

import asyncio
from typing import Any

from agents.agent_impl.general import run_general_step
from agents.agent_impl.rag_agent import run_rag_step
from agents.agent_impl.research import run_research_step
from agents.agent_impl.tool_agent import run_tool_step
from agents.core.metrics_utils import merge_metrics
from agents.core.plan_utils import get_current_group_id, get_current_group_steps
from agents.tracing.event_factory import state_trace_event
from agents.state import AgentState
from agents.tracing.labels import format_step_label


async def dispatch_step(step: dict[str, Any], state: AgentState) -> dict[str, Any]:
    agent = step["agent"]
    if agent == "research_agent":
        return await run_research_step(step, state)
    if agent == "tool_agent":
        return await run_tool_step(step, state)
    if agent == "rag_agent":
        return await run_rag_step(step, state)
    return await run_general_step(step, state)


async def execute_group(state: AgentState) -> AgentState:
    steps = get_current_group_steps(state)
    results = await asyncio.gather(*[dispatch_step(step, state) for step in steps])
    group_trace_events = state_trace_event(
        state,
        "execute_group",
        "group_started",
        group_id=get_current_group_id(state),
        steps=[
            {
                "step_id": step["id"],
                "agent": step["agent"],
                "goal": step["goal"],
                "display_label": format_step_label(step["agent"], step["goal"], step["id"]),
            }
            for step in steps
        ],
    )
    for result in results:
        group_trace_events += result.get("trace_events", [])
    combined_text = "\n\n".join(
        f"[{result['step_id']} - {result['agent']}]\n{result['result']}" for result in results
    )
    total_duration_ms = sum(int(result.get("metrics", {}).get("duration_ms", 0)) for result in results)
    total_model_calls = sum(int(result.get("metrics", {}).get("model_calls", 0)) for result in results)
    total_tool_calls = sum(int(result.get("metrics", {}).get("tool_calls", 0)) for result in results)
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for result in results:
        usage = result.get("metrics", {}).get("usage", {})
        total_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0))
        total_usage["completion_tokens"] += int(usage.get("completion_tokens", 0))
        total_usage["total_tokens"] += int(usage.get("total_tokens", 0))
    return {
        "metrics_summary": merge_metrics(
            state.get("metrics_summary"),
            duration_ms=total_duration_ms,
            usage=total_usage,
            model_calls=total_model_calls,
            tool_calls=total_tool_calls,
        ),
        "workflow_status": "in_progress",
        "current_group_results": results,
        "current_step_result": combined_text,
        "current_step_agent": "parallel_group" if len(results) > 1 else results[0]["agent"],
        "current_step_goal": "; ".join(step["goal"] for step in steps),
        "tool_iterations": 0,
        "workflow_trace": group_trace_events
        + state_trace_event(
            state,
            "execute_group",
            "group_executed",
            group_id=get_current_group_id(state),
            step_ids=[result["step_id"] for result in results],
            duration_ms=total_duration_ms,
            model_calls=total_model_calls,
            tool_calls=total_tool_calls,
            usage=total_usage,
        ),
        "final_reply": "",
    }
