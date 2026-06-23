"""Planner node — calls LLM to decompose user request into a workflow plan."""

from agents.core.artifact_utils import artifact_context_text
from agents.core.message_utils import get_latest_user_input, recent_conversation_text
from agents.core.metrics_utils import merge_metrics
from agents.core.plan_parser import call_planner_model, extract_planner_thinking, normalize_plan, parse_planner_response
from agents.core.trace_utils import state_trace_event
from agents.state import AgentState
from agents.trace_labels import format_step_label


async def planner(state: AgentState) -> AgentState:
    user_prompt = (
        f"Latest user request:\n{get_latest_user_input(state)}\n\n"
        f"Conversation context:\n{recent_conversation_text(state)}\n\n"
        f"document_id present: {'yes' if state.get('document_id') else 'no'}"
    )
    raw_content, meta = await call_planner_model(state, user_prompt)
    plan = normalize_plan(parse_planner_response(raw_content), state)
    planning_thought = extract_planner_thinking(raw_content)
    turn_thinking_log = list(state.get("turn_thinking_log") or [])
    if planning_thought:
        turn_thinking_log.append({"phase": "planner", "content": planning_thought})
    return {
        "artifacts": {},
        "turn_thinking_log": turn_thinking_log,
        "metrics_summary": merge_metrics(
            state.get("metrics_summary"),
            duration_ms=meta["duration_ms"],
            usage=meta["usage"],
            model_calls=1,
        ),
        "workflow_trace": state_trace_event(
            state,
            "planner",
            "plan_created",
            step_count=len(plan["steps"]),
            workflow_status=plan["workflow_status"],
            route_reason=plan["route_reason"],
            user_request=get_latest_user_input(state),
            steps=[
                {
                    "id": step["id"],
                    "agent": step["agent"],
                    "goal": step["goal"],
                    "group": step["parallel_group"],
                    "display_label": format_step_label(step["agent"], step["goal"], step["id"]),
                }
                for step in plan["steps"]
            ],
            duration_ms=meta["duration_ms"],
            usage=meta["usage"],
        ),
        "workflow_status": plan["workflow_status"],
        "workflow_plan": plan["steps"],
        "route_reason": plan["route_reason"],
        "success_criteria": plan["success_criteria"],
        "current_group_index": 0,
        "current_group_results": [],
        "current_step_result": "",
        "current_step_agent": "",
        "current_step_goal": "",
        "review_decision": "",
        "review_reason": "",
        "review_failure_category": "",
        "review_rollback_target": "",
        "step_results": [],
        "step_retry_count": 0,
        "tool_iterations": 0,
        "approval_response": "",
        "pending_approval_group": "",
        "final_reply": "",
    }
