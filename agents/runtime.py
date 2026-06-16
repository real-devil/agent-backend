import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from agents.schemas import ArtifactRecord, MetricsSummary, ReviewPayload, StepSpec, StructuredStepOutput, TraceEvent
from agents.state import AgentState
from services.rag import rag_chat
from tools.search import SEARCH_TOOL, search_documents
from tools.weather import WEATHER_TOOL, get_weather

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-4o-mini"
TOOLS = [WEATHER_TOOL, SEARCH_TOOL]
MAX_TOOL_ITERATIONS = 5
MAX_STEP_RETRIES = 1
WORKFLOW_TIMEOUT_MS = 300_000
APPROVAL_APPROVED = {"approved", "approve", "yes", "continue"}
APPROVAL_REJECTED = {"rejected", "reject", "no", "deny", "denied", "stop"}

PLANNER_PROMPT = """
You are the planner and supervisor of a production multi-agent system.
Break the user's request into a small executable workflow.

Available agents:
- research_agent: gathers context, constraints, or reference material
- tool_agent: uses tools such as weather lookup or document search
- rag_agent: answers directly from the uploaded document knowledge base
- general_agent: reasoning, writing, transformation, or synthesis without tools

Rules:
- Return JSON only.
- Use 1-6 steps.
- Steps with the same parallel_group can be executed in parallel.
- Each step may declare depends_on as a list of prior step IDs.
- Set approval_required=true for risky, ambiguous, high-cost, or externally consequential steps.
- If a document_id is present, prefer rag_agent or tool_agent/search_documents when useful.
- The final user-facing answer will be written later by a synthesizer node.

Schema:
{
  "workflow_status": "simple" | "multi_step",
  "route_reason": "short explanation",
  "success_criteria": ["criterion 1", "criterion 2"],
  "steps": [
    {
      "id": "step_1",
      "agent": "research_agent" | "tool_agent" | "rag_agent" | "general_agent",
      "goal": "what this step should achieve",
      "parallel_group": 0,
      "approval_required": false,
      "depends_on": ["step_0"],
      "output_key": "short_machine_readable_name"
    }
  ]
}
""".strip()

RESEARCH_AGENT_PROMPT = """
You are a research agent inside a multi-agent workflow.
Produce concise research notes for the current step.
Focus on facts, constraints, open questions, and useful context for downstream agents.
Do not answer as the final assistant unless the step explicitly asks for that.
""".strip()

GENERAL_AGENT_PROMPT = """
You are a general-purpose execution agent inside a multi-agent workflow.
Complete the current step without using tools.
Be concise but useful, and optimize for downstream agents consuming your output.
""".strip()

TOOL_AGENT_PROMPT = """
You are a tool execution agent inside a multi-agent workflow.
Use tools when needed and return a concise result for the current step.
Do not produce the final user-facing answer unless the step explicitly requires it.
""".strip()

STRUCTURED_STEP_OUTPUT_PROMPT = """
Return JSON only with this schema:
{
  "summary": "short concise summary",
  "artifact_type": "notes" | "facts" | "answer" | "analysis" | "tool_result",
  "artifact_data": "string or object containing the useful output",
  "confidence": "high" | "medium" | "low"
}
""".strip()

REVIEWER_PROMPT = """
You are the reviewer of a multi-agent workflow.
Inspect the current group results and decide whether to continue, retry, or finish.

Return JSON only with this schema:
{
  "decision": "continue" | "retry" | "finish",
  "reason": "short explanation",
  "failure_category": "none" | "missing_info" | "tool_failure" | "low_confidence" | "invalid_plan",
  "rollback_to_step_id": "optional previous step id"
}

Guidance:
- retry: the current group result is unusable or clearly insufficient
- continue: the current group is acceptable and the workflow should move to the next group
- finish: the workflow already has enough information to produce the final answer
- Use rollback_to_step_id only when the workflow should restart from an earlier accepted step group
""".strip()

SYNTHESIZER_PROMPT = """
You are the final answer synthesizer of a production multi-agent system.
Use the workflow plan and accepted step results to answer the user's latest request.
Be direct, accurate, and grounded in the collected step results.
If the workflow evidence is insufficient, say so clearly.
""".strip()

_agent_graph: Any | None = None
_checkpointer_cm: AsyncIterator[Any] | None = None
_checkpointer_kind = "memory"
_runtime_last_error: str | None = None


def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def _get_model_name() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def _graph_config(session_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": session_id}}


def _get_checkpointer_mode() -> str:
    return os.getenv("LANGGRAPH_CHECKPOINTER", "memory").strip().lower()


def _get_postgres_url() -> str | None:
    return (
        os.getenv("LANGGRAPH_POSTGRES_URL")
        or os.getenv("DATABASE_URL")
        or os.getenv("POSTGRES_URL")
    )


def get_checkpointer_kind() -> str:
    return _checkpointer_kind


def validate_runtime_config() -> list[str]:
    errors: list[str] = []

    if not os.getenv("OPENAI_API_KEY"):
        errors.append("Missing OPENAI_API_KEY")

    mode = _get_checkpointer_mode()
    if mode not in {"memory", "postgres"}:
        errors.append("LANGGRAPH_CHECKPOINTER must be either 'memory' or 'postgres'")

    if mode == "postgres" and not _get_postgres_url():
        errors.append(
            "LANGGRAPH_CHECKPOINTER is 'postgres' but LANGGRAPH_POSTGRES_URL, DATABASE_URL, "
            "or POSTGRES_URL is not configured"
        )

    return errors


def get_runtime_health() -> dict[str, Any]:
    return {
        "initialized": _agent_graph is not None,
        "checkpointer": _checkpointer_kind,
        "configured_mode": _get_checkpointer_mode(),
        "has_postgres_url": bool(_get_postgres_url()),
        "config_errors": validate_runtime_config(),
        "last_error": _runtime_last_error,
    }


async def _build_checkpointer() -> tuple[Any, AsyncIterator[Any] | None, str]:
    mode = _get_checkpointer_mode()
    if mode == "memory":
        return InMemorySaver(), None, "memory"

    if mode != "postgres":
        raise ValueError(f"Unsupported LANGGRAPH_CHECKPOINTER mode: {mode}")

    postgres_url = _get_postgres_url()
    if not postgres_url:
        raise ValueError(
            "LANGGRAPH_CHECKPOINTER is set to postgres but no LANGGRAPH_POSTGRES_URL, "
            "DATABASE_URL, or POSTGRES_URL was provided."
        )

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        raise RuntimeError(
            "Postgres checkpointer dependencies are unavailable. "
            "Install langgraph-checkpoint-postgres and psycopg-binary."
        ) from exc

    checkpointer_cm = AsyncPostgresSaver.from_conn_string(postgres_url)
    checkpointer = await checkpointer_cm.__aenter__()
    await checkpointer.setup()
    return checkpointer, checkpointer_cm, "postgres"


def _get_latest_user_input(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _recent_conversation_text(state: AgentState, limit: int = 8) -> str:
    recent = state["messages"][-limit:]
    lines = []
    for message in recent:
        lines.append(f"{message.get('role', 'unknown')}: {message.get('content', '')}")
    return "\n".join(lines)


def _parse_json_object(raw_content: str) -> dict[str, Any]:
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


def _serialize_assistant_message(message: Any) -> dict[str, Any]:
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


def _artifact_context_text(state: AgentState) -> str:
    artifacts = state.get("artifacts", {})
    if not artifacts:
        return "{}"
    return json.dumps(artifacts, ensure_ascii=False)


def _trace_event(node: str, event_type: str, **detail: Any) -> list[dict[str, Any]]:
    return [TraceEvent(event_type=event_type, node=node, detail=detail).model_dump()]


def _step_trace_event(step: dict[str, Any], event_type: str, **detail: Any) -> list[dict[str, Any]]:
    return _trace_event(
        step.get("agent", "step"),
        event_type,
        step_id=step.get("id"),
        goal=step.get("goal"),
        output_key=step.get("output_key"),
        parallel_group=step.get("parallel_group"),
        **detail,
    )


def _default_metrics_summary() -> dict[str, Any]:
    return MetricsSummary().model_dump()


def _usage_to_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _merge_metrics(
    current: dict[str, Any] | None,
    *,
    duration_ms: int = 0,
    usage: dict[str, int] | None = None,
    model_calls: int = 0,
    tool_calls: int = 0,
    approval_requested: int = 0,
    approval_granted: int = 0,
    approval_rejected: int = 0,
    rollback_count: int = 0,
    failure_category: str | None = None,
) -> dict[str, Any]:
    merged = dict(current or _default_metrics_summary())
    merged["total_duration_ms"] = int(merged.get("total_duration_ms", 0)) + int(duration_ms)
    merged["total_model_calls"] = int(merged.get("total_model_calls", 0)) + int(model_calls)
    merged["total_tool_calls"] = int(merged.get("total_tool_calls", 0)) + int(tool_calls)
    merged["approval_requests"] = int(merged.get("approval_requests", 0)) + int(approval_requested)
    merged["approval_grants"] = int(merged.get("approval_grants", 0)) + int(approval_granted)
    merged["approval_rejections"] = int(merged.get("approval_rejections", 0)) + int(approval_rejected)
    merged["rollback_count"] = int(merged.get("rollback_count", 0)) + int(rollback_count)

    normalized_usage = usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    merged["prompt_tokens"] = int(merged.get("prompt_tokens", 0)) + int(normalized_usage.get("prompt_tokens", 0))
    merged["completion_tokens"] = int(merged.get("completion_tokens", 0)) + int(normalized_usage.get("completion_tokens", 0))
    merged["total_tokens"] = int(merged.get("total_tokens", 0)) + int(normalized_usage.get("total_tokens", 0))

    failure_counts = dict(merged.get("failure_counts", {}))
    if failure_category and failure_category != "none":
        failure_counts[failure_category] = int(failure_counts.get(failure_category, 0)) + 1
    merged["failure_counts"] = failure_counts
    return merged


def _parse_structured_step_output(raw_content: str, output_key: str) -> dict[str, Any]:
    payload = _parse_json_object(raw_content)
    if not payload:
        return StructuredStepOutput(
            summary=raw_content.strip() or output_key,
            artifact_type="analysis",
            artifact_data=raw_content.strip(),
            confidence="medium",
        ).model_dump()

    summary = str(payload.get("summary", "")).strip() or output_key
    artifact_type = str(payload.get("artifact_type", "analysis")).strip().lower() or "analysis"
    confidence = str(payload.get("confidence", "medium")).strip().lower() or "medium"

    try:
        return StructuredStepOutput(
            summary=summary,
            artifact_type=artifact_type,
            artifact_data=payload.get("artifact_data", summary),
            confidence=confidence,
        ).model_dump()
    except Exception:
        return StructuredStepOutput(
            summary=summary,
            artifact_type="analysis",
            artifact_data=payload.get("artifact_data", summary),
            confidence="medium",
        ).model_dump()


def _build_artifact_record(
    *,
    step: dict[str, Any],
    agent: str,
    summary: str,
    artifact_type: str,
    artifact_data: Any,
    confidence: str,
) -> dict[str, Any]:
    return ArtifactRecord(
        step_id=step["id"],
        output_key=step["output_key"],
        agent=agent,
        artifact_type=artifact_type,
        summary=summary,
        confidence=confidence,
        data=artifact_data,
    ).model_dump()


def _rebuild_artifacts_from_step_results(step_results: list[dict[str, Any]]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for result in step_results:
        output_key = str(result.get("output_key", "")).strip()
        artifact = result.get("artifact")
        if output_key and artifact is not None:
            artifacts[output_key] = artifact
    return artifacts


def _fallback_plan(state: AgentState) -> dict[str, Any]:
    latest_input = _get_latest_user_input(state).lower()
    if state.get("document_id"):
        steps = [
            {
                "id": "step_1",
                "agent": "rag_agent",
                "goal": _get_latest_user_input(state),
                "parallel_group": 0,
                "approval_required": False,
                "depends_on": [],
                "output_key": "document_answer",
            }
        ]
        reason = "Document-specific request routed to rag_agent."
    elif any(keyword in latest_input for keyword in ("weather", "temperature", "forecast", "天气")):
        steps = [
            {
                "id": "step_1",
                "agent": "tool_agent",
                "goal": _get_latest_user_input(state),
                "parallel_group": 0,
                "approval_required": False,
                "depends_on": [],
                "output_key": "tool_result",
            }
        ]
        reason = "Weather-style request routed to tool_agent."
    else:
        steps = [
            {
                "id": "step_1",
                "agent": "general_agent",
                "goal": _get_latest_user_input(state),
                "parallel_group": 0,
                "approval_required": False,
                "depends_on": [],
                "output_key": "general_answer",
            }
        ]
        reason = "Fallback single-step general workflow."

    return {
        "workflow_status": "simple",
        "route_reason": reason,
        "success_criteria": ["Produce a direct and accurate answer for the user."],
        "steps": steps,
    }


def _normalize_plan(payload: dict[str, Any], state: AgentState) -> dict[str, Any]:
    if not payload.get("steps"):
        payload = _fallback_plan(state)

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
        goal = str(raw_step.get("goal", "")).strip() or _get_latest_user_input(state)
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
        fallback = _fallback_plan(state)
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

    return {
        "workflow_status": payload.get("workflow_status", "multi_step" if len(steps) > 1 else "simple"),
        "route_reason": str(payload.get("route_reason", "Planner generated workflow.")),
        "success_criteria": [
            str(item) for item in payload.get("success_criteria", []) if str(item).strip()
        ] or ["Answer the user's request accurately."],
        "steps": steps,
    }


def _ordered_group_ids(state: AgentState) -> list[int]:
    groups: list[int] = []
    for step in state.get("workflow_plan", []):
        group = int(step.get("parallel_group", 0))
        if group not in groups:
            groups.append(group)
    return groups or [0]


def _group_index_by_step_id(state: AgentState, step_id: str) -> int | None:
    ordered_groups = _ordered_group_ids(state)
    group_id_to_index = {group_id: index for index, group_id in enumerate(ordered_groups)}
    for step in state.get("workflow_plan", []):
        if step["id"] == step_id:
            return group_id_to_index.get(int(step.get("parallel_group", 0)))
    return None


def _truncate_step_results_before_group(state: AgentState, target_group_index: int) -> list[dict[str, Any]]:
    allowed_group_ids = set(_ordered_group_ids(state)[:target_group_index])
    retained: list[dict[str, Any]] = []
    for result in state.get("step_results", []):
        group_index = _group_index_by_step_id(state, str(result.get("step_id", "")))
        if group_index is not None and group_index in allowed_group_ids:
            retained.append(result)
    return retained


def _merge_artifacts(current_artifacts: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(current_artifacts)
    for result in results:
        output_key = str(result.get("output_key", "")).strip()
        artifact = result.get("artifact")
        if output_key and artifact is not None:
            merged[output_key] = artifact
    return merged


def _get_current_group_id(state: AgentState) -> int:
    groups = _ordered_group_ids(state)
    index = min(state.get("current_group_index", 0), len(groups) - 1)
    return groups[index]


def _get_current_group_steps(state: AgentState) -> list[dict[str, Any]]:
    group_id = _get_current_group_id(state)
    return [step for step in state.get("workflow_plan", []) if int(step.get("parallel_group", 0)) == group_id]


def _has_remaining_groups(state: AgentState) -> bool:
    return state.get("current_group_index", 0) + 1 < len(_ordered_group_ids(state))


async def _call_text_model(system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    response = await _get_client().chat.completions.create(
        model=_get_model_name(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    duration_ms = int((time.perf_counter() - started) * 1000)
    usage = _usage_to_dict(getattr(response, "usage", None))
    return response.choices[0].message.content or "", {
        "duration_ms": duration_ms,
        "usage": usage,
    }


async def _call_structured_step_model(system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_content, meta = await _call_text_model(
        f"{system_prompt}\n\n{STRUCTURED_STEP_OUTPUT_PROMPT}",
        user_prompt,
    )
    return _parse_json_object(raw_content), meta


async def _entry(state: AgentState) -> AgentState:
    return {}


async def _planner(state: AgentState) -> AgentState:
    user_prompt = (
        f"Latest user request:\n{_get_latest_user_input(state)}\n\n"
        f"Conversation context:\n{_recent_conversation_text(state)}\n\n"
        f"document_id present: {'yes' if state.get('document_id') else 'no'}"
    )
    raw_content, meta = await _call_text_model(PLANNER_PROMPT, user_prompt)
    plan = _normalize_plan(_parse_json_object(raw_content), state)
    return {
        "artifacts": {},
        "metrics_summary": _merge_metrics(
            state.get("metrics_summary"),
            duration_ms=meta["duration_ms"],
            usage=meta["usage"],
            model_calls=1,
        ),
        "workflow_trace": _trace_event(
            "planner",
            "plan_created",
            step_count=len(plan["steps"]),
            workflow_status=plan["workflow_status"],
            route_reason=plan["route_reason"],
            user_request=_get_latest_user_input(state),
            steps=[
                {
                    "id": step["id"],
                    "agent": step["agent"],
                    "goal": step["goal"],
                    "group": step["parallel_group"],
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


async def _approval_gate(state: AgentState) -> AgentState:
    current_steps = _get_current_group_steps(state)
    requires_approval = any(bool(step.get("approval_required")) for step in current_steps)
    if not requires_approval:
        return {
            "metrics_summary": state.get("metrics_summary", _default_metrics_summary()),
            "workflow_trace": _trace_event("approval_gate", "approval_skipped", group_id=_get_current_group_id(state)),
            "workflow_status": "in_progress",
            "pending_approval_group": "",
            "approval_response": "",
            "final_reply": "",
        }

    current_group = str(_get_current_group_id(state))
    approval_response = state.get("approval_response", "").strip().lower()
    if approval_response in APPROVAL_REJECTED:
        message = f"Workflow stopped because approval was rejected for group {current_group}."
        return {
            "metrics_summary": _merge_metrics(
                state.get("metrics_summary"),
                approval_rejected=1,
            ),
            "workflow_trace": _trace_event("approval_gate", "approval_rejected", group_id=current_group),
            "workflow_status": "rejected",
            "pending_approval_group": current_group,
            "final_reply": message,
            "messages": [{"role": "assistant", "content": message}],
        }

    if approval_response not in APPROVAL_APPROVED:
        goals = "; ".join(step["goal"] for step in current_steps)
        message = (
            f"Approval required before executing workflow group {current_group}. "
            f"Pending goals: {goals}"
        )
        return {
            "metrics_summary": _merge_metrics(
                state.get("metrics_summary"),
                approval_requested=1,
            ),
            "workflow_trace": _trace_event("approval_gate", "approval_requested", group_id=current_group, goals=goals),
            "workflow_status": "awaiting_approval",
            "pending_approval_group": current_group,
            "final_reply": message,
            "messages": [{"role": "assistant", "content": message}],
        }

    return {
        "metrics_summary": _merge_metrics(
            state.get("metrics_summary"),
            approval_granted=1,
        ),
        "workflow_trace": _trace_event("approval_gate", "approval_granted", group_id=current_group),
        "workflow_status": "in_progress",
        "pending_approval_group": "",
        "approval_response": "",
        "review_failure_category": "",
        "review_rollback_target": "",
        "final_reply": "",
    }


async def _run_research_step(step: dict[str, Any], state: AgentState) -> dict[str, Any]:
    trace_events: list[dict[str, Any]] = []
    trace_events += _step_trace_event(step, "step_started", agent="research_agent")
    if state.get("document_id"):
        started = time.perf_counter()
        result = await search_documents(query=step["goal"], document_id=state["document_id"])
        duration_ms = int((time.perf_counter() - started) * 1000)
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        parsed = {
            "summary": f"Retrieved document evidence for {step['output_key']}",
            "artifact_type": "facts",
            "artifact_data": result,
            "confidence": "medium",
        }
    else:
        structured_payload, meta = await _call_structured_step_model(
            RESEARCH_AGENT_PROMPT,
            (
                f"Current step goal:\n{step['goal']}\n\n"
                f"Latest user request:\n{_get_latest_user_input(state)}\n\n"
                f"Accepted step results so far:\n{json.dumps(state.get('step_results', []), ensure_ascii=False)}\n\n"
                f"Existing artifacts:\n{_artifact_context_text(state)}"
            ),
        )
        parsed = _parse_structured_step_output(
            json.dumps(structured_payload, ensure_ascii=False),
            step["output_key"],
        )
        result = str(parsed["summary"])
        duration_ms = meta["duration_ms"]
        usage = meta["usage"]
    artifact = _build_artifact_record(
        step=step,
        agent="research_agent",
        summary=str(parsed["summary"]),
        artifact_type=str(parsed["artifact_type"]),
        artifact_data=parsed["artifact_data"],
        confidence=str(parsed["confidence"]),
    )
    trace_events += _step_trace_event(
        step,
        "step_completed",
        agent="research_agent",
        duration_ms=duration_ms,
        summary=str(parsed["summary"]),
        confidence=str(parsed["confidence"]),
        usage=usage,
    )
    return {
        "step_id": step["id"],
        "output_key": step["output_key"],
        "agent": "research_agent",
        "goal": step["goal"],
        "result": result,
        "artifact": artifact,
        "trace_events": trace_events,
        "metrics": {
            "duration_ms": duration_ms,
            "usage": usage,
            "model_calls": 0 if state.get("document_id") else 1,
            "tool_calls": 1 if state.get("document_id") else 0,
        },
    }


async def _tool_execute(name: str, args: dict[str, Any], state: AgentState) -> str:
    if name == "get_weather":
        return await get_weather(**args)
    if name == "search_documents":
        if state.get("document_id") and "document_id" not in args:
            args["document_id"] = state["document_id"]
        return await search_documents(**args)
    return f"Unknown tool: {name}"


async def _run_tool_step(step: dict[str, Any], state: AgentState) -> dict[str, Any]:
    trace_events: list[dict[str, Any]] = []
    trace_events += _step_trace_event(step, "step_started", agent="tool_agent")
    local_messages: list[dict[str, Any]] = [
        {"role": "system", "content": f"{TOOL_AGENT_PROMPT}\n\n{STRUCTURED_STEP_OUTPUT_PROMPT}"},
        {
            "role": "user",
            "content": (
                f"Current step goal:\n{step['goal']}\n\n"
                f"Latest user request:\n{_get_latest_user_input(state)}\n\n"
                f"Accepted step results so far:\n{json.dumps(state.get('step_results', []), ensure_ascii=False)}\n\n"
                f"Existing artifacts:\n{_artifact_context_text(state)}"
            ),
        },
    ]

    final_result = "Tool agent ended without producing a final result."
    final_parsed = {
        "summary": final_result,
        "artifact_type": "tool_result",
        "artifact_data": final_result,
        "confidence": "medium",
    }
    total_duration_ms = 0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    tool_call_count = 0
    model_call_count = 0
    for _ in range(MAX_TOOL_ITERATIONS):
        started = time.perf_counter()
        response = await _get_client().chat.completions.create(
            model=_get_model_name(),
            messages=local_messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        model_call_count += 1
        total_duration_ms += int((time.perf_counter() - started) * 1000)
        usage = _usage_to_dict(getattr(response, "usage", None))
        total_usage["prompt_tokens"] += usage["prompt_tokens"]
        total_usage["completion_tokens"] += usage["completion_tokens"]
        total_usage["total_tokens"] += usage["total_tokens"]
        assistant_message = _serialize_assistant_message(response.choices[0].message)
        if not assistant_message.get("tool_calls"):
            final_result = assistant_message.get("content", "") or final_result
            final_parsed = _parse_structured_step_output(final_result, step["output_key"])
            break
        local_messages.append(assistant_message)
        for tool_call in assistant_message["tool_calls"]:
            tool_call_count += 1
            fn_name = tool_call["function"]["name"]
            raw_args = tool_call["function"]["arguments"]
            trace_events += _step_trace_event(
                step,
                "tool_called",
                agent="tool_agent",
                tool_name=fn_name,
                tool_args=raw_args,
            )
            try:
                fn_args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError as exc:
                tool_result = f"Tool argument parsing failed for {fn_name}: {exc}"
            else:
                tool_result = await _tool_execute(fn_name, fn_args, state)
                logger.info("tool_agent %s(%s) => %s", fn_name, fn_args, tool_result)
            trace_events += _step_trace_event(
                step,
                "tool_result",
                agent="tool_agent",
                tool_name=fn_name,
                result_preview=str(tool_result)[:400],
            )
            local_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result,
                }
            )
    artifact = _build_artifact_record(
        step=step,
        agent="tool_agent",
        summary=str(final_parsed["summary"]),
        artifact_type=str(final_parsed["artifact_type"]),
        artifact_data=final_parsed["artifact_data"],
        confidence=str(final_parsed["confidence"]),
    )
    trace_events += _step_trace_event(
        step,
        "step_completed",
        agent="tool_agent",
        duration_ms=total_duration_ms,
        summary=str(final_parsed["summary"]),
        confidence=str(final_parsed["confidence"]),
        usage=total_usage,
        tool_calls=tool_call_count,
    )
    return {
        "step_id": step["id"],
        "output_key": step["output_key"],
        "agent": "tool_agent",
        "goal": step["goal"],
        "result": str(final_parsed["summary"]),
        "artifact": artifact,
        "trace_events": trace_events,
        "metrics": {
            "duration_ms": total_duration_ms,
            "usage": total_usage,
            "model_calls": model_call_count,
            "tool_calls": tool_call_count,
        },
    }


async def _run_rag_step(step: dict[str, Any], state: AgentState) -> dict[str, Any]:
    trace_events: list[dict[str, Any]] = []
    trace_events += _step_trace_event(step, "step_started", agent="rag_agent")
    started = time.perf_counter()
    result = await rag_chat(question=step["goal"], document_id=state.get("document_id"))
    duration_ms = int((time.perf_counter() - started) * 1000)
    artifact = _build_artifact_record(
        step=step,
        agent="rag_agent",
        summary=f"Answered document question for {step['output_key']}",
        artifact_type="answer",
        artifact_data=result or "",
        confidence="medium",
    )
    trace_events += _step_trace_event(
        step,
        "step_completed",
        agent="rag_agent",
        duration_ms=duration_ms,
        summary=(result or "")[:300],
        confidence="medium",
    )
    return {
        "step_id": step["id"],
        "output_key": step["output_key"],
        "agent": "rag_agent",
        "goal": step["goal"],
        "result": result or "",
        "artifact": artifact,
        "trace_events": trace_events,
        "metrics": {
            "duration_ms": duration_ms,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "model_calls": 1,
            "tool_calls": 0,
        },
    }


async def _run_general_step(step: dict[str, Any], state: AgentState) -> dict[str, Any]:
    trace_events: list[dict[str, Any]] = []
    trace_events += _step_trace_event(step, "step_started", agent="general_agent")
    structured_payload, meta = await _call_structured_step_model(
        GENERAL_AGENT_PROMPT,
        (
            f"Current step goal:\n{step['goal']}\n\n"
            f"Latest user request:\n{_get_latest_user_input(state)}\n\n"
            f"Accepted step results so far:\n{json.dumps(state.get('step_results', []), ensure_ascii=False)}\n\n"
            f"Existing artifacts:\n{_artifact_context_text(state)}"
        ),
    )
    parsed = _parse_structured_step_output(
        json.dumps(structured_payload, ensure_ascii=False),
        step["output_key"],
    )
    artifact = _build_artifact_record(
        step=step,
        agent="general_agent",
        summary=str(parsed["summary"]),
        artifact_type=str(parsed["artifact_type"]),
        artifact_data=parsed["artifact_data"],
        confidence=str(parsed["confidence"]),
    )
    trace_events += _step_trace_event(
        step,
        "step_completed",
        agent="general_agent",
        duration_ms=meta["duration_ms"],
        summary=str(parsed["summary"]),
        confidence=str(parsed["confidence"]),
        usage=meta["usage"],
    )
    return {
        "step_id": step["id"],
        "output_key": step["output_key"],
        "agent": "general_agent",
        "goal": step["goal"],
        "result": str(parsed["summary"]),
        "artifact": artifact,
        "trace_events": trace_events,
        "metrics": {
            "duration_ms": meta["duration_ms"],
            "usage": meta["usage"],
            "model_calls": 1,
            "tool_calls": 0,
        },
    }


async def _dispatch_step(step: dict[str, Any], state: AgentState) -> dict[str, Any]:
    agent = step["agent"]
    if agent == "research_agent":
        return await _run_research_step(step, state)
    if agent == "tool_agent":
        return await _run_tool_step(step, state)
    if agent == "rag_agent":
        return await _run_rag_step(step, state)
    return await _run_general_step(step, state)


async def _execute_group(state: AgentState) -> AgentState:
    steps = _get_current_group_steps(state)
    results = await asyncio.gather(*[_dispatch_step(step, state) for step in steps])
    group_trace_events = _trace_event(
        "execute_group",
        "group_started",
        group_id=_get_current_group_id(state),
        steps=[
            {
                "step_id": step["id"],
                "agent": step["agent"],
                "goal": step["goal"],
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
        "metrics_summary": _merge_metrics(
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
        + _trace_event(
            "execute_group",
            "group_executed",
            group_id=_get_current_group_id(state),
            step_ids=[result["step_id"] for result in results],
            duration_ms=total_duration_ms,
            model_calls=total_model_calls,
            tool_calls=total_tool_calls,
            usage=total_usage,
        ),
        "final_reply": "",
    }


async def _reviewer(state: AgentState) -> AgentState:
    user_prompt = (
        f"Latest user request:\n{_get_latest_user_input(state)}\n\n"
        f"Success criteria:\n{json.dumps(state.get('success_criteria', []), ensure_ascii=False)}\n\n"
        f"Current group index: {state.get('current_group_index', 0)}\n\n"
        f"Current group results:\n{json.dumps(state.get('current_group_results', []), ensure_ascii=False)}\n\n"
        f"Accepted step results so far:\n{json.dumps(state.get('step_results', []), ensure_ascii=False)}\n\n"
        f"Current artifacts:\n{_artifact_context_text(state)}\n\n"
        f"Current retry count: {state.get('step_retry_count', 0)}\n"
        f"Has remaining groups after this one: {'yes' if _has_remaining_groups(state) else 'no'}"
    )
    raw_content, meta = await _call_text_model(REVIEWER_PROMPT, user_prompt)
    payload = _parse_json_object(raw_content)
    try:
        review_payload = ReviewPayload(
            decision=str(payload.get("decision", "continue")).lower(),
            reason=str(payload.get("reason", "")).strip() or "Reviewer decision applied.",
            failure_category=str(payload.get("failure_category", "none")).strip().lower() or "none",
            rollback_to_step_id=str(payload.get("rollback_to_step_id", "")).strip(),
        )
    except Exception:
        review_payload = ReviewPayload(
            decision="continue",
            reason="Reviewer decision applied.",
            failure_category="none",
            rollback_to_step_id="",
        )

    decision = review_payload.decision
    reason = review_payload.reason
    failure_category = review_payload.failure_category
    rollback_to_step_id = review_payload.rollback_to_step_id
    if rollback_to_step_id and _group_index_by_step_id(state, rollback_to_step_id) is None:
        rollback_to_step_id = ""

    retry_count = state.get("step_retry_count", 0)
    if decision == "retry" and retry_count >= MAX_STEP_RETRIES:
        rollback_group_index = _group_index_by_step_id(state, rollback_to_step_id) if rollback_to_step_id else None
        if rollback_group_index is not None and rollback_group_index < state.get("current_group_index", 0):
            reason = "Retry limit reached; rolling workflow back to an earlier group."
        else:
            decision = "continue" if _has_remaining_groups(state) else "finish"
            reason = "Retry limit reached; proceeding with the workflow."
    if decision == "continue" and not _has_remaining_groups(state):
        decision = "finish"

    step_results = state.get("step_results", [])
    artifacts = state.get("artifacts", {})
    if decision in {"continue", "finish"}:
        step_results = step_results + state.get("current_group_results", [])
        artifacts = _merge_artifacts(artifacts, state.get("current_group_results", []))

    return {
        "review_decision": decision,
        "review_reason": reason,
        "review_failure_category": failure_category,
        "review_rollback_target": rollback_to_step_id,
        "artifacts": artifacts,
        "metrics_summary": _merge_metrics(
            state.get("metrics_summary"),
            duration_ms=meta["duration_ms"],
            usage=meta["usage"],
            model_calls=1,
            failure_category=failure_category,
        ),
        "step_results": step_results,
        "step_retry_count": retry_count + 1 if decision == "retry" else 0,
        "workflow_trace": _trace_event(
            "reviewer",
            "review_completed",
            decision=decision,
            failure_category=failure_category,
            rollback_to_step_id=rollback_to_step_id,
            duration_ms=meta["duration_ms"],
            usage=meta["usage"],
        ),
    }


async def _advance_group(state: AgentState) -> AgentState:
    return {
        "current_group_index": state.get("current_group_index", 0) + 1,
        "current_group_results": [],
        "current_step_result": "",
        "current_step_agent": "",
        "current_step_goal": "",
        "review_decision": "",
        "review_reason": "",
        "review_failure_category": "",
        "review_rollback_target": "",
        "step_retry_count": 0,
        "approval_response": "",
        "pending_approval_group": "",
        "workflow_trace": _trace_event(
            "advance_group",
            "group_advanced",
            next_group_index=state.get("current_group_index", 0) + 1,
        ),
        "final_reply": "",
    }


async def _rollback_group(state: AgentState) -> AgentState:
    rollback_target = state.get("review_rollback_target", "")
    rollback_group_index = _group_index_by_step_id(state, rollback_target) or 0
    retained_step_results = _truncate_step_results_before_group(state, rollback_group_index)
    return {
        "current_group_index": rollback_group_index,
        "current_group_results": [],
        "current_step_result": "",
        "current_step_agent": "",
        "current_step_goal": "",
        "review_decision": "",
        "review_reason": "",
        "review_failure_category": "",
        "review_rollback_target": "",
        "artifacts": _rebuild_artifacts_from_step_results(retained_step_results),
        "metrics_summary": _merge_metrics(
            state.get("metrics_summary"),
            rollback_count=1,
        ),
        "step_results": retained_step_results,
        "step_retry_count": 0,
        "approval_response": "",
        "pending_approval_group": "",
        "workflow_trace": _trace_event(
            "rollback_group",
            "workflow_rolled_back",
            rollback_target=rollback_target,
            rollback_group_index=rollback_group_index,
        ),
        "final_reply": "",
    }


async def _synthesizer(state: AgentState) -> AgentState:
    user_prompt = (
        f"Latest user request:\n{_get_latest_user_input(state)}\n\n"
        f"Workflow reason:\n{state.get('route_reason', '')}\n\n"
        f"Workflow plan:\n{json.dumps(state.get('workflow_plan', []), ensure_ascii=False)}\n\n"
        f"Accepted step results:\n{json.dumps(state.get('step_results', []), ensure_ascii=False)}\n\n"
        f"Artifacts:\n{_artifact_context_text(state)}"
    )
    final_reply, meta = await _call_text_model(SYNTHESIZER_PROMPT, user_prompt)
    return {
        "final_reply": final_reply,
        "metrics_summary": _merge_metrics(
            state.get("metrics_summary"),
            duration_ms=meta["duration_ms"],
            usage=meta["usage"],
            model_calls=1,
        ),
        "workflow_status": "completed",
        "workflow_trace": _trace_event(
            "synthesizer",
            "final_answer_created",
            duration_ms=meta["duration_ms"],
            usage=meta["usage"],
        ),
        "messages": [{"role": "assistant", "content": final_reply}],
    }


def _route_from_entry(state: AgentState) -> str:
    if (
        state.get("workflow_status") in {"resume", "awaiting_approval", "rejected"}
        and state.get("workflow_plan")
    ) or (state.get("pending_approval_group") and state.get("workflow_plan")):
        return "approval_gate"
    return "planner"


def _route_after_approval_gate(state: AgentState) -> str:
    status = state.get("workflow_status", "")
    if status in {"awaiting_approval", "rejected"}:
        return END
    return "execute_group"


def _route_after_review(state: AgentState) -> str:
    decision = state.get("review_decision", "continue")
    if decision == "retry":
        rollback_group_index = _group_index_by_step_id(state, state.get("review_rollback_target", ""))
        if rollback_group_index is not None and rollback_group_index < state.get("current_group_index", 0):
            return "rollback_group"
        return "approval_gate"
    if decision == "continue":
        return "advance_group"
    return "synthesizer"


def _route_after_advance(state: AgentState) -> str:
    return "approval_gate"


def build_agent_graph(checkpointer: Any):
    graph = StateGraph(AgentState)
    graph.add_node("entry", _entry)
    graph.add_node("planner", _planner)
    graph.add_node("approval_gate", _approval_gate)
    graph.add_node("execute_group", _execute_group)
    graph.add_node("reviewer", _reviewer)
    graph.add_node("advance_group", _advance_group)
    graph.add_node("rollback_group", _rollback_group)
    graph.add_node("synthesizer", _synthesizer)

    graph.add_edge(START, "entry")
    graph.add_conditional_edges(
        "entry",
        _route_from_entry,
        {
            "planner": "planner",
            "approval_gate": "approval_gate",
        },
    )
    graph.add_edge("planner", "approval_gate")
    graph.add_conditional_edges(
        "approval_gate",
        _route_after_approval_gate,
        {
            "execute_group": "execute_group",
            END: END,
        },
    )
    graph.add_edge("execute_group", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        _route_after_review,
        {
            "approval_gate": "approval_gate",
            "advance_group": "advance_group",
            "rollback_group": "rollback_group",
            "synthesizer": "synthesizer",
        },
    )
    graph.add_conditional_edges(
        "advance_group",
        _route_after_advance,
        {
            "approval_gate": "approval_gate",
        },
    )
    graph.add_conditional_edges(
        "rollback_group",
        _route_after_advance,
        {
            "approval_gate": "approval_gate",
        },
    )
    graph.add_edge("synthesizer", END)

    return graph.compile(checkpointer=checkpointer)


async def initialize_agent_runtime() -> None:
    global _agent_graph, _checkpointer_cm, _checkpointer_kind, _runtime_last_error
    if _agent_graph is not None:
        return

    config_errors = validate_runtime_config()
    if config_errors:
        _runtime_last_error = "; ".join(config_errors)
        raise RuntimeError(_runtime_last_error)

    checkpointer, checkpointer_cm, checkpointer_kind = await _build_checkpointer()
    _agent_graph = build_agent_graph(checkpointer)
    _checkpointer_cm = checkpointer_cm
    _checkpointer_kind = checkpointer_kind
    _runtime_last_error = None
    logger.info("Initialized agent runtime with %s checkpointer", checkpointer_kind)


async def shutdown_agent_runtime() -> None:
    global _agent_graph, _checkpointer_cm, _checkpointer_kind, _runtime_last_error
    if _checkpointer_cm is not None:
        await _checkpointer_cm.__aexit__(None, None, None)
    _agent_graph = None
    _checkpointer_cm = None
    _checkpointer_kind = "memory"
    _runtime_last_error = None


async def _get_agent_graph():
    if _agent_graph is None:
        await initialize_agent_runtime()
    return _agent_graph


def _is_waiting_for_approval(snapshot: dict[str, Any]) -> bool:
    return (
        snapshot.get("workflow_status") == "awaiting_approval"
        and bool(snapshot.get("workflow_plan"))
    )


async def _invoke_with_timeout(
    graph: Any,
    input_state: dict[str, Any],
    config: dict[str, dict[str, str]],
) -> dict[str, Any]:
    try:
        return await asyncio.wait_for(
            graph.ainvoke(input_state, config),
            timeout=WORKFLOW_TIMEOUT_MS / 1000,
        )
    except asyncio.TimeoutError:
        timeout_message = f"Workflow timed out after {WORKFLOW_TIMEOUT_MS // 1000} seconds."
        snapshot = await graph.aget_state(config)
        existing_state = snapshot.values or {}
        await graph.aupdate_state(
            config,
            {
                "workflow_status": "timed_out",
                "final_reply": timeout_message,
                "messages": [{"role": "assistant", "content": timeout_message}],
                "metrics_summary": existing_state.get("metrics_summary", _default_metrics_summary()),
                "workflow_trace": _trace_event(
                    "entry",
                    "workflow_timed_out",
                    timeout_ms=WORKFLOW_TIMEOUT_MS,
                ),
            },
        )
        return {
            "final_reply": timeout_message,
            "messages": [{"role": "assistant", "content": timeout_message}],
        }


async def get_workflow_snapshot(session_id: str) -> dict[str, Any]:
    graph = await _get_agent_graph()
    snapshot = await graph.aget_state(_graph_config(session_id))
    return snapshot.values or {}


async def run_agent_graph(
    user_input: str,
    session_id: str,
    document_id: str | None = None,
) -> str:
    graph = await _get_agent_graph()
    config = _graph_config(session_id)
    snapshot = await graph.aget_state(config)
    existing_state = snapshot.values or {}

    if _is_waiting_for_approval(existing_state):
        input_state = {
            "messages": [{"role": "user", "content": user_input}],
            "session_id": session_id,
            "document_id": document_id,
            "workflow_status": "awaiting_approval",
            "final_reply": "",
        }
    else:
        input_state = {
            "messages": [{"role": "user", "content": user_input}],
            "session_id": session_id,
            "document_id": document_id,
            "artifacts": {},
            "workflow_trace": _trace_event("entry", "workflow_started"),
            "workflow_status": "planning",
            "workflow_plan": [],
            "success_criteria": [],
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

    final_state = await _invoke_with_timeout(graph, input_state, config)
    return final_state.get("final_reply") or final_state["messages"][-1].get("content") or ""


async def resume_agent_graph(
    session_id: str,
    approval_response: str,
    user_input: str | None = None,
) -> str:
    graph = await _get_agent_graph()
    final_state = await _invoke_with_timeout(
        graph,
        {
            "messages": [{"role": "user", "content": user_input}] if user_input else [],
            "session_id": session_id,
            "workflow_trace": _trace_event("entry", "workflow_resumed", approval_response=approval_response),
            "workflow_status": "resume",
            "approval_response": approval_response,
            "final_reply": "",
        },
        _graph_config(session_id),
    )
    return final_state.get("final_reply") or final_state["messages"][-1].get("content") or ""
