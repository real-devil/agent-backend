import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from agents.state import AgentState
from services.rag import rag_chat
from tools.search import SEARCH_TOOL, search_documents
from tools.weather import WEATHER_TOOL, get_weather

logger = logging.getLogger(__name__)

TOOLS = [WEATHER_TOOL, SEARCH_TOOL]
MAX_TOOL_ITERATIONS = 5
MAX_STEP_RETRIES = 1

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
- Use 1-4 steps.
- Prefer multi-step plans for anything that requires research, tools, or validation.
- If a document_id is present, prefer rag_agent or tool_agent/search_documents when useful.
- The final user-facing answer will be written later by a synthesizer node, so steps should focus on producing useful intermediate outputs.

Schema:
{
  "workflow_status": "simple" | "multi_step",
  "route_reason": "short explanation",
  "success_criteria": ["criterion 1", "criterion 2"],
  "steps": [
    {"id": "step_1", "agent": "research_agent" | "tool_agent" | "rag_agent" | "general_agent", "goal": "what this step should achieve"}
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

REVIEWER_PROMPT = """
You are the reviewer of a multi-agent workflow.
Inspect the current step result and decide whether to continue, retry, or finish.

Return JSON only with this schema:
{
  "decision": "continue" | "retry" | "finish",
  "reason": "short explanation"
}

Guidance:
- retry: the current step result is unusable or clearly insufficient
- continue: the step is acceptable and the workflow should move to the next step
- finish: the workflow already has enough information to produce the final answer
""".strip()

SYNTHESIZER_PROMPT = """
You are the final answer synthesizer of a production multi-agent system.
Use the workflow plan and step results to answer the user's latest request.
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


def _recent_conversation_text(state: AgentState, limit: int = 6) -> str:
    recent = state["messages"][-limit:]
    lines = []
    for message in recent:
        role = message.get("role", "unknown")
        content = str(message.get("content", ""))
        lines.append(f"{role}: {content}")
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
            snippet = raw_content[start : end + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON object: %s", raw_content)
                return {}
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


def _fallback_plan(state: AgentState) -> dict[str, Any]:
    latest_input = _get_latest_user_input(state).lower()
    if state.get("document_id"):
        steps = [{"id": "step_1", "agent": "rag_agent", "goal": _get_latest_user_input(state)}]
        reason = "Document-specific request routed to rag_agent."
    elif any(keyword in latest_input for keyword in ("weather", "temperature", "forecast", "天气")):
        steps = [{"id": "step_1", "agent": "tool_agent", "goal": _get_latest_user_input(state)}]
        reason = "Weather-style request routed to tool_agent."
    else:
        steps = [{"id": "step_1", "agent": "general_agent", "goal": _get_latest_user_input(state)}]
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
    for index, step in enumerate(payload.get("steps", []), start=1):
        agent = str(step.get("agent", "general_agent"))
        if agent not in {"research_agent", "tool_agent", "rag_agent", "general_agent"}:
            agent = "general_agent"
        goal = str(step.get("goal", "")).strip() or _get_latest_user_input(state)
        steps.append(
            {
                "id": str(step.get("id", f"step_{index}")),
                "agent": agent,
                "goal": goal,
            }
        )

    if not steps:
        fallback = _fallback_plan(state)
        steps = fallback["steps"]
        payload = fallback

    return {
        "workflow_status": payload.get("workflow_status", "multi_step" if len(steps) > 1 else "simple"),
        "route_reason": str(payload.get("route_reason", "Planner generated workflow.")),
        "success_criteria": [
            str(item) for item in payload.get("success_criteria", []) if str(item).strip()
        ] or ["Answer the user's request accurately."],
        "steps": steps,
    }


def _get_current_step(state: AgentState) -> dict[str, Any]:
    steps = state.get("workflow_plan", [])
    index = state.get("current_step_index", 0)
    if 0 <= index < len(steps):
        return steps[index]
    return {"id": "step_1", "agent": "general_agent", "goal": _get_latest_user_input(state)}


def _has_remaining_steps(state: AgentState) -> bool:
    return state.get("current_step_index", 0) + 1 < len(state.get("workflow_plan", []))


async def _call_text_model(system_prompt: str, user_prompt: str) -> str:
    response = await _get_client().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


async def _planner(state: AgentState) -> AgentState:
    user_prompt = (
        f"Latest user request:\n{_get_latest_user_input(state)}\n\n"
        f"Conversation context:\n{_recent_conversation_text(state)}\n\n"
        f"document_id present: {'yes' if state.get('document_id') else 'no'}"
    )
    raw_content = await _call_text_model(PLANNER_PROMPT, user_prompt)
    payload = _parse_json_object(raw_content)
    plan = _normalize_plan(payload, state)
    first_step = plan["steps"][0]

    return {
        "workflow_status": plan["workflow_status"],
        "workflow_plan": plan["steps"],
        "route_reason": plan["route_reason"],
        "success_criteria": plan["success_criteria"],
        "current_step_index": 0,
        "current_step_agent": first_step["agent"],
        "current_step_goal": first_step["goal"],
        "current_step_result": "",
        "step_results": [],
        "review_decision": "",
        "review_reason": "",
        "step_retry_count": 0,
        "tool_iterations": 0,
        "final_reply": "",
    }


async def _research_agent(state: AgentState) -> AgentState:
    step = _get_current_step(state)
    if state.get("document_id"):
        result = await search_documents(
            query=step["goal"],
            document_id=state["document_id"],
        )
    else:
        user_prompt = (
            f"Current step goal:\n{step['goal']}\n\n"
            f"Latest user request:\n{_get_latest_user_input(state)}\n\n"
            f"Previous accepted step results:\n{json.dumps(state.get('step_results', []), ensure_ascii=False)}"
        )
        result = await _call_text_model(RESEARCH_AGENT_PROMPT, user_prompt)

    return {
        "current_step_agent": "research_agent",
        "current_step_goal": step["goal"],
        "current_step_result": result,
    }


async def _tool_execute(name: str, args: dict[str, Any], state: AgentState) -> str:
    if name == "get_weather":
        return await get_weather(**args)

    if name == "search_documents":
        if state.get("document_id") and "document_id" not in args:
            args["document_id"] = state["document_id"]
        return await search_documents(**args)

    return f"Unknown tool: {name}"


async def _tool_agent(state: AgentState) -> AgentState:
    step = _get_current_step(state)
    local_messages: list[dict[str, Any]] = [
        {"role": "system", "content": TOOL_AGENT_PROMPT},
        {
            "role": "user",
            "content": (
                f"Current step goal:\n{step['goal']}\n\n"
                f"Latest user request:\n{_get_latest_user_input(state)}\n\n"
                f"Previous accepted step results:\n{json.dumps(state.get('step_results', []), ensure_ascii=False)}"
            ),
        },
    ]

    final_result = "Tool agent ended without producing a final result."
    for _ in range(MAX_TOOL_ITERATIONS):
        response = await _get_client().chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"),
            messages=local_messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        assistant_message = _serialize_assistant_message(response.choices[0].message)

        if not assistant_message.get("tool_calls"):
            final_result = assistant_message.get("content", "") or final_result
            break

        local_messages.append(assistant_message)

        for tool_call in assistant_message["tool_calls"]:
            fn_name = tool_call["function"]["name"]
            raw_args = tool_call["function"]["arguments"]
            try:
                fn_args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError as exc:
                tool_result = f"Tool argument parsing failed for {fn_name}: {exc}"
            else:
                tool_result = await _tool_execute(fn_name, fn_args, state)
                logger.info("tool_agent %s(%s) => %s", fn_name, fn_args, tool_result)

            local_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result,
                }
            )

    return {
        "current_step_agent": "tool_agent",
        "current_step_goal": step["goal"],
        "current_step_result": final_result,
    }


async def _rag_agent(state: AgentState) -> AgentState:
    step = _get_current_step(state)
    result = await rag_chat(
        question=step["goal"],
        document_id=state.get("document_id"),
    )
    return {
        "current_step_agent": "rag_agent",
        "current_step_goal": step["goal"],
        "current_step_result": result or "",
    }


async def _general_agent(state: AgentState) -> AgentState:
    step = _get_current_step(state)
    user_prompt = (
        f"Current step goal:\n{step['goal']}\n\n"
        f"Latest user request:\n{_get_latest_user_input(state)}\n\n"
        f"Previous accepted step results:\n{json.dumps(state.get('step_results', []), ensure_ascii=False)}"
    )
    result = await _call_text_model(GENERAL_AGENT_PROMPT, user_prompt)
    return {
        "current_step_agent": "general_agent",
        "current_step_goal": step["goal"],
        "current_step_result": result,
    }


async def _reviewer(state: AgentState) -> AgentState:
    step = _get_current_step(state)
    user_prompt = (
        f"Latest user request:\n{_get_latest_user_input(state)}\n\n"
        f"Success criteria:\n{json.dumps(state.get('success_criteria', []), ensure_ascii=False)}\n\n"
        f"Current step:\n{json.dumps(step, ensure_ascii=False)}\n\n"
        f"Current step result:\n{state.get('current_step_result', '')}\n\n"
        f"Accepted step results so far:\n{json.dumps(state.get('step_results', []), ensure_ascii=False)}\n\n"
        f"Current retry count: {state.get('step_retry_count', 0)}\n"
        f"Has remaining steps after this one: {'yes' if _has_remaining_steps(state) else 'no'}"
    )
    raw_content = await _call_text_model(REVIEWER_PROMPT, user_prompt)
    payload = _parse_json_object(raw_content)
    decision = str(payload.get("decision", "continue")).lower()
    reason = str(payload.get("reason", "")).strip() or "Reviewer decision applied."

    if decision not in {"continue", "retry", "finish"}:
        decision = "continue"

    retry_count = state.get("step_retry_count", 0)
    if decision == "retry" and retry_count >= MAX_STEP_RETRIES:
        decision = "continue" if _has_remaining_steps(state) else "finish"
        reason = "Retry limit reached; proceeding with the workflow."

    if decision == "continue" and not _has_remaining_steps(state):
        decision = "finish"

    step_results = state.get("step_results", [])
    if decision in {"continue", "finish"}:
        step_results = step_results + [
            {
                "step_id": step["id"],
                "agent": step["agent"],
                "goal": step["goal"],
                "result": state.get("current_step_result", ""),
            }
        ]

    return {
        "review_decision": decision,
        "review_reason": reason,
        "step_results": step_results,
        "step_retry_count": retry_count + 1 if decision == "retry" else 0,
    }


async def _advance_step(state: AgentState) -> AgentState:
    next_index = state.get("current_step_index", 0) + 1
    next_step = state.get("workflow_plan", [])[next_index]
    return {
        "current_step_index": next_index,
        "current_step_agent": next_step["agent"],
        "current_step_goal": next_step["goal"],
        "current_step_result": "",
        "review_decision": "",
        "review_reason": "",
        "step_retry_count": 0,
    }


async def _synthesizer(state: AgentState) -> AgentState:
    user_prompt = (
        f"Latest user request:\n{_get_latest_user_input(state)}\n\n"
        f"Workflow reason:\n{state.get('route_reason', '')}\n\n"
        f"Workflow plan:\n{json.dumps(state.get('workflow_plan', []), ensure_ascii=False)}\n\n"
        f"Accepted step results:\n{json.dumps(state.get('step_results', []), ensure_ascii=False)}"
    )
    final_reply = await _call_text_model(SYNTHESIZER_PROMPT, user_prompt)
    return {
        "final_reply": final_reply,
        "workflow_status": "completed",
        "messages": [{"role": "assistant", "content": final_reply}],
    }


def _route_current_step(state: AgentState) -> str:
    agent = _get_current_step(state)["agent"]
    if agent in {"research_agent", "tool_agent", "rag_agent", "general_agent"}:
        return agent
    return "general_agent"


def _route_after_review(state: AgentState) -> str:
    decision = state.get("review_decision", "continue")
    if decision == "retry":
        return _route_current_step(state)
    if decision == "continue":
        return "advance_step"
    return "synthesizer"


def _route_after_advance(state: AgentState) -> str:
    return _route_current_step(state)


def build_agent_graph(checkpointer: Any):
    graph = StateGraph(AgentState)
    graph.add_node("planner", _planner)
    graph.add_node("research_agent", _research_agent)
    graph.add_node("tool_agent", _tool_agent)
    graph.add_node("rag_agent", _rag_agent)
    graph.add_node("general_agent", _general_agent)
    graph.add_node("reviewer", _reviewer)
    graph.add_node("advance_step", _advance_step)
    graph.add_node("synthesizer", _synthesizer)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        _route_current_step,
        {
            "research_agent": "research_agent",
            "tool_agent": "tool_agent",
            "rag_agent": "rag_agent",
            "general_agent": "general_agent",
        },
    )
    graph.add_edge("research_agent", "reviewer")
    graph.add_edge("tool_agent", "reviewer")
    graph.add_edge("rag_agent", "reviewer")
    graph.add_edge("general_agent", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        _route_after_review,
        {
            "research_agent": "research_agent",
            "tool_agent": "tool_agent",
            "rag_agent": "rag_agent",
            "general_agent": "general_agent",
            "advance_step": "advance_step",
            "synthesizer": "synthesizer",
        },
    )
    graph.add_conditional_edges(
        "advance_step",
        _route_after_advance,
        {
            "research_agent": "research_agent",
            "tool_agent": "tool_agent",
            "rag_agent": "rag_agent",
            "general_agent": "general_agent",
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


async def run_agent_graph(
    user_input: str,
    session_id: str,
    document_id: str | None = None,
) -> str:
    graph = await _get_agent_graph()
    config = _graph_config(session_id)

    initial_state: AgentState = {
        "messages": [{"role": "user", "content": user_input}],
        "session_id": session_id,
        "document_id": document_id,
        "workflow_status": "planning",
        "workflow_plan": [],
        "success_criteria": [],
        "current_step_index": 0,
        "current_step_result": "",
        "current_step_agent": "",
        "current_step_goal": "",
        "review_decision": "",
        "review_reason": "",
        "step_results": [],
        "step_retry_count": 0,
        "tool_iterations": 0,
        "final_reply": "",
    }

    final_state = await graph.ainvoke(initial_state, config)
    if final_state.get("final_reply"):
        return final_state["final_reply"]

    last_message = final_state["messages"][-1]
    return last_message.get("content") or "Processing finished without a final answer."
