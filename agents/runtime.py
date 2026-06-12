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

SUPERVISOR_PROMPT = """
You are the supervisor of a small multi-agent system.
Decide which specialist should handle the user's latest request.

Routes:
- rag: use for questions about uploaded documents, files, contracts, reports, or knowledge-base retrieval
- general: use for everything else, including weather and normal conversation

Return JSON only with this schema:
{"route":"rag"|"general","reason":"short explanation"}
""".strip()

GENERAL_AGENT_PROMPT = (
    "You are a helpful AI assistant. "
    "Use search_documents for questions about uploaded files, reports, contracts, or other document content. "
    "Use get_weather for weather questions. "
    "Answer directly when no tool is needed."
)

_agent_graph: Any | None = None
_checkpointer_cm: AsyncIterator[Any] | None = None
_checkpointer_kind = "memory"


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


async def _supervisor(state: AgentState) -> AgentState:
    if state.get("document_id"):
        return {
            "route": "rag",
            "route_reason": "Request is scoped to a specific uploaded document.",
        }

    user_input = _get_latest_user_input(state)
    response = await _get_client().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {"role": "user", "content": user_input},
        ],
    )

    raw_content = response.choices[0].message.content or "{}"
    try:
        payload = json.loads(raw_content)
    except json.JSONDecodeError:
        logger.warning("Supervisor returned invalid JSON: %s", raw_content)
        return {
            "route": "general",
            "route_reason": "Fallback to general because supervisor output was invalid.",
        }

    route = payload.get("route", "general")
    if route not in {"rag", "general"}:
        route = "general"

    return {
        "route": route,
        "route_reason": payload.get("reason", ""),
    }


async def _general_agent(state: AgentState) -> AgentState:
    response = await _get_client().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"),
        messages=[
            {"role": "system", "content": GENERAL_AGENT_PROMPT},
            *state["messages"][1:],
        ],
        tools=TOOLS,
        tool_choice="auto",
    )
    assistant_message = _serialize_assistant_message(response.choices[0].message)
    next_state: AgentState = {"messages": [assistant_message]}

    if not assistant_message.get("tool_calls"):
        next_state["final_reply"] = assistant_message.get("content", "")

    return next_state


async def _rag_agent(state: AgentState) -> AgentState:
    reply = await rag_chat(
        question=_get_latest_user_input(state),
        document_id=state.get("document_id"),
    )
    return {"final_reply": reply or ""}


async def _execute_tool(name: str, args: dict[str, Any], state: AgentState) -> str:
    if name == "get_weather":
        return await get_weather(**args)

    if name == "search_documents":
        if state.get("document_id") and "document_id" not in args:
            args["document_id"] = state["document_id"]
        return await search_documents(**args)

    return f"Unknown tool: {name}"


async def _run_tools(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    tool_messages: list[dict[str, Any]] = []

    for tool_call in last_message.get("tool_calls", []):
        fn_name = tool_call["function"]["name"]
        raw_args = tool_call["function"]["arguments"]

        try:
            fn_args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as exc:
            result = f"Tool argument parsing failed for {fn_name}: {exc}"
        else:
            result = await _execute_tool(fn_name, fn_args, state)
            logger.info("tool_call %s(%s) => %s", fn_name, fn_args, result)

        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result,
            }
        )

    return {"messages": tool_messages, "tool_iterations": state.get("tool_iterations", 0) + 1}


def _route_after_supervisor(state: AgentState) -> str:
    return "rag_agent" if state.get("route") == "rag" else "general_agent"


def _route_after_general_agent(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.get("tool_calls") and state.get("tool_iterations", 0) < MAX_TOOL_ITERATIONS:
        return "run_tools"
    return END


def build_agent_graph(checkpointer: Any):
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", _supervisor)
    graph.add_node("general_agent", _general_agent)
    graph.add_node("rag_agent", _rag_agent)
    graph.add_node("run_tools", _run_tools)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "rag_agent": "rag_agent",
            "general_agent": "general_agent",
        },
    )
    graph.add_conditional_edges(
        "general_agent",
        _route_after_general_agent,
        {
            "run_tools": "run_tools",
            END: END,
        },
    )
    graph.add_edge("run_tools", "general_agent")
    graph.add_edge("rag_agent", END)

    return graph.compile(checkpointer=checkpointer)


async def initialize_agent_runtime() -> None:
    global _agent_graph, _checkpointer_cm, _checkpointer_kind
    if _agent_graph is not None:
        return

    checkpointer, checkpointer_cm, checkpointer_kind = await _build_checkpointer()
    _agent_graph = build_agent_graph(checkpointer)
    _checkpointer_cm = checkpointer_cm
    _checkpointer_kind = checkpointer_kind
    logger.info("Initialized agent runtime with %s checkpointer", checkpointer_kind)


async def shutdown_agent_runtime() -> None:
    global _agent_graph, _checkpointer_cm, _checkpointer_kind
    if _checkpointer_cm is not None:
        await _checkpointer_cm.__aexit__(None, None, None)
    _agent_graph = None
    _checkpointer_cm = None
    _checkpointer_kind = "memory"


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
    existing_state = await graph.aget_state(config)

    incoming_messages: list[dict[str, Any]] = []
    if not existing_state.values:
        incoming_messages.append({"role": "system", "content": GENERAL_AGENT_PROMPT})
    incoming_messages.append({"role": "user", "content": user_input})

    initial_state: AgentState = {
        "messages": incoming_messages,
        "session_id": session_id,
        "document_id": document_id,
        "tool_iterations": 0,
    }

    final_state = await graph.ainvoke(initial_state, config)
    if final_state.get("final_reply"):
        return final_state["final_reply"]

    last_message = final_state["messages"][-1]
    if last_message.get("role") == "assistant" and not last_message.get("tool_calls"):
        return last_message.get("content") or ""

    return "Processing timed out before the assistant produced a final answer."
