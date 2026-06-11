import json
import logging
import os
from typing import Any

from langgraph.graph import END, START, StateGraph
from openai import AsyncOpenAI

from agents.state import AgentState
from tools.search import SEARCH_TOOL, search_documents
from tools.weather import WEATHER_TOOL, get_weather

logger = logging.getLogger(__name__)

TOOLS = [WEATHER_TOOL, SEARCH_TOOL]
MAX_TOOL_ITERATIONS = 5

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Use search_documents for questions about uploaded files, reports, contracts, or other document content. "
    "Use get_weather for weather questions. "
    "Answer directly when no tool is needed."
)


def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


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


async def _call_model(state: AgentState) -> AgentState:
    response = await _get_client().chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"),
        messages=state["messages"],
        tools=TOOLS,
        tool_choice="auto",
    )
    assistant_message = _serialize_assistant_message(response.choices[0].message)
    return {"messages": state["messages"] + [assistant_message]}


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

    return {
        "messages": state["messages"] + tool_messages,
        "tool_iterations": state.get("tool_iterations", 0) + 1,
    }


def _route_after_model(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.get("tool_calls") and state.get("tool_iterations", 0) < MAX_TOOL_ITERATIONS:
        return "run_tools"
    return END


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("call_model", _call_model)
    graph.add_node("run_tools", _run_tools)
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges(
        "call_model",
        _route_after_model,
        {"run_tools": "run_tools", END: END},
    )
    graph.add_edge("run_tools", "call_model")
    return graph.compile()


_agent_graph = build_agent_graph()


async def run_agent_graph(
    user_input: str,
    session_id: str | None = None,
    document_id: str | None = None,
) -> str:
    initial_state: AgentState = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        "session_id": session_id,
        "document_id": document_id,
        "tool_iterations": 0,
    }

    final_state = await _agent_graph.ainvoke(initial_state)
    last_message = final_state["messages"][-1]

    if last_message.get("role") == "assistant" and not last_message.get("tool_calls"):
        return last_message.get("content") or ""

    return "Processing timed out before the assistant produced a final answer."
