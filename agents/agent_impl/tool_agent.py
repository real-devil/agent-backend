"""Tool agent — ReAct loop with tool calling (weather, document search)."""

import json
import logging
import time
from typing import Any

from agents.core.artifact_utils import artifact_context_text, build_artifact_record, parse_structured_step_output
from agents.core.constants import MAX_TOOL_ITERATIONS
from agents.core.llm import get_client, get_model_name
from agents.core.message_utils import get_latest_user_input, serialize_assistant_message
from agents.core.metrics_utils import usage_to_dict
from agents.core.prompts import STRUCTURED_STEP_OUTPUT_PROMPT, TOOL_AGENT_PROMPT
from agents.tracing.event_factory import state_step_trace_event
from tools.search import SEARCH_TOOL
from tools.weather import WEATHER_TOOL

logger = logging.getLogger(__name__)

TOOLS = [WEATHER_TOOL, SEARCH_TOOL]


async def _tool_execute(name: str, args: dict[str, Any], state: dict[str, Any]) -> str:
    from tools.search import search_documents
    from tools.weather import get_weather

    if name == "get_weather":
        return await get_weather(**args)
    if name == "search_documents":
        if state.get("document_id") and "document_id" not in args:
            args["document_id"] = state["document_id"]
        return await search_documents(**args)
    return f"Unknown tool: {name}"


async def run_tool_step(step: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    trace_events: list[dict[str, Any]] = []
    trace_events += state_step_trace_event(state, step, "step_started", agent="tool_agent")

    local_messages: list[dict[str, Any]] = [
        {"role": "system", "content": f"{TOOL_AGENT_PROMPT}\n\n{STRUCTURED_STEP_OUTPUT_PROMPT}"},
        {
            "role": "user",
            "content": (
                f"Current step goal:\n{step['goal']}\n\n"
                f"Latest user request:\n{get_latest_user_input(state)}\n\n"
                f"Accepted step results so far:\n{json.dumps(state.get('step_results', []), ensure_ascii=False)}\n\n"
                f"Existing artifacts:\n{artifact_context_text(state)}"
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
        response = await get_client().chat.completions.create(
            model=get_model_name(),
            messages=local_messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        model_call_count += 1
        total_duration_ms += int((time.perf_counter() - started) * 1000)
        usage = usage_to_dict(getattr(response, "usage", None))
        total_usage["prompt_tokens"] += usage["prompt_tokens"]
        total_usage["completion_tokens"] += usage["completion_tokens"]
        total_usage["total_tokens"] += usage["total_tokens"]
        assistant_message = serialize_assistant_message(response.choices[0].message)
        if not assistant_message.get("tool_calls"):
            final_result = assistant_message.get("content", "") or final_result
            final_parsed = parse_structured_step_output(final_result, step["output_key"])
            break
        local_messages.append(assistant_message)
        for tool_call in assistant_message["tool_calls"]:
            tool_call_count += 1
            fn_name = tool_call["function"]["name"]
            raw_args = tool_call["function"]["arguments"]
            trace_events += state_step_trace_event(
                state,
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
            trace_events += state_step_trace_event(
                state,
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
    artifact = build_artifact_record(
        step=step,
        agent="tool_agent",
        summary=str(final_parsed["summary"]),
        artifact_type=str(final_parsed["artifact_type"]),
        artifact_data=final_parsed["artifact_data"],
        confidence=str(final_parsed["confidence"]),
    )
    trace_events += state_step_trace_event(
        state,
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
