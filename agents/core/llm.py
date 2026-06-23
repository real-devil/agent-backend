"""LLM client factory and call helpers (streaming + non-streaming)."""

import os
import time
from typing import Any

from openai import AsyncOpenAI

from agents.core.constants import DEFAULT_MODEL
from agents.core.prompts import STRUCTURED_STEP_OUTPUT_PROMPT


def get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


def get_model_name() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


async def stream_text_model(
    system_prompt: str,
    user_prompt: str,
    *,
    on_delta: Any | None = None,
) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    stream = await get_client().chat.completions.create(
        model=get_model_name(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )
    parts: list[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    async for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            parts.append(delta)
            if on_delta is not None:
                on_delta(delta)
        if getattr(chunk, "usage", None) is not None:
            from agents.core.metrics_utils import usage_to_dict
            usage = usage_to_dict(chunk.usage)

    duration_ms = int((time.perf_counter() - started) * 1000)
    return "".join(parts), {"duration_ms": duration_ms, "usage": usage}


async def call_text_model(system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
    from agents.core.metrics_utils import usage_to_dict

    started = time.perf_counter()
    response = await get_client().chat.completions.create(
        model=get_model_name(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    duration_ms = int((time.perf_counter() - started) * 1000)
    usage = usage_to_dict(getattr(response, "usage", None))
    return response.choices[0].message.content or "", {
        "duration_ms": duration_ms,
        "usage": usage,
    }


async def call_structured_step_model(system_prompt: str, user_prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from agents.core.message_utils import parse_json_object

    raw_content, meta = await call_text_model(
        f"{system_prompt}\n\n{STRUCTURED_STEP_OUTPUT_PROMPT}",
        user_prompt,
    )
    return parse_json_object(raw_content), meta
