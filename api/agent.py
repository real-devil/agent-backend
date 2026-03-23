import json
import os
import logging
from fastapi import APIRouter
from pydantic import BaseModel
from openai import AsyncOpenAI
from tools.weather import WEATHER_TOOL, get_weather
from tools.search import SEARCH_TOOL, search_documents

logger = logging.getLogger(__name__)

router = APIRouter()

TOOLS = [WEATHER_TOOL, SEARCH_TOOL]

SYSTEM_PROMPT = (
    "你是一个智能助手，可以调用工具来回答用户问题。"
    "需要查询文档知识库时调用 search_documents，"
    "需要查询天气时调用 get_weather，"
    "普通问题直接回答。"
)


def _get_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


class AgentRequest(BaseModel):
    input: str
    session_id: str | None = None
    document_id: str | None = None  # 透传给 search_documents


@router.post("/run")
async def run_agent(request: AgentRequest):
    client = _get_client()
    model = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request.input},
    ]

    for _ in range(5):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            return {"reply": msg.content, "session_id": request.session_id}

        messages.append(msg)

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            if fn_name == "get_weather":
                result = await get_weather(**fn_args)
            elif fn_name == "search_documents":
                # 如果请求带了 document_id，且 LLM 没有指定，则自动注入
                if request.document_id and "document_id" not in fn_args:
                    fn_args["document_id"] = request.document_id
                result = await search_documents(**fn_args)
            else:
                result = f"未知工具：{fn_name}"

            print(f"[tool_call] {fn_name}({fn_args}) => {result}", flush=True)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return {"reply": msg.content or "处理超时", "session_id": request.session_id}
