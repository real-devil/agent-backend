from fastapi import APIRouter
from pydantic import BaseModel
from openai import AsyncOpenAI
import os

router = APIRouter()


def get_client():
    return AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


@router.post("/")
async def chat(request: ChatRequest):
    response = await get_client().chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": request.message}],
    )
    reply = response.choices[0].message.content
    return {"reply": reply, "session_id": request.session_id}
