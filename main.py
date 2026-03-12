from fastapi import FastAPI
import os
from dotenv import load_dotenv
from api import agent, chat

load_dotenv()
app = FastAPI()

app.include_router(agent.router, prefix="/agent", tags=["agent"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])


@app.get("/")
def root():
    return {"message": "Agent 后端初始化成功！", "python_version": os.environ.get("PYTHON_VERSION")}
