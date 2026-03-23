from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from api import agent, chat, document

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent.router, prefix="/agent", tags=["agent"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(document.router, prefix="/document", tags=["document"])


@app.get("/")
def root():
    return {"message": "Agent 后端初始化成功！", "python_version": os.environ.get("PYTHON_VERSION")}


@app.get("/health/db")
def check_db():
    from db.supabase_client import get_client
    db = get_client()
    result = db.table("document_chunks").select("id").limit(1).execute()
    return {"status": "connected", "data": result.data}
