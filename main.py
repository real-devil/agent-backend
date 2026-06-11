import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.runtime import (
    get_checkpointer_kind,
    get_runtime_health,
    initialize_agent_runtime,
    shutdown_agent_runtime,
)
from api import agent, chat, document

load_dotenv()
app = FastAPI()


def _get_cors_origins() -> list[str]:
    raw_origins = os.getenv("BACKEND_CORS_ORIGINS", "http://localhost:3000")
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return origins or ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent.router, prefix="/agent", tags=["agent"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(document.router, prefix="/document", tags=["document"])


@app.on_event("startup")
async def startup_agent_runtime():
    await initialize_agent_runtime()


@app.on_event("shutdown")
async def shutdown_agent_runtime_handler():
    await shutdown_agent_runtime()


@app.get("/")
def root():
    return {
        "message": "Agent backend initialized",
        "python_version": os.environ.get("PYTHON_VERSION"),
        "checkpointer": get_checkpointer_kind(),
    }


@app.get("/health/db")
def check_db():
    from db.supabase_client import get_client

    db = get_client()
    result = db.table("document_chunks").select("id").limit(1).execute()
    return {"status": "connected", "data": result.data}


@app.get("/health/agent")
def check_agent():
    health = get_runtime_health()
    health["status"] = "ok" if health["initialized"] and not health["config_errors"] else "degraded"
    return health
