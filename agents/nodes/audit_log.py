"""Audit log node — records every workflow completion for compliance."""

import json
import logging
import time

from agents.core.safety_utils import safety_gates_enabled
from agents.tracing.event_factory import state_trace_event
from agents.state import AgentState

logger = logging.getLogger("audit")

# ---------------------------------------------------------------------------
# 生产环境应对接：
#   - 写入专用审计数据库（不可删除、不可篡改）
#   - 结构化日志格式（JSON → Elasticsearch / Splunk）
#   - 包含：user_id / session_id / turn_id / 每步 LLM token / tool 调用
# ---------------------------------------------------------------------------


async def audit_log(state: AgentState) -> AgentState:
    """Record a structured audit entry for this turn."""
    if not safety_gates_enabled():
        return {}

    turn_id = state.get("current_turn_id", "?")
    session_id = state.get("session_id", "?")
    status = state.get("workflow_status", "?")
    metrics = state.get("metrics_summary", {})

    entry = {
        "timestamp": time.time(),
        "session_id": str(session_id),
        "turn_id": str(turn_id),
        "workflow_status": str(status),
        "total_duration_ms": metrics.get("total_duration_ms", 0),
        "total_tokens": metrics.get("total_tokens", 0),
        "total_model_calls": metrics.get("total_model_calls", 0),
        "total_tool_calls": metrics.get("total_tool_calls", 0),
        "step_count": len(state.get("step_results", [])),
    }

    logger.info("AUDIT: %s", json.dumps(entry, ensure_ascii=False))
    return {
        "workflow_trace": state_trace_event(state, "audit_log", "audit_recorded"),
    }
