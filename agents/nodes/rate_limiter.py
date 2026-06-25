"""Rate limiter node — per-session throttle before entering the workflow."""

import logging
import time
from threading import Lock

from agents.core.safety_utils import gate_passed_trace, safety_gates_enabled, should_bypass_input_gates
from agents.core.trace_utils import state_trace_event
from agents.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 生产环境应对接：
#   - Redis 滑动窗口（跨实例共享）
#   - 按 user_id / tenant_id 分组限流
#   - 动态配额（付费 tier）
# ---------------------------------------------------------------------------

_lock = Lock()
_window: dict[str, list[float]] = {}   # session_id → [timestamp, ...]

MAX_REQUESTS_PER_MINUTE = 30
WINDOW_SECONDS = 60


def _check_and_record(session_id: str) -> bool:
    now = time.time()
    with _lock:
        timestamps = _window.get(session_id, [])
        timestamps = [t for t in timestamps if now - t < WINDOW_SECONDS]
        if len(timestamps) >= MAX_REQUESTS_PER_MINUTE:
            _window[session_id] = timestamps
            return False
        timestamps.append(now)
        _window[session_id] = timestamps
        return True


async def rate_limiter(state: AgentState) -> AgentState:
    """Throttle requests per session.

    On rejection: sets workflow_status="rejected" + final_reply.
    """
    if not safety_gates_enabled() or should_bypass_input_gates(state):
        return {}

    session_id = str(state.get("session_id") or "")
    if not session_id:
        return {}

    allowed = _check_and_record(session_id)
    if not allowed:
        message = "请求过于频繁，请稍后再试。"
        logger.warning("rate_limiter blocked: session=%s", session_id)
        return {
            "workflow_status": "rejected",
            "final_reply": message,
            "messages": [{"role": "assistant", "content": message}],
            "workflow_trace": state_trace_event(
                state,
                "rate_limiter",
                "rate_limited",
            ),
        }

    return {"workflow_trace": gate_passed_trace(state, "rate_limiter")}
