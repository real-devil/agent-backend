"""Circuit breaker node — stops workflow after consecutive failures."""

import logging
from threading import Lock

from agents.core.safety_utils import gate_passed_trace, is_circuit_failure, safety_gates_enabled
from agents.tracing.event_factory import state_trace_event
from agents.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 生产环境应对接：
#   - Redis 计数器（跨实例共享）
#   - 分级熔断（soft: 降级回复 / hard: 完全拒绝）
#   - 自动恢复（冷却时间后尝试放行一个请求）
# ---------------------------------------------------------------------------

_lock = Lock()
_failure_counts: dict[str, int] = {}    # session_id → consecutive failures

MAX_CONSECUTIVE_FAILURES = 3


def _record_and_check(session_id: str, is_failure: bool) -> bool:
    """Returns True if circuit is OPEN (should reject)."""
    with _lock:
        if is_failure:
            _failure_counts[session_id] = _failure_counts.get(session_id, 0) + 1
        else:
            _failure_counts[session_id] = 0
        return _failure_counts.get(session_id, 0) >= MAX_CONSECUTIVE_FAILURES


async def circuit_breaker(state: AgentState) -> AgentState:
    """Check failure count. If threshold reached, short-circuit with degraded reply.

    Only timed_out workflows increment the failure counter.
    Safety gate rejections and user approval rejections do not.
    """
    if not safety_gates_enabled():
        return {}

    session_id = str(state.get("session_id") or "")
    is_failure = is_circuit_failure(state)
    circuit_open = _record_and_check(session_id, is_failure)

    if circuit_open:
        message = "系统暂时繁忙，请稍后再试。"
        logger.warning(
            "circuit_breaker OPEN: session=%s failures=%d",
            session_id,
            _failure_counts.get(session_id, 0),
        )
        return {
            "final_reply": message,
            "messages": [{"role": "assistant", "content": message}],
            "workflow_trace": state_trace_event(
                state,
                "circuit_breaker",
                "circuit_open",
            ),
        }

    return {"workflow_trace": gate_passed_trace(state, "circuit_breaker")}
