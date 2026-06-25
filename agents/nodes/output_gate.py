"""Output gate node — filters and validates final reply before returning to user."""

import logging

from agents.core.safety_utils import gate_passed_trace, safety_gates_enabled
from agents.core.trace_utils import state_trace_event
from agents.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 生产环境应对接：
#   - 内容安全模型（OpenAI Moderation / Llama Guard）
#   - PII 检测与脱敏
#   - 幻觉检测（与 source documents 比对）
#   - 合规关键词检查
# ---------------------------------------------------------------------------

BLOCKED_OUTPUT_PATTERNS: list[str] = []  # 按业务需求配置


def _check_output_safety(text: str) -> str | None:
    """Returns rejection reason or None if safe."""
    if not text.strip():
        return "empty_response"
    lower = text.lower()
    for pattern in BLOCKED_OUTPUT_PATTERNS:
        if pattern in lower:
            return f"blocked_pattern: {pattern}"
    return None


async def output_gate(state: AgentState) -> AgentState:
    """Validate the final reply before returning to the user."""
    if not safety_gates_enabled():
        return {}

    final_reply = state.get("final_reply", "") or ""
    rejection = _check_output_safety(final_reply)

    if rejection:
        safe_message = "抱歉，当前回答无法显示，请重新提问。"
        logger.warning("output_gate blocked: reason=%s", rejection)
        return {
            "final_reply": safe_message,
            "messages": [{"role": "assistant", "content": safe_message}],
            "workflow_trace": state_trace_event(
                state,
                "output_gate",
                "gate_rejected",
                reason=rejection,
            ),
        }

    return {"workflow_trace": gate_passed_trace(state, "output_gate")}
