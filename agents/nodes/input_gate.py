"""Input gate node — validates and sanitizes user input before it reaches Planner."""

import logging

from agents.core.message_utils import get_latest_user_input
from agents.core.safety_utils import gate_passed_trace, safety_gates_enabled, should_bypass_input_gates
from agents.core.trace_utils import state_trace_event
from agents.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 生产环境应对接：
#   - PII 脱敏服务（如 AWS Comprehend / Azure PII）
#   - 越狱检测模型（如 Llama Guard / OpenAI Moderation）
#   - 敏感词库（本地或远程）
# ---------------------------------------------------------------------------

MAX_INPUT_LENGTH = 16_000
BLOCKED_PATTERNS = [
    "ignore all previous instructions",
    "ignore previous instructions",
    "disregard your system prompt",
    "forget your instructions",
]


def _contains_blocked_pattern(text: str) -> str | None:
    lower = text.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in lower:
            return pattern
    return None


async def input_gate(state: AgentState) -> AgentState:
    """Validate user input before passing to Planner.

    On rejection: sets workflow_status="rejected" + final_reply.
    """
    if not safety_gates_enabled() or should_bypass_input_gates(state):
        return {}

    user_input = get_latest_user_input(state)

    if len(user_input) > MAX_INPUT_LENGTH:
        message = f"输入过长（{len(user_input)} 字符），最大允许 {MAX_INPUT_LENGTH} 字符。"
        logger.warning("input_gate blocked: length=%d", len(user_input))
        return {
            "workflow_status": "rejected",
            "final_reply": message,
            "messages": [{"role": "assistant", "content": message}],
            "workflow_trace": state_trace_event(
                state,
                "input_gate",
                "gate_rejected",
                reason="input_too_long",
            ),
        }

    blocked = _contains_blocked_pattern(user_input)
    if blocked:
        message = "输入包含不允许的内容，请重新表述您的问题。"
        logger.warning("input_gate blocked: pattern=%s", blocked)
        return {
            "workflow_status": "rejected",
            "final_reply": message,
            "messages": [{"role": "assistant", "content": message}],
            "workflow_trace": state_trace_event(
                state,
                "input_gate",
                "gate_rejected",
                reason="blocked_pattern",
            ),
        }

    return {"workflow_trace": gate_passed_trace(state, "input_gate")}
