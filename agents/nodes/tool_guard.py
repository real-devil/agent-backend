"""Tool guard node — validates tool permissions before step execution."""

import logging

from agents.core.safety_utils import gate_passed_trace, safety_gates_enabled
from agents.state import AgentState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 生产环境应对接：
#   - RBAC 策略引擎（如 OPA / Casbin）
#   - 按 user / tenant / plan 限制 tool 可用列表
#   - 参数白名单校验（如只允许查特定城市天气）
# ---------------------------------------------------------------------------

AGENT_TOOL_WHITELIST: dict[str, set[str]] = {
    "tool_agent": {"get_weather", "search_documents"},
    "research_agent": {"search_documents"},
    "rag_agent": set(),
    "general_agent": set(),
}


async def tool_guard(state: AgentState) -> AgentState:
    """Check current group steps against tool whitelist before execution."""
    if not safety_gates_enabled():
        return {}

    from agents.core.plan_utils import get_current_group_steps

    steps = get_current_group_steps(state)
    for step in steps:
        agent = str(step.get("agent", ""))
        if agent not in AGENT_TOOL_WHITELIST:
            continue
        allowed = AGENT_TOOL_WHITELIST[agent]
        logger.debug("tool_guard: agent=%s allowed_tools=%s", agent, allowed)

    return {"workflow_trace": gate_passed_trace(state, "tool_guard")}
