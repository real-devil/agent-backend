"""LangGraph graph builder — compiles the StateGraph with all nodes and conditional edges.

Node inventory (14 nodes):
  Safety layer:
    input_gate → rate_limiter → tool_guard → output_gate → circuit_breaker → audit_log
  Orchestration layer:
    entry → planner → approval_gate → execute_group → reviewer
    → advance_group / rollback_group / synthesizer

Flow:
  START → input_gate → rate_limiter → entry → planner → approval_gate
             ↓ rejected           ↓ rejected                    ↓ awaiting → END
             ↓                    ↓                             ↓ rejected
             └────────────────────┴─────────────────────────────┘
                                          ↓
                                    tool_guard → execute_group → reviewer
                                                                   ↓
                                                       advance / rollback / synthesizer
                                                                   ↓
                                            output_gate → circuit_breaker → audit_log → END
"""

from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.core.plan_utils import group_index_by_step_id
from agents.nodes.advance import advance_group, rollback_group
from agents.nodes.approval import approval_gate
from agents.nodes.audit_log import audit_log
from agents.nodes.circuit_breaker import circuit_breaker
from agents.nodes.entry import entry, route_from_entry
from agents.nodes.executor import execute_group
from agents.nodes.input_gate import input_gate
from agents.nodes.output_gate import output_gate
from agents.nodes.planner import planner
from agents.nodes.rate_limiter import rate_limiter
from agents.nodes.reviewer import reviewer
from agents.nodes.synthesizer import synthesizer
from agents.nodes.tool_guard import tool_guard
from agents.state import AgentState


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_safety_gate(state: AgentState) -> str:
    """After input_gate / rate_limiter: rejected → skip to output chain, else continue."""
    status = state.get("workflow_status", "")
    if status == "rejected":
        return "output_gate"
    return "__next__"  # caller maps this to the actual next node


def route_after_approval_gate(state: AgentState) -> str:
    status = state.get("workflow_status", "")
    if status == "awaiting_approval":
        return END
    if status == "rejected":
        return "output_gate"
    return "tool_guard"


def route_after_review(state: AgentState) -> str:
    decision = state.get("review_decision", "continue")
    if decision == "retry":
        rollback_group_index = group_index_by_step_id(state, state.get("review_rollback_target", ""))
        if rollback_group_index is not None and rollback_group_index < state.get("current_group_index", 0):
            return "rollback_group"
        return "approval_gate"
    if decision == "continue":
        return "advance_group"
    return "synthesizer"


def route_after_advance(state: AgentState) -> str:
    return "approval_gate"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_agent_graph(checkpointer: Any):
    graph = StateGraph(AgentState)

    # -- safety layer nodes --
    graph.add_node("input_gate", input_gate)
    graph.add_node("rate_limiter", rate_limiter)
    graph.add_node("tool_guard", tool_guard)
    graph.add_node("output_gate", output_gate)
    graph.add_node("circuit_breaker", circuit_breaker)
    graph.add_node("audit_log", audit_log)

    # -- orchestration layer nodes --
    graph.add_node("entry", entry)
    graph.add_node("planner", planner)
    graph.add_node("approval_gate", approval_gate)
    graph.add_node("execute_group", execute_group)
    graph.add_node("reviewer", reviewer)
    graph.add_node("advance_group", advance_group)
    graph.add_node("rollback_group", rollback_group)
    graph.add_node("synthesizer", synthesizer)

    # =====================================================================
    # Safety layer: START → input_gate → rate_limiter → entry
    # =====================================================================
    graph.add_edge(START, "input_gate")
    graph.add_conditional_edges(
        "input_gate",
        route_after_safety_gate,
        {"output_gate": "output_gate", "__next__": "rate_limiter"},
    )
    graph.add_conditional_edges(
        "rate_limiter",
        route_after_safety_gate,
        {"output_gate": "output_gate", "__next__": "entry"},
    )

    # =====================================================================
    # Orchestration layer: entry → planner → approval_gate → tool_guard → execute → review
    # =====================================================================
    graph.add_conditional_edges(
        "entry",
        route_from_entry,
        {"planner": "planner", "approval_gate": "approval_gate"},
    )
    graph.add_edge("planner", "approval_gate")
    graph.add_conditional_edges(
        "approval_gate",
        route_after_approval_gate,
        {
            "tool_guard": "tool_guard",
            "output_gate": "output_gate",
            END: END,
        },
    )
    graph.add_edge("tool_guard", "execute_group")
    graph.add_edge("execute_group", "reviewer")
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "approval_gate": "approval_gate",
            "advance_group": "advance_group",
            "rollback_group": "rollback_group",
            "synthesizer": "synthesizer",
        },
    )
    graph.add_conditional_edges(
        "advance_group",
        route_after_advance,
        {"approval_gate": "approval_gate"},
    )
    graph.add_conditional_edges(
        "rollback_group",
        route_after_advance,
        {"approval_gate": "approval_gate"},
    )

    # =====================================================================
    # Safety output chain: ... → output_gate → circuit_breaker → audit_log → END
    # =====================================================================
    graph.add_edge("synthesizer", "output_gate")
    graph.add_edge("output_gate", "circuit_breaker")
    graph.add_edge("circuit_breaker", "audit_log")
    graph.add_edge("audit_log", END)

    return graph.compile(checkpointer=checkpointer)
