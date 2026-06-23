"""LangGraph graph builder — compiles the StateGraph with all nodes and conditional edges."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.core.plan_utils import group_index_by_step_id
from agents.nodes.advance import advance_group, rollback_group
from agents.nodes.approval import approval_gate
from agents.nodes.entry import entry, route_from_entry
from agents.nodes.executor import execute_group
from agents.nodes.planner import planner
from agents.nodes.reviewer import reviewer
from agents.nodes.synthesizer import synthesizer
from agents.state import AgentState


def route_after_approval_gate(state: AgentState) -> str:
    status = state.get("workflow_status", "")
    if status in {"awaiting_approval", "rejected"}:
        return END
    return "execute_group"


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


def build_agent_graph(checkpointer: Any):
    graph = StateGraph(AgentState)
    graph.add_node("entry", entry)
    graph.add_node("planner", planner)
    graph.add_node("approval_gate", approval_gate)
    graph.add_node("execute_group", execute_group)
    graph.add_node("reviewer", reviewer)
    graph.add_node("advance_group", advance_group)
    graph.add_node("rollback_group", rollback_group)
    graph.add_node("synthesizer", synthesizer)

    graph.add_edge(START, "entry")
    graph.add_conditional_edges(
        "entry",
        route_from_entry,
        {
            "planner": "planner",
            "approval_gate": "approval_gate",
        },
    )
    graph.add_edge("planner", "approval_gate")
    graph.add_conditional_edges(
        "approval_gate",
        route_after_approval_gate,
        {
            "execute_group": "execute_group",
            END: END,
        },
    )
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
        {
            "approval_gate": "approval_gate",
        },
    )
    graph.add_conditional_edges(
        "rollback_group",
        route_after_advance,
        {
            "approval_gate": "approval_gate",
        },
    )
    graph.add_edge("synthesizer", END)

    return graph.compile(checkpointer=checkpointer)
