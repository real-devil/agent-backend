"""Shared helpers for safety-layer graph nodes."""

import os
from typing import Any

from agents.state import AgentState


def safety_gates_enabled() -> bool:
    """Set SAFETY_GATES_ENABLED=false to bypass all safety nodes (zero-impact deploy)."""
    return os.getenv("SAFETY_GATES_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def should_bypass_input_gates(state: AgentState) -> bool:
    """Resume without new user message should not re-run input_gate / rate_limiter."""
    return bool(state.get("bypass_input_gates"))


def is_circuit_failure(state: AgentState) -> bool:
    """Only execution-level failures trip the circuit — not safety or approval rejects."""
    return str(state.get("workflow_status", "")).strip() == "timed_out"


def gate_passed_trace(state: dict[str, Any], node: str, **detail: Any) -> list[dict[str, Any]]:
    from agents.core.trace_utils import state_trace_event

    return state_trace_event(state, node, "gate_passed", **detail)
