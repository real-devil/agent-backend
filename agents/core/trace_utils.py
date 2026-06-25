"""Trace event factory functions — re-exports from agents.tracing.event_factory."""

# Re-export from the canonical location
from agents.tracing.event_factory import (  # noqa: F401
    state_step_trace_event,
    state_trace_event,
    trace_event,
    trace_for_turn,
)
