"""Metrics accumulation utilities for the agent runtime."""

from typing import Any

from agents.schemas import MetricsSummary


def default_metrics_summary() -> dict[str, Any]:
    return MetricsSummary().model_dump()


def usage_to_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def merge_metrics(
    current: dict[str, Any] | None,
    *,
    duration_ms: int = 0,
    usage: dict[str, int] | None = None,
    model_calls: int = 0,
    tool_calls: int = 0,
    approval_requested: int = 0,
    approval_granted: int = 0,
    approval_rejected: int = 0,
    rollback_count: int = 0,
    failure_category: str | None = None,
) -> dict[str, Any]:
    merged = dict(current or default_metrics_summary())
    merged["total_duration_ms"] = int(merged.get("total_duration_ms", 0)) + int(duration_ms)
    merged["total_model_calls"] = int(merged.get("total_model_calls", 0)) + int(model_calls)
    merged["total_tool_calls"] = int(merged.get("total_tool_calls", 0)) + int(tool_calls)
    merged["approval_requests"] = int(merged.get("approval_requests", 0)) + int(approval_requested)
    merged["approval_grants"] = int(merged.get("approval_grants", 0)) + int(approval_granted)
    merged["approval_rejections"] = int(merged.get("approval_rejections", 0)) + int(approval_rejected)
    merged["rollback_count"] = int(merged.get("rollback_count", 0)) + int(rollback_count)

    normalized_usage = usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    merged["prompt_tokens"] = int(merged.get("prompt_tokens", 0)) + int(normalized_usage.get("prompt_tokens", 0))
    merged["completion_tokens"] = int(merged.get("completion_tokens", 0)) + int(normalized_usage.get("completion_tokens", 0))
    merged["total_tokens"] = int(merged.get("total_tokens", 0)) + int(normalized_usage.get("total_tokens", 0))

    failure_counts = dict(merged.get("failure_counts", {}))
    if failure_category and failure_category != "none":
        failure_counts[failure_category] = int(failure_counts.get(failure_category, 0)) + 1
    merged["failure_counts"] = failure_counts
    return merged
