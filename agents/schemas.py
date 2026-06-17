from typing import Any, Literal

from pydantic import BaseModel, Field


class StepSpec(BaseModel):
    id: str
    agent: Literal["research_agent", "tool_agent", "rag_agent", "general_agent"]
    goal: str
    parallel_group: int = 0
    approval_required: bool = False
    depends_on: list[str] = Field(default_factory=list)
    output_key: str


class StructuredStepOutput(BaseModel):
    summary: str
    artifact_type: Literal["notes", "facts", "answer", "analysis", "tool_result"] = "analysis"
    artifact_data: Any
    confidence: Literal["high", "medium", "low"] = "medium"


class ArtifactRecord(BaseModel):
    step_id: str
    output_key: str
    agent: str
    artifact_type: Literal["notes", "facts", "answer", "analysis", "tool_result"]
    summary: str
    confidence: Literal["high", "medium", "low"] = "medium"
    data: Any


class ReviewPayload(BaseModel):
    decision: Literal["continue", "retry", "finish"] = "continue"
    reason: str = ""
    failure_category: Literal["none", "missing_info", "tool_failure", "low_confidence", "invalid_plan"] = "none"
    rollback_to_step_id: str = ""


class TraceEvent(BaseModel):
    event_type: str
    node: str
    turn_id: str | None = None
    display_label: str | None = None
    activity_kind: Literal[
        "session",
        "plan",
        "parallel",
        "step",
        "tool",
        "approval",
        "review",
        "synthesize",
        "system",
    ] | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class MetricsSummary(BaseModel):
    total_duration_ms: int = 0
    total_model_calls: int = 0
    total_tool_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    approval_requests: int = 0
    approval_grants: int = 0
    approval_rejections: int = 0
    rollback_count: int = 0
    failure_counts: dict[str, int] = Field(default_factory=dict)
