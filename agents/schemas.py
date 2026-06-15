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
    detail: dict[str, Any] = Field(default_factory=dict)
