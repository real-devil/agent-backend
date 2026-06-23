"""Artifact record builders and helpers for passing data between steps."""

import json
from typing import Any

from agents.schemas import ArtifactRecord, StructuredStepOutput
from agents.state import AgentState


def artifact_context_text(state: AgentState) -> str:
    artifacts = state.get("artifacts", {})
    if not artifacts:
        return "{}"
    return json.dumps(artifacts, ensure_ascii=False)


def parse_structured_step_output(raw_content: str, output_key: str) -> dict[str, Any]:
    from agents.core.message_utils import parse_json_object

    payload = parse_json_object(raw_content)
    if not payload:
        return StructuredStepOutput(
            summary=raw_content.strip() or output_key,
            artifact_type="analysis",
            artifact_data=raw_content.strip(),
            confidence="medium",
        ).model_dump()

    summary = str(payload.get("summary", "")).strip() or output_key
    artifact_type = str(payload.get("artifact_type", "analysis")).strip().lower() or "analysis"
    confidence = str(payload.get("confidence", "medium")).strip().lower() or "medium"

    try:
        return StructuredStepOutput(
            summary=summary,
            artifact_type=artifact_type,
            artifact_data=payload.get("artifact_data", summary),
            confidence=confidence,
        ).model_dump()
    except Exception:
        return StructuredStepOutput(
            summary=summary,
            artifact_type="analysis",
            artifact_data=payload.get("artifact_data", summary),
            confidence="medium",
        ).model_dump()


def build_artifact_record(
    *,
    step: dict[str, Any],
    agent: str,
    summary: str,
    artifact_type: str,
    artifact_data: Any,
    confidence: str,
) -> dict[str, Any]:
    return ArtifactRecord(
        step_id=step["id"],
        output_key=step["output_key"],
        agent=agent,
        artifact_type=artifact_type,
        summary=summary,
        confidence=confidence,
        data=artifact_data,
    ).model_dump()


def rebuild_artifacts_from_step_results(step_results: list[dict[str, Any]]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for result in step_results:
        output_key = str(result.get("output_key", "")).strip()
        artifact = result.get("artifact")
        if output_key and artifact is not None:
            artifacts[output_key] = artifact
    return artifacts


def merge_artifacts(current_artifacts: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(current_artifacts)
    for result in results:
        output_key = str(result.get("output_key", "")).strip()
        artifact = result.get("artifact")
        if output_key and artifact is not None:
            merged[output_key] = artifact
    return merged
