"""Magic numbers and configuration constants for the agent runtime."""

DEFAULT_MODEL = "openai/gpt-4o-mini"
MAX_TOOL_ITERATIONS = 5
MAX_STEP_RETRIES = 1
WORKFLOW_TIMEOUT_MS = 90_000
PLAN_JSON_MARKER = "<plan_json>"

APPROVAL_APPROVED = {"approved", "approve", "yes", "continue"}
APPROVAL_REJECTED = {"rejected", "reject", "no", "deny", "denied", "stop"}
