SYSTEM_PROMPT = (
    "你是一个专业的文档助手。"
    "请根据提供的文档内容回答用户问题，回答要简洁准确。"
    "如果文档中没有相关信息，请直接说明无法从文档中找到答案，不要编造内容。"
)

def build_context_message(chunks: list[str]) -> str:
    parts = [f"[文档片段 {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)]
    return "以下是与问题相关的文档内容：\n\n" + "\n\n".join(parts)
