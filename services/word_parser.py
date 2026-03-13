from docx import Document
import io


def parse_word(file_bytes: bytes) -> str:
    """Word 字节流 → 纯文本"""
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)
