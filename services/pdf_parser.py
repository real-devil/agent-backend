import fitz  # pymupdf


def parse_pdf(file_bytes: bytes) -> str:
    """PDF 字节流 → 纯文本"""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    texts = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(texts)
