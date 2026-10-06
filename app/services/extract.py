"""Plain-text extraction from uploaded files."""

from __future__ import annotations

import io
from pathlib import Path

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".yaml", ".yml"}


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _extract_pdf(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ValueError("PDF support needs the 'pypdf' package") from exc

    reader = PdfReader(io.BytesIO(raw))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)


def _extract_docx(raw: bytes) -> str:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ValueError("DOCX support needs the 'python-docx' package") from exc

    document = docx.Document(io.BytesIO(raw))
    return "\n\n".join(p.text.strip() for p in document.paragraphs if p.text.strip())


def extract_text(raw: bytes, *, filename: str) -> str:
    """Return the text of ``raw``, dispatching on the file extension.

    Raises ``ValueError`` for a type this service cannot read, which the API
    surfaces as a 415 rather than silently indexing bytes as mojibake.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        text = _extract_pdf(raw)
    elif suffix == ".docx":
        text = _extract_docx(raw)
    elif suffix in _TEXT_SUFFIXES or not suffix:
        text = _decode(raw)
    else:
        raise ValueError(f"unsupported file type '{suffix}'")

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError("the file contains no extractable text")
    return text
