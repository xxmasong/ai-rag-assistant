"""Split documents into overlapping chunks that respect paragraph boundaries.

Fixed-width splitting cuts sentences in half, which shows up later as retrieved
passages that start mid-thought and read badly in a citation. This splitter packs
whole paragraphs up to the size limit instead, and only falls back to hard
character slicing for a single paragraph that is itself oversized.
"""

from __future__ import annotations

import re

from app.services.types import Chunk

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")



def _split_oversized(paragraph: str, limit: int) -> list[str]:
    """Break one too-long paragraph on sentence boundaries where possible."""
    sentences = _SENTENCE_END.split(paragraph)
    pieces: list[str] = []
    buf = ""

    for sentence in sentences:
        if buf and len(buf) + len(sentence) + 1 > limit:
            pieces.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}".strip() if buf else sentence

    if buf:
        pieces.append(buf)

    # A single sentence longer than the limit still has to be cut somewhere.
    out: list[str] = []
    for piece in pieces:
        while len(piece) > limit:
            out.append(piece[:limit])
            piece = piece[limit:]
        if piece:
            out.append(piece)
    return out


def chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[Chunk]:
    """Pack paragraphs into chunks of at most ``chunk_size`` characters.

    Consecutive chunks share roughly ``overlap`` characters of trailing context so
    that a fact mentioned at a boundary stays retrievable from either side.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    units: list[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > chunk_size:
            units.extend(_split_oversized(paragraph, chunk_size))
        else:
            units.append(paragraph)

    chunks: list[Chunk] = []
    buf = ""
    cursor = 0

    def flush(body: str, start: int) -> int:
        chunks.append(
            Chunk(text=body, index=len(chunks), char_start=start, char_end=start + len(body))
        )
        return start + len(body)

    for unit in units:
        candidate = f"{buf}\n\n{unit}" if buf else unit
        if len(candidate) <= chunk_size:
            buf = candidate
            continue

        if buf:
            cursor = flush(buf, cursor)
            tail = buf[-overlap:] if overlap else ""
            # Resume from the overlap so the next chunk keeps prior context.
            cursor -= len(tail)
            buf = f"{tail}\n\n{unit}".strip() if tail else unit
            if len(buf) > chunk_size:
                cursor = flush(buf[:chunk_size], cursor)
                buf = buf[chunk_size:]
        else:
            buf = unit

    if buf:
        flush(buf, cursor)

    return chunks
