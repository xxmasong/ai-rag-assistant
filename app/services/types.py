"""Shared value types.

These live apart from the persistence and model clients so that the pure logic
(chunking, citation resolution) can be imported and tested without a database
driver or an API SDK installed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    index: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class Retrieved:
    chunk_id: int
    document_id: int
    document_title: str
    chunk_index: int
    text: str
    score: float
