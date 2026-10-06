"""Postgres + pgvector persistence for documents and their embedded chunks."""

from __future__ import annotations

from typing import Sequence

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from app.services.types import Retrieved

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    title       TEXT        NOT NULL,
    source      TEXT,
    metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT         NOT NULL,
    text         TEXT        NOT NULL,
    char_start   INT         NOT NULL,
    char_end     INT         NOT NULL,
    embedding    VECTOR(1024),
    UNIQUE (document_id, chunk_index)
);

-- Cosine distance index. Build it after the first bulk load so the lists
-- parameter is chosen against real row counts.
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);
"""



class Store:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._dsn, row_factory=dict_row)

    def migrate(self) -> None:
        with self._connect() as conn:
            conn.execute(SCHEMA)
            conn.commit()

    def add_document(
        self,
        *,
        title: str,
        source: str | None,
        metadata: dict,
        chunks: Sequence[tuple[str, int, int, int]],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Insert a document and its chunks in one transaction.

        ``chunks`` holds ``(text, chunk_index, char_start, char_end)`` tuples,
        positionally aligned with ``embeddings``.
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO documents (title, source, metadata) VALUES (%s, %s, %s) RETURNING id",
                    (title, source, Jsonb(metadata)),
                )
                document_id = cur.fetchone()["id"]

                cur.executemany(
                    """
                    INSERT INTO chunks
                        (document_id, chunk_index, text, char_start, char_end, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (document_id, idx, text, start, end, list(vector))
                        for (text, idx, start, end), vector in zip(chunks, embeddings)
                    ],
                )
            conn.commit()
        return document_id

    def search(
        self, embedding: Sequence[float], *, top_k: int, min_score: float
    ) -> list[Retrieved]:
        """Return the nearest chunks by cosine similarity, best first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.id,
                       c.document_id,
                       d.title       AS document_title,
                       c.chunk_index,
                       c.text,
                       1 - (c.embedding <=> %s::vector) AS score
                  FROM chunks c
                  JOIN documents d ON d.id = c.document_id
                 WHERE c.embedding IS NOT NULL
                 ORDER BY c.embedding <=> %s::vector
                 LIMIT %s
                """,
                (list(embedding), list(embedding), top_k),
            ).fetchall()

        return [
            Retrieved(
                chunk_id=r["id"],
                document_id=r["document_id"],
                document_title=r["document_title"],
                chunk_index=r["chunk_index"],
                text=r["text"],
                score=float(r["score"]),
            )
            for r in rows
            if float(r["score"]) >= min_score
        ]

    def delete_document(self, document_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            conn.commit()
            return cur.rowcount > 0

    def list_documents(self) -> list[dict]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT d.id, d.title, d.source, d.created_at, count(c.id) AS chunk_count
                  FROM documents d
                  LEFT JOIN chunks c ON c.document_id = d.id
                 GROUP BY d.id
                 ORDER BY d.created_at DESC
                """
            ).fetchall()
