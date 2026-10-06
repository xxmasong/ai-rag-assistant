"""Embedding generation, batched and retried."""

from __future__ import annotations

import time
from typing import Protocol, Sequence

import httpx

_BATCH = 64
_MAX_RETRIES = 4


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class VoyageEmbedder:
    """Voyage AI embeddings.

    Document chunks and queries are embedded with different ``input_type`` values,
    which is what the model expects: it encodes a passage and a question into the
    same space asymmetrically, and skipping this measurably hurts recall.
    """

    _URL = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, api_key: str, *, model: str = "voyage-3", timeout: float = 30.0) -> None:
        if not api_key:
            raise ValueError("a Voyage API key is required")
        self._model = model
        self._client = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    def _post(self, texts: Sequence[str], input_type: str) -> list[list[float]]:
        payload = {"input": list(texts), "model": self._model, "input_type": input_type}

        for attempt in range(_MAX_RETRIES):
            response = self._client.post(self._URL, json=payload)
            if response.status_code == 200:
                data = response.json()["data"]
                # The API may return results out of order; index is authoritative.
                return [row["embedding"] for row in sorted(data, key=lambda r: r["index"])]

            if response.status_code in (429, 500, 502, 503, 529) and attempt < _MAX_RETRIES - 1:
                time.sleep(2**attempt)
                continue

            response.raise_for_status()

        raise RuntimeError("embedding request failed after retries")

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), _BATCH):
            out.extend(self._post(texts[i : i + _BATCH], "document"))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._post([text], "query")[0]
