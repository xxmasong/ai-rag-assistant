"""Turn retrieved passages into a grounded answer with citations.

The contract this enforces: every claim in the answer must point at a passage that
was actually retrieved. The model is told to cite with ``[n]`` markers, and the
markers it returns are validated against the passages that were sent. A marker
that refers to nothing is dropped rather than shown, because a citation that does
not resolve is worse than no citation at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from app.services.types import Retrieved

if TYPE_CHECKING:  # The SDK is only needed for typing; importing it here keeps
    from anthropic import Anthropic  # this module testable without the package.

_CITATION = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = """\
You answer questions strictly from the numbered passages supplied by the user.

Rules:
- Use only the passages. Do not add outside knowledge.
- Cite every factual claim with the passage number in square brackets, like [2].
  A sentence drawing on several passages cites each one: [1][3].
- If the passages do not contain the answer, say so plainly and name what is
  missing. Do not guess, and do not pad the reply.
- Quote figures, dates and names exactly as they appear.
- Be concise. No preamble, no restating the question.\
"""


@dataclass
class Citation:
    marker: int
    document_id: int
    document_title: str
    chunk_index: int
    excerpt: str


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    grounded: bool = True
    usage: dict = field(default_factory=dict)


def build_context(passages: Sequence[Retrieved]) -> str:
    blocks = []
    for n, p in enumerate(passages, start=1):
        blocks.append(f"[{n}] (source: {p.document_title})\n{p.text}")
    return "\n\n".join(blocks)


class Answerer:
    def __init__(self, client: "Anthropic | Any", *, model: str) -> None:
        self._client = client
        self._model = model

    def answer(self, question: str, passages: Sequence[Retrieved]) -> Answer:
        if not passages:
            return Answer(
                text="No indexed passage matches that question, so there is nothing to answer from.",
                grounded=False,
            )

        context = build_context(passages)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Passages:\n\n{context}\n\n---\n\nQuestion: {question}",
                }
            ],
        )

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        citations, text = self._resolve_citations(text, passages)

        return Answer(
            text=text,
            citations=citations,
            grounded=bool(citations),
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    def _resolve_citations(
        self, text: str, passages: Sequence[Retrieved]
    ) -> tuple[list[Citation], str]:
        """Map ``[n]`` markers onto passages, discarding any that do not resolve."""
        seen: dict[int, Citation] = {}
        dangling: set[str] = set()

        for match in _CITATION.finditer(text):
            marker = int(match.group(1))
            if not 1 <= marker <= len(passages):
                dangling.add(match.group(0))
                continue
            if marker in seen:
                continue
            p = passages[marker - 1]
            seen[marker] = Citation(
                marker=marker,
                document_id=p.document_id,
                document_title=p.document_title,
                chunk_index=p.chunk_index,
                excerpt=p.text[:280],
            )

        for marker in dangling:
            text = text.replace(marker, "")

        return [seen[k] for k in sorted(seen)], text.strip()
