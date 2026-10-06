"""Citation resolution tests, run against a stub client so no API key is needed."""

from dataclasses import dataclass

from app.services.answerer import Answerer, build_context
from app.services.types import Retrieved


@dataclass
class _Block:
    text: str
    type: str = "text"


@dataclass
class _Usage:
    input_tokens: int = 10
    output_tokens: int = 20


@dataclass
class _Response:
    content: list
    usage: _Usage


class _StubClient:
    """Stands in for Anthropic(), returning a canned completion."""

    def __init__(self, reply: str) -> None:
        self._reply = reply
        self.messages = self

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _Response(content=[_Block(self._reply)], usage=_Usage())


def _passages(n: int = 3) -> list[Retrieved]:
    return [
        Retrieved(
            chunk_id=i,
            document_id=100 + i,
            document_title=f"Doc {i}",
            chunk_index=0,
            text=f"Body of passage {i}.",
            score=0.9 - i * 0.1,
        )
        for i in range(1, n + 1)
    ]


def test_no_passages_returns_ungrounded_answer():
    answerer = Answerer(_StubClient("unused"), model="test-model")
    result = answerer.answer("What is the revenue?", [])
    assert result.grounded is False
    assert result.citations == []
    assert "nothing to answer from" in result.text


def test_valid_markers_resolve_to_passages():
    answerer = Answerer(_StubClient("Revenue grew [1] and margin held [3]."), model="test-model")
    result = answerer.answer("How did revenue do?", _passages())

    assert result.grounded is True
    assert [c.marker for c in result.citations] == [1, 3]
    assert result.citations[0].document_title == "Doc 1"
    assert result.citations[1].document_id == 103


def test_dangling_markers_are_stripped():
    answerer = Answerer(_StubClient("Grounded claim [1] and invented one [9]."), model="test-model")
    result = answerer.answer("q", _passages())

    assert [c.marker for c in result.citations] == [1]
    assert "[9]" not in result.text
    assert "[1]" in result.text


def test_repeated_markers_are_deduplicated():
    answerer = Answerer(_StubClient("A [2] then B [2] then C [2]."), model="test-model")
    result = answerer.answer("q", _passages())
    assert [c.marker for c in result.citations] == [2]


def test_answer_with_only_bad_markers_is_not_grounded():
    answerer = Answerer(_StubClient("Unsupported assertion [7]."), model="test-model")
    result = answerer.answer("q", _passages())
    assert result.grounded is False
    assert result.citations == []


def test_usage_is_reported():
    answerer = Answerer(_StubClient("Fine [1]."), model="test-model")
    result = answerer.answer("q", _passages())
    assert result.usage == {"input_tokens": 10, "output_tokens": 20}


def test_context_is_numbered_from_one():
    context = build_context(_passages(2))
    assert context.startswith("[1] (source: Doc 1)")
    assert "[2] (source: Doc 2)" in context
    assert "[0]" not in context
