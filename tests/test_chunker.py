from app.services.chunker import chunk_text


def test_empty_input_produces_no_chunks():
    assert chunk_text("", chunk_size=100, overlap=10) == []
    assert chunk_text("   \n\n  ", chunk_size=100, overlap=10) == []


def test_short_text_is_a_single_chunk():
    chunks = chunk_text("One short paragraph.", chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "One short paragraph."
    assert chunks[0].index == 0


def test_paragraphs_are_packed_up_to_the_limit():
    text = "\n\n".join(["aaa", "bbb", "ccc"])
    chunks = chunk_text(text, chunk_size=100, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].text == "aaa\n\nbbb\n\nccc"


def test_no_chunk_exceeds_the_size_limit():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(12))
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 for c in chunks)


def test_oversized_paragraph_is_split_on_sentences():
    paragraph = " ".join(f"Sentence number {i} runs on for a while." for i in range(40))
    chunks = chunk_text(paragraph, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)


def test_chunk_indices_are_sequential():
    text = "\n\n".join("word " * 60 for _ in range(8))
    chunks = chunk_text(text, chunk_size=250, overlap=40)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_overlap_carries_context_between_chunks():
    text = "\n\n".join(f"Block {i} " + "filler " * 30 for i in range(6))
    chunks = chunk_text(text, chunk_size=260, overlap=60)
    # The tail of one chunk should reappear at the head of the next.
    assert len(chunks) > 1
    assert any(chunks[i].text[-20:] in chunks[i + 1].text for i in range(len(chunks) - 1))


def test_invalid_parameters_are_rejected():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=100, overlap=-1)
