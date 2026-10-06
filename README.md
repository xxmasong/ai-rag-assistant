# RAG Assistant

Document question answering over your own files, where every claim in the answer
carries a citation that resolves to the passage it came from.

The point of the project is the part most retrieval demos skip: **making the
citations trustworthy**. A language model asked to cite its sources will
cheerfully emit `[7]` when only four passages were supplied. Here the markers it
returns are validated against the passages that were actually sent, unresolvable
ones are stripped, and an answer left with no valid citation is reported as
ungrounded rather than presented as fact.

## How it works

```
upload ──▶ extract ──▶ chunk ──▶ embed ──▶ Postgres + pgvector
                                                    │
question ──▶ embed ──▶ cosine search ──▶ top-k passages
                                                    │
                                             answer + citations
                                                    │
                                        validate markers ──▶ response
```

| Stage | Choice | Why |
| --- | --- | --- |
| Chunking | paragraph-packing, 1600 chars, 200 overlap | Fixed-width splits cut sentences mid-thought, which reads badly inside a quoted citation. |
| Embeddings | Voyage `voyage-3`, asymmetric | Passages and questions are embedded with different `input_type` values, as the model expects; skipping this costs recall. |
| Vector store | Postgres + pgvector, ivfflat cosine | One datastore for rows and vectors. No second service to run or keep consistent. |
| Retrieval | top-k 8, keep 4, score floor 0.25 | Over-fetch then trim; the floor stops unrelated passages padding the prompt. |
| Answering | `claude-sonnet-5-5`, passages only | A strict system prompt plus marker validation after the fact. |

## Running it

```bash
# Postgres with pgvector
docker compose up -d db

pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY and VOYAGE_API_KEY

uvicorn app.main:app --reload
```

The schema is created on startup, so there is no separate migration step.

## API

```bash
# Index a document
curl -F file=@report.pdf -F title="FY25 report" localhost:8000/documents
# {"document_id": 1, "chunks": 42}

# Ask against it
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What was revenue in Q3?"}'
```

```json
{
  "answer": "Q3 revenue was $4.2M, up 18% year over year [1].",
  "grounded": true,
  "citations": [
    {
      "marker": 1,
      "document_id": 1,
      "document_title": "FY25 report",
      "chunk_index": 11,
      "excerpt": "Revenue for the third quarter reached $4.2M..."
    }
  ],
  "passages_considered": 4,
  "usage": {"input_tokens": 2841, "output_tokens": 37}
}
```

`GET /documents`, `DELETE /documents/{id}` and `GET /health` round out the surface.
Interactive docs are at `/docs`.

Supported uploads: `.pdf`, `.docx`, `.txt`, `.md`, `.csv`, `.json`, `.yaml`.
An unreadable type returns 415 rather than indexing bytes as mojibake.

## Tests

```bash
python -m pytest tests/ -q
```

Chunking and citation resolution are covered without a database or an API key —
the shared value types live in `app/services/types.py` precisely so the pure
logic imports cleanly on its own. The model client is stubbed in
`tests/test_answerer.py`.

## Layout

```
app/
  core/config.py        settings from the environment
  services/
    types.py            shared value types, dependency-free
    extract.py          bytes ──▶ text, dispatched on file type
    chunker.py          paragraph-aware splitting
    embedder.py         Voyage client, batched with backoff
    store.py            Postgres + pgvector persistence and search
    answerer.py         grounded answering and citation validation
  main.py               FastAPI routes
tests/
```

## Known limits

- Retrieval is dense-only. Hybrid search with BM25 would do better on exact
  identifiers and rare proper nouns.
- `rerank_keep` trims by vector score; a cross-encoder reranker would order the
  surviving passages more faithfully.
- The ivfflat `lists = 100` figure suits tens of thousands of chunks. Rebuild the
  index against real row counts beyond that.
- No authentication. It is meant to sit behind a gateway, not on the open internet.
