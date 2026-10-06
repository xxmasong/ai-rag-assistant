"""FastAPI entrypoint: ingest documents, then ask questions against them."""

from __future__ import annotations

from contextlib import asynccontextmanager

from anthropic import Anthropic
from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.services.answerer import Answerer
from app.services.chunker import chunk_text
from app.services.embedder import VoyageEmbedder
from app.services.extract import extract_text
from app.services.store import Store


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    store = Store(settings.database_url)
    store.migrate()
    app.state.store = store
    app.state.embedder = VoyageEmbedder(settings.voyage_api_key, model=settings.embedding_model)
    app.state.answerer = Answerer(
        Anthropic(api_key=settings.anthropic_api_key), model=settings.answer_model
    )
    yield


app = FastAPI(
    title="RAG Assistant",
    description="Document question answering with verifiable citations.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)


class CitationOut(BaseModel):
    marker: int
    document_id: int
    document_title: str
    chunk_index: int
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    citations: list[CitationOut]
    passages_considered: int
    usage: dict


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/documents")
def list_documents() -> list[dict]:
    return app.state.store.list_documents()


@app.post("/documents", status_code=201)
async def ingest(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    settings: Settings = Depends(get_settings),
) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "the uploaded file is empty")
    if len(raw) > settings.max_upload_bytes:
        raise HTTPException(
            413, f"file exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit"
        )

    try:
        text = extract_text(raw, filename=file.filename or "upload")
    except ValueError as exc:
        raise HTTPException(415, str(exc)) from exc

    chunks = chunk_text(text, chunk_size=settings.chunk_size, overlap=settings.chunk_overlap)
    if not chunks:
        raise HTTPException(422, "no extractable text found in the file")

    embeddings = app.state.embedder.embed_documents([c.text for c in chunks])

    document_id = app.state.store.add_document(
        title=title or file.filename or "untitled",
        source=file.filename,
        metadata={"bytes": len(raw), "content_type": file.content_type},
        chunks=[(c.text, c.index, c.char_start, c.char_end) for c in chunks],
        embeddings=embeddings,
    )

    return {"document_id": document_id, "chunks": len(chunks)}


@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: int) -> None:
    if not app.state.store.delete_document(document_id):
        raise HTTPException(404, "no such document")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, settings: Settings = Depends(get_settings)) -> AskResponse:
    query_vector = app.state.embedder.embed_query(request.question)

    passages = app.state.store.search(
        query_vector,
        top_k=request.top_k or settings.top_k,
        min_score=settings.min_score,
    )[: settings.rerank_keep]

    result = app.state.answerer.answer(request.question, passages)

    return AskResponse(
        answer=result.text,
        grounded=result.grounded,
        citations=[CitationOut(**vars(c)) for c in result.citations],
        passages_considered=len(passages),
        usage=result.usage,
    )
