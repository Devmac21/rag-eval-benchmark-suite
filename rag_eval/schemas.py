from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Single chunk with stable id for overlap / attribution."""

    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QAExample(BaseModel):
    """One benchmark item with optional relevance labels per chunk id."""

    query_id: str
    query: str
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    gold_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkPack(BaseModel):
    """Corpus + questions for an evaluation run."""

    corpus_id: str = "default"
    chunks: list[DocumentChunk]
    examples: list[QAExample]


class RetrievedHit(BaseModel):
    chunk_id: str
    score: float
    text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query_id: str
    hits: list[RetrievedHit]


class GenerationSample(BaseModel):
    query_id: str
    query: str
    answer: str
    contexts: list[str] = Field(default_factory=list)
    gold_answer: str | None = None
