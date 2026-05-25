"""Embedding model presets for comparative benchmarking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingProvider(ABC):
    key: str

    def encode_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.encode(texts, batch_size=batch_size, is_query=False)

    def encode_queries(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self.encode(texts, batch_size=batch_size, is_query=True)

    @abstractmethod
    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        *,
        is_query: bool = False,
    ) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedding(EmbeddingProvider):
    def __init__(
        self,
        key: str,
        model_name: str,
        instruct_document: str | None = None,
        instruct_query: str | None = None,
        doc_prefix: str = "",
        query_prefix: str = "",
    ):
        self.key = key
        self.model_name = model_name
        self.instruct_document = instruct_document
        self.instruct_query = instruct_query
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer as ST

            self._model = ST(self.model_name)
        return self._model

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        *,
        is_query: bool = False,
    ) -> np.ndarray:
        if not texts:
            dim = self.model.get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)
        prefix = self.query_prefix if is_query else self.doc_prefix
        instruct = self.instruct_query if is_query else self.instruct_document
        prefixed = [f"{prefix}{t}" for t in texts]
        if instruct:
            pairs = [[instruct, t] for t in prefixed]
            emb = self.model.encode(
                pairs,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            emb = self.model.encode(
                prefixed,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(emb, dtype=np.float32)


EMBEDDING_PRESETS: dict[str, SentenceTransformerEmbedding] = {}


def register_preset(
    key: str,
    model_name: str,
    *,
    instruct_document: str | None = None,
    instruct_query: str | None = None,
    doc_prefix: str = "",
    query_prefix: str = "",
) -> None:
    EMBEDDING_PRESETS[key] = SentenceTransformerEmbedding(
        key,
        model_name,
        instruct_document=instruct_document,
        instruct_query=instruct_query,
        doc_prefix=doc_prefix,
        query_prefix=query_prefix,
    )


# Lightweight defaults suitable for CI / laptops; swap for larger checkpoints in prod benchmarks.
register_preset("bge", "BAAI/bge-small-en-v1.5")
register_preset(
    "e5",
    "intfloat/e5-small-v2",
    doc_prefix="passage: ",
    query_prefix="query: ",
)
register_preset(
    "instructor",
    "hkunlp/instructor-base",
    instruct_document="Represent the document for retrieval:",
    instruct_query="Represent the query for retrieving supporting documents:",
)
register_preset("minilm", "sentence-transformers/all-MiniLM-L6-v2")


def get_embedding_provider(key: str) -> EmbeddingProvider:
    if key not in EMBEDDING_PRESETS:
        raise ValueError(f"Unknown embedding key '{key}'. Available: {list(EMBEDDING_PRESETS)}")
    return EMBEDDING_PRESETS[key]
