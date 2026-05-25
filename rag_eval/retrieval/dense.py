"""Dense retrieval over FAISS or Qdrant."""

from __future__ import annotations

import numpy as np

from rag_eval.embeddings import EmbeddingProvider
from rag_eval.schemas import DocumentChunk, RetrievedHit
from rag_eval.vectorstores.faiss_store import FaissVectorStore
from rag_eval.vectorstores.qdrant_store import QdrantVectorStore


class DenseRetriever:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        chunks: list[DocumentChunk],
        backend: str = "faiss",
        qdrant_url: str | None = None,
        qdrant_collection: str | None = None,
        qdrant_api_key: str | None = None,
    ):
        self.embedder = embedder
        self.chunks = chunks
        self.backend = backend
        texts = [c.text for c in chunks]
        embs = self.embedder.encode_documents(texts)
        dim = embs.shape[1]
        if backend == "faiss":
            self._store: FaissVectorStore | QdrantVectorStore = FaissVectorStore(dim)
            self._store.add(embs, chunks)
        elif backend == "qdrant":
            if not qdrant_url:
                raise ValueError("qdrant_url required for Qdrant backend")
            coll = qdrant_collection or f"rag_eval_{embedder.key}"
            self._store = QdrantVectorStore(qdrant_url, coll, dim, api_key=qdrant_api_key)
            self._store.reset()
            self._store.upsert(embs, chunks)
        else:
            raise ValueError("backend must be 'faiss' or 'qdrant'")

    def search(self, query: str, top_k: int) -> list[RetrievedHit]:
        qv = self.embedder.encode_queries([query])[0]
        pairs = self._store.search(qv, top_k)
        return [
            RetrievedHit(chunk_id=ch.id, score=float(score), text=ch.text, metadata=ch.metadata)
            for ch, score in pairs
        ]
