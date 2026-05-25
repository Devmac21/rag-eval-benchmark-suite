"""In-memory FAISS vector index."""

from __future__ import annotations

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("faiss-cpu is required for FAISS backend") from e

from rag_eval.schemas import DocumentChunk


class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: list[DocumentChunk] = []

    def add(self, embeddings: np.ndarray, chunks: list[DocumentChunk]) -> None:
        if embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings and chunks length mismatch")
        self._chunks.extend(chunks)
        self._index.add(np.asarray(embeddings, dtype=np.float32))

    def search(self, query_emb: np.ndarray, top_k: int) -> list[tuple[DocumentChunk, float]]:
        q = np.asarray(query_emb, dtype=np.float32).reshape(1, -1)
        scores, idxs = self._index.search(q, min(top_k, len(self._chunks)))
        out: list[tuple[DocumentChunk, float]] = []
        for i, s in zip(idxs[0], scores[0]):
            if i < 0:
                continue
            out.append((self._chunks[i], float(s)))
        return out

    @property
    def size(self) -> int:
        return len(self._chunks)
