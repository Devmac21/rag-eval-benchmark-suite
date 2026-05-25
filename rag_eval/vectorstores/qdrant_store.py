"""Optional Qdrant-backed dense retrieval."""

from __future__ import annotations

import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from rag_eval.schemas import DocumentChunk


class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        collection_name: str,
        dim: int,
        api_key: str | None = None,
    ):
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection = collection_name
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        exists = self.client.collection_exists(self.collection)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            )

    def reset(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self._ensure_collection()

    def upsert(self, embeddings: np.ndarray, chunks: list[DocumentChunk]) -> None:
        points = []
        for emb, ch in zip(embeddings, chunks):
            pid = abs(hash(ch.id)) % (2**63 - 1)
            points.append(
                qm.PointStruct(
                    id=pid,
                    vector=np.asarray(emb, dtype=np.float32).tolist(),
                    payload={"chunk_id": ch.id, "text": ch.text, "meta": ch.metadata},
                )
            )
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def search(self, query_emb: np.ndarray, top_k: int) -> list[tuple[DocumentChunk, float]]:
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=np.asarray(query_emb, dtype=np.float32).tolist(),
            limit=top_k,
            with_payload=True,
        )
        out: list[tuple[DocumentChunk, float]] = []
        for h in hits:
            pl = h.payload or {}
            ch = DocumentChunk(
                id=str(pl.get("chunk_id", uuid.uuid4())),
                text=str(pl.get("text", "")),
                metadata=dict(pl.get("meta") or {}),
            )
            out.append((ch, float(h.score)))
        return out
