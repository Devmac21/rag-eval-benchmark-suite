"""Cross-encoder reranking."""

from __future__ import annotations

from typing import Any

from rag_eval.schemas import RetrievedHit


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, hits: list[RetrievedHit], top_k: int | None = None) -> list[RetrievedHit]:
        if not hits:
            return []
        pairs = [(query, h.text or "") for h in hits]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        out = [RetrievedHit(chunk_id=h.chunk_id, score=float(s), text=h.text, metadata=h.metadata) for h, s in ranked]
        if top_k is not None:
            return out[:top_k]
        return out
