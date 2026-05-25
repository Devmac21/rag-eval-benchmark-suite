"""BM25 lexical retrieval."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from rag_eval.schemas import DocumentChunk, RetrievedHit


def tokenize(text: str) -> list[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if t]


class BM25Index:
    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = chunks
        self._corpus_tokens = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, top_k: int) -> list[RetrievedHit]:
        q_tokens = tokenize(query)
        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            RetrievedHit(chunk_id=self.chunks[i].id, score=float(s), text=self.chunks[i].text)
            for i, s in ranked
        ]
