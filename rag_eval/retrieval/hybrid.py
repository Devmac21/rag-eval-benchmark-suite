"""Hybrid retrieval via reciprocal rank fusion (RRF)."""

from __future__ import annotations

from collections import defaultdict

from rag_eval.schemas import RetrievedHit


def rrf_fuse(
    ranked_lists: list[list[RetrievedHit]],
    k: int = 60,
    top_k: int = 10,
) -> list[RetrievedHit]:
    scores: dict[str, float] = defaultdict(float)
    texts: dict[str, str] = {}
    meta: dict[str, dict] = {}
    for lst in ranked_lists:
        for rank, hit in enumerate(lst, start=1):
            scores[hit.chunk_id] += 1.0 / (k + rank)
            texts.setdefault(hit.chunk_id, hit.text)
            meta.setdefault(hit.chunk_id, hit.metadata)
    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        RetrievedHit(chunk_id=cid, score=s, text=texts.get(cid, ""), metadata=meta.get(cid, {}))
        for cid, s in merged
    ]
