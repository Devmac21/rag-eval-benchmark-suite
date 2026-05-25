"""Retrieval metrics: Recall@K, MRR, NDCG."""

from __future__ import annotations

import math
from typing import Iterable

from rag_eval.schemas import RetrievalResult


def recall_at_k(results: list[RetrievalResult], relevant_map: dict[str, set[str]], k: int) -> float:
    """Fraction of queries with ≥1 relevant chunk in top-k."""
    if not results:
        return 0.0
    hits = 0
    for r in results:
        rel = relevant_map.get(r.query_id, set())
        if not rel:
            continue
        top_ids = {h.chunk_id for h in r.hits[:k]}
        if top_ids & rel:
            hits += 1
    denom = sum(1 for r in results if relevant_map.get(r.query_id))
    return hits / denom if denom else 0.0


def mrr(results: list[RetrievalResult], relevant_map: dict[str, set[str]]) -> float:
    recip = []
    for r in results:
        rel = relevant_map.get(r.query_id, set())
        if not rel:
            continue
        rank = None
        for i, h in enumerate(r.hits, start=1):
            if h.chunk_id in rel:
                rank = i
                break
        recip.append(1.0 / rank if rank else 0.0)
    return sum(recip) / len(recip) if recip else 0.0


def _dcg(relevances: list[float]) -> float:
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(
    results: list[RetrievalResult],
    relevant_map: dict[str, set[str]],
    k: int,
) -> float:
    """NDCG@k with binary relevance (1 if chunk id in relevant set else 0)."""
    scores = []
    for r in results:
        rel = relevant_map.get(r.query_id, set())
        if not rel:
            continue
        rel_vec = [1.0 if h.chunk_id in rel else 0.0 for h in r.hits[:k]]
        ideal_vec = sorted(rel_vec, reverse=True)
        dcg = _dcg(rel_vec)
        idcg = _dcg(ideal_vec)
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def retrieval_aggregate(
    results: list[RetrievalResult],
    relevant_map: dict[str, set[str]],
    ks: Iterable[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    out: dict[str, float] = {"mrr": mrr(results, relevant_map)}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(results, relevant_map, k)
        out[f"ndcg@{k}"] = ndcg_at_k(results, relevant_map, k)
    return out
