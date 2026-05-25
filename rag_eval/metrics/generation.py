"""Generation-side metrics without mandatory LLM judge."""

from __future__ import annotations

import re
import warnings
from numbers import Real
from typing import Any, Protocol

import numpy as np

from rag_eval.schemas import GenerationSample, RetrievedHit


class GenerationJudgeProtocol(Protocol):
    def score(
        self,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> tuple[dict[str, float], dict[str, int]]:
        ...


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def token_faithfulness(answer: str, contexts: list[str]) -> float:
    """Fraction of content tokens in answer that appear in retrieved contexts."""
    a = _tokens(answer)
    if not a:
        return 1.0
    ctx = _tokens("\n".join(contexts))
    overlap = len(a & ctx)
    return overlap / len(a)


_rel_models: dict[str, Any] = {}
_st_warned: bool = False


def _rel_model_singleton(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    if model_name not in _rel_models:
        _rel_models[model_name] = SentenceTransformer(model_name)
    return _rel_models[model_name]


def _answer_relevance_lexical(answer: str, query: str) -> float:
    """Dice overlap on content tokens; used when embedding backend is unavailable."""
    a = _tokens(answer)
    q = _tokens(query)
    if not a or not q:
        return 0.0
    inter = len(a & q)
    return (2.0 * inter) / (len(a) + len(q))


def answer_relevance_embed(answer: str, query: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> float:
    global _st_warned
    try:
        m = _rel_model_singleton(model_name)
        va = m.encode([answer], normalize_embeddings=True)[0]
        vq = m.encode([query], normalize_embeddings=True)[0]
        return float(np.dot(va, vq))
    except (ImportError, OSError):
        # Missing package or DLL init failure (e.g. WinError 1114); keep benchmarking usable.
        if not _st_warned:
            warnings.warn(
                "sentence-transformers / PyTorch could not load; using lexical answer_relevance fallback. "
                "Install a working CPU torch build or fix VC++ DLLs for full embedding metrics.",
                UserWarning,
                stacklevel=2,
            )
            _st_warned = True
        return _answer_relevance_lexical(answer, query)


def context_precision(
    retrieved: list[RetrievedHit],
    relevant_ids: set[str],
    *,
    k: int | None = None,
) -> float:
    """Among top-k retrieved, fraction labeled relevant (requires chunk labels)."""
    subset = retrieved[:k] if k else retrieved
    if not subset:
        return 0.0
    rel = sum(1 for h in subset if h.chunk_id in relevant_ids)
    return rel / len(subset)


def sentence_hallucination_rate(answer: str, contexts: list[str]) -> float:
    """Share of answer sentences with no token overlap with retrieved contexts."""
    ctx_toks = _tokens("\n".join(contexts))
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    if not sents:
        return 0.0
    bad = 0
    counted = 0
    for s in sents:
        stoks = _tokens(s)
        if not stoks:
            continue
        counted += 1
        if not (stoks & ctx_toks):
            bad += 1
    return bad / counted if counted else 0.0


def enrich_generation_samples(
    samples: list[GenerationSample],
    retrieval_by_query: dict[str, list[RetrievedHit]],
) -> list[GenerationSample]:
    enriched: list[GenerationSample] = []
    for s in samples:
        hits = retrieval_by_query.get(s.query_id, [])
        ctx = list(s.contexts) if s.contexts else [h.text for h in hits]
        enriched.append(s.model_copy(update={"contexts": ctx}))
    return enriched


def aggregate_generation_metrics(sample: GenerationSample, retrieved: list[RetrievedHit], relevant_ids: set[str]) -> dict[str, float]:
    faith = token_faithfulness(sample.answer, sample.contexts)
    rel = answer_relevance_embed(sample.answer, sample.query)
    cp = context_precision(retrieved, relevant_ids, k=len(retrieved))
    hall = sentence_hallucination_rate(sample.answer, sample.contexts)
    return {
        "faithfulness": faith,
        "answer_relevance": rel,
        "context_precision": cp,
        "hallucination_rate": hall,
    }


def batch_generation_metrics(
    samples: list[GenerationSample],
    retrieval_by_query: dict[str, list[RetrievedHit]],
    relevant_by_query: dict[str, set[str]],
    *,
    judge: GenerationJudgeProtocol | None = None,
    include_nli_faithfulness: bool = False,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for s in samples:
        retr = retrieval_by_query.get(s.query_id, [])
        rel_ids = relevant_by_query.get(s.query_id, set())
        row: dict[str, Any] = dict(aggregate_generation_metrics(s, retr, rel_ids))
        if include_nli_faithfulness:
            from rag_eval.metrics.nli import nli_entailment_faithfulness

            row["faithfulness_nli"] = nli_entailment_faithfulness(s.answer, s.contexts)
        if judge is not None:
            try:
                llm_scores, usage = judge.score(s.query, s.answer, s.contexts)
                row.update(llm_scores)
                for k in usage_totals:
                    usage_totals[k] += usage.get(k, 0)
            except Exception as exc:
                row["judge_error"] = str(exc)
        rows.append(row)
    if not rows:
        out_empty: dict[str, Any] = {"per_query": [], "mean": {}}
        if judge is not None:
            out_empty["judge_token_usage"] = usage_totals
        return out_empty

    float_keys: set[str] = set()
    for r in rows:
        float_keys.update(k for k, v in r.items() if isinstance(v, Real) and not isinstance(v, bool))

    mean: dict[str, float] = {}
    for k in sorted(float_keys):
        vals = [float(r[k]) for r in rows if k in r and isinstance(r[k], Real) and not isinstance(r[k], bool)]
        if vals:
            mean[k] = float(np.mean(vals))

    out: dict[str, Any] = {"per_query": rows, "mean": mean}
    if judge is not None:
        out["judge_token_usage"] = usage_totals
    return out
