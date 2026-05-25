"""Hallucination-oriented analysis built on lexical and embedding signals."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rag_eval.metrics.generation import sentence_hallucination_rate, token_faithfulness


@dataclass
class HallucinationReport:
    query_id: str
    faithfulness: float
    hallucination_rate: float
    max_claim_similarity: float
    unsupported_claims: list[str]


def split_claims(answer: str) -> list[str]:
    parts = [p.strip() for p in answer.replace("\n", " ").split(".") if p.strip()]
    return [p + "." for p in parts]


def analyze_answer(
    query_id: str,
    answer: str,
    contexts: list[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> HallucinationReport:
    faith = token_faithfulness(answer, contexts)
    hall_rate = sentence_hallucination_rate(answer, contexts)
    claims = split_claims(answer)
    ctx_blob = "\n".join(contexts)
    unsupported: list[str] = []
    sims: list[float] = []
    if claims and contexts:
        try:
            from sentence_transformers import SentenceTransformer

            m = SentenceTransformer(model_name)
            ctx_emb = m.encode([ctx_blob], normalize_embeddings=True)[0]
            for c in claims:
                ce = m.encode([c], normalize_embeddings=True)[0]
                s = float(np.dot(ce, ctx_emb))
                sims.append(s)
                if s < 0.35:
                    unsupported.append(c)
        except (ImportError, OSError):
            sims = [1.0]
    else:
        sims = [1.0]
    return HallucinationReport(
        query_id=query_id,
        faithfulness=faith,
        hallucination_rate=hall_rate,
        max_claim_similarity=max(sims) if sims else 0.0,
        unsupported_claims=unsupported,
    )
