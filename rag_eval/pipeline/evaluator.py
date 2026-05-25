"""Orchestrates retrieval modes, metrics, latency, and artifact emission."""

from __future__ import annotations

import time
from typing import Any, Literal

try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore

from rag_eval.config import get_settings
from rag_eval.latency import LatencySummary, timed
from rag_eval.metrics.generation import batch_generation_metrics, enrich_generation_samples
from rag_eval.metrics.retrieval import retrieval_aggregate
from rag_eval.retrieval.bm25 import BM25Index
from rag_eval.retrieval.hybrid import rrf_fuse
from rag_eval.schemas import BenchmarkPack, DocumentChunk, GenerationSample, QAExample, RetrievalResult, RetrievedHit


RetrievalMode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]


def _approx_tokens(text: str) -> int:
    if tiktoken:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


class RAGEvaluator:
    def __init__(
        self,
        *,
        embedding_key: str = "minilm",
        dense_backend: Literal["faiss", "qdrant"] = "faiss",
        top_k: int = 10,
    ):
        self.embedding_key = embedding_key
        self.dense_backend = dense_backend
        self.top_k = top_k
        self.settings = get_settings()

    def run_retrieval(
        self,
        pack: BenchmarkPack,
        mode: RetrievalMode,
        *,
        latency: LatencySummary | None = None,
        rerank_top_n: int | None = None,
    ) -> tuple[list[RetrievalResult], dict[str, Any]]:
        chunks = pack.chunks
        bm25 = BM25Index(chunks)
        dense = None
        reranker = None

        if mode in ("dense", "hybrid", "hybrid_rerank"):
            from rag_eval.embeddings import get_embedding_provider
            from rag_eval.retrieval.dense import DenseRetriever

            embedder = get_embedding_provider(self.embedding_key)

            def build_dense():
                return DenseRetriever(
                    embedder,
                    chunks,
                    backend=self.dense_backend,
                    qdrant_url=self.settings.qdrant_url,
                    qdrant_collection=f"{pack.corpus_id}_{self.embedding_key}",
                    qdrant_api_key=self.settings.qdrant_api_key,
                )

            if latency:
                dense = timed("index_dense", latency, build_dense)
            else:
                dense = build_dense()

        if mode == "hybrid_rerank":
            from rag_eval.retrieval.rerank import CrossEncoderReranker

            reranker = CrossEncoderReranker()

        results: list[RetrievalResult] = []
        token_counts: dict[str, int] = {"queries": 0, "contexts": 0}

        def retrieve_one(ex) -> list[RetrievedHit]:
            token_counts["queries"] += _approx_tokens(ex.query)
            if mode == "bm25":
                return bm25.search(ex.query, self.top_k)
            if mode == "dense" and dense:
                return dense.search(ex.query, self.top_k)
            if mode == "hybrid" and dense:
                lex = bm25.search(ex.query, self.top_k)
                vec = dense.search(ex.query, self.top_k)
                return rrf_fuse([lex, vec], top_k=self.top_k)
            if mode == "hybrid_rerank" and dense:
                lex = bm25.search(ex.query, self.top_k * 2)
                vec = dense.search(ex.query, self.top_k * 2)
                fused = rrf_fuse([lex, vec], top_k=rerank_top_n or self.top_k * 2)
                assert reranker is not None
                return reranker.rerank(ex.query, fused, top_k=self.top_k)
            raise RuntimeError(f"Unhandled mode {mode}")

        for ex in pack.examples:
            if latency:
                hits = timed("retrieve_query", latency, lambda: retrieve_one(ex))
            else:
                hits = retrieve_one(ex)
            results.append(RetrievalResult(query_id=ex.query_id, hits=hits))
            token_counts["contexts"] += sum(_approx_tokens(h.text) for h in hits)

        rel_map = {ex.query_id: set(ex.relevant_chunk_ids) for ex in pack.examples}
        metrics = retrieval_aggregate(results, rel_map)
        aux = {"token_usage_estimate": token_counts, "mode": mode, "embedding": self.embedding_key}
        return results, {"retrieval_metrics": metrics, **aux}

    def run_generation_benchmark(
        self,
        pack: BenchmarkPack,
        samples: list[GenerationSample],
        retrieval_results: list[RetrievalResult],
        *,
        use_llm_judge: bool = False,
        use_nli_faithfulness: bool = False,
    ) -> dict[str, Any]:
        by_q = {r.query_id: r.hits for r in retrieval_results}
        rel_map = {ex.query_id: set(ex.relevant_chunk_ids) for ex in pack.examples}
        enriched = enrich_generation_samples(samples, by_q)
        judge = None
        if use_llm_judge:
            from rag_eval.judge import OpenAICompatibleJudge

            judge = OpenAICompatibleJudge.from_settings(self.settings)
        gen = batch_generation_metrics(
            enriched,
            by_q,
            rel_map,
            judge=judge,
            include_nli_faithfulness=use_nli_faithfulness,
        )
        return {"generation_metrics": gen}

    def full_benchmark(
        self,
        pack: BenchmarkPack,
        samples: list[GenerationSample],
        mode: RetrievalMode,
        *,
        use_llm_judge: bool = False,
        use_nli_faithfulness: bool = False,
    ) -> dict[str, Any]:
        latency = LatencySummary()
        t0 = time.perf_counter()
        retr, retr_payload = self.run_retrieval(pack, mode, latency=latency)
        retr_wall_ms = (time.perf_counter() - t0) * 1000
        gen_payload = self.run_generation_benchmark(
            pack,
            samples,
            retr,
            use_llm_judge=use_llm_judge,
            use_nli_faithfulness=use_nli_faithfulness,
        )
        return {
            **retr_payload,
            **gen_payload,
            "latency_ms": {"wall_clock_total": retr_wall_ms, **latency.aggregate()},
        }


def compare_embeddings(pack: BenchmarkPack, mode: RetrievalMode, keys: list[str]) -> dict[str, Any]:
    if mode == "bm25":
        raise ValueError("Embedding comparison requires a dense-capable mode (dense, hybrid, hybrid_rerank).")
    rows = []
    for key in keys:
        ev = RAGEvaluator(embedding_key=key)
        _, payload = ev.run_retrieval(pack, mode)
        rows.append({"embedding": key, **payload["retrieval_metrics"], **payload["token_usage_estimate"]})
    return {"embedding_comparison": rows}


def compare_chunk_strategies(
    raw_documents: list[tuple[str, str]],
    pack_builder,
    mode: RetrievalMode,
    embedding_key: str,
    *,
    dense_backend: Literal["faiss", "qdrant"] = "faiss",
    top_k: int = 10,
) -> dict[str, Any]:
    """Compare chunking strategies given raw (doc_id, text) pairs and a pack_builder."""
    from rag_eval.chunking import STRATEGIES

    out = []
    for name in sorted(STRATEGIES):
        strat_cls = STRATEGIES[name]
        strategy = strat_cls()
        chunks = []
        for doc_id, text in raw_documents:
            chunks.extend(strategy.chunk(doc_id, text))
        pack = pack_builder(chunks)
        ev = RAGEvaluator(embedding_key=embedding_key, dense_backend=dense_backend, top_k=top_k)
        _, payload = ev.run_retrieval(pack, mode)
        out.append({"chunking": name, **payload["retrieval_metrics"]})
    return {"chunking_comparison": out}


def retrieval_overlap_matrix(results_a: list[RetrievalResult], results_b: list[RetrievalResult], k: int = 5) -> float:
    """Mean Jaccard overlap of top-k chunk ids between two retrieval configurations."""
    by_id_a = {r.query_id: {h.chunk_id for h in r.hits[:k]} for r in results_a}
    scores = []
    for r in results_b:
        sa = by_id_a.get(r.query_id, set())
        sb = {h.chunk_id for h in r.hits[:k]}
        if not sa and not sb:
            continue
        inter = len(sa & sb)
        union = len(sa | sb) or 1
        scores.append(inter / union)
    return sum(scores) / len(scores) if scores else 0.0


def compare_chunk_strategies_from_documents(
    documents: list[tuple[str, str]],
    examples: list[tuple[str, str, str]],
    mode: RetrievalMode,
    embedding_key: str,
    *,
    dense_backend: Literal["faiss", "qdrant"] = "faiss",
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Compare chunking strategies using document-level labels.

    ``examples`` entries are ``(query_id, query, gold_document_id)``. Every chunk whose
    metadata ``source_doc_id`` equals ``gold_document_id`` is treated as relevant.
    """

    def pack_builder(chunks: list[DocumentChunk]) -> BenchmarkPack:
        doc_to_chunks: dict[str, list[str]] = {}
        for ch in chunks:
            sid = str(ch.metadata.get("source_doc_id") or "")
            if sid:
                doc_to_chunks.setdefault(sid, []).append(ch.id)
        qa_examples: list[QAExample] = []
        for qid, query, gold_doc in examples:
            rel_ids = list(dict.fromkeys(doc_to_chunks.get(gold_doc, [])))
            qa_examples.append(QAExample(query_id=qid, query=query, relevant_chunk_ids=rel_ids))
        return BenchmarkPack(corpus_id="chunk_strategy_compare", chunks=chunks, examples=qa_examples)

    return compare_chunk_strategies(
        documents,
        pack_builder,
        mode,
        embedding_key,
        dense_backend=dense_backend,
        top_k=top_k,
    )
