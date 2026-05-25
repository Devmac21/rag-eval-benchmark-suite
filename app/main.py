"""FastAPI service exposing benchmark operations."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.deps import verify_optional_api_key
from rag_eval.schemas import BenchmarkPack, GenerationSample, RetrievalResult

_PROTECTED = [Depends(verify_optional_api_key)]


class RetrievalEvalRequest(BaseModel):
    pack: BenchmarkPack
    mode: str = Field(description="bm25 | dense | hybrid | hybrid_rerank")
    embedding_key: str = "minilm"
    dense_backend: str = "faiss"
    top_k: int = 10


class GenerationEvalRequest(BaseModel):
    pack: BenchmarkPack
    samples: list[GenerationSample]
    retrieval_results: list[RetrievalResult]
    use_llm_judge: bool = False
    use_nli_faithfulness: bool = False


class FullBenchmarkRequest(BaseModel):
    pack: BenchmarkPack
    samples: list[GenerationSample]
    mode: str = "hybrid"
    embedding_key: str = "minilm"
    dense_backend: str = "faiss"
    top_k: int = 10
    persist: bool = True
    use_llm_judge: bool = False
    use_nli_faithfulness: bool = False


class EmbeddingCompareRequest(BaseModel):
    pack: BenchmarkPack
    mode: str = "dense"
    embedding_keys: list[str] = Field(default_factory=lambda: ["bge", "e5", "instructor", "minilm"])


class RawDoc(BaseModel):
    doc_id: str
    text: str


class ChunkCompareGold(BaseModel):
    query_id: str
    query: str
    gold_document_id: str


class ChunkCompareRequest(BaseModel):
    documents: list[RawDoc]
    examples: list[ChunkCompareGold]
    mode: str = "bm25"
    embedding_key: str = "minilm"
    dense_backend: str = "faiss"
    top_k: int = 10


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Eval Benchmark Suite", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/evaluate/retrieval", dependencies=_PROTECTED)
    def evaluate_retrieval(body: RetrievalEvalRequest):
        from rag_eval.pipeline.evaluator import RAGEvaluator

        ev = RAGEvaluator(
            embedding_key=body.embedding_key,
            dense_backend=body.dense_backend,  # type: ignore[arg-type]
            top_k=body.top_k,
        )
        results, payload = ev.run_retrieval(body.pack, body.mode)  # type: ignore[arg-type]
        return {"hits": [r.model_dump() for r in results], **payload}

    @app.post("/evaluate/generation", dependencies=_PROTECTED)
    def evaluate_generation(body: GenerationEvalRequest):
        from rag_eval.judge import JudgeConfigurationError, OpenAICompatibleJudge
        from rag_eval.metrics.generation import batch_generation_metrics, enrich_generation_samples

        by_q = {r.query_id: r.hits for r in body.retrieval_results}
        rel_map = {ex.query_id: set(ex.relevant_chunk_ids) for ex in body.pack.examples}
        try:
            judge = OpenAICompatibleJudge.from_settings() if body.use_llm_judge else None
        except JudgeConfigurationError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        enriched = enrich_generation_samples(body.samples, by_q)
        return batch_generation_metrics(
            enriched,
            by_q,
            rel_map,
            judge=judge,
            include_nli_faithfulness=body.use_nli_faithfulness,
        )

    @app.post("/benchmark/full", dependencies=_PROTECTED)
    def benchmark_full(body: FullBenchmarkRequest):
        from rag_eval.experiments.tracker import ExperimentTracker
        from rag_eval.judge import JudgeConfigurationError
        from rag_eval.pipeline.evaluator import RAGEvaluator
        from rag_eval.reports.generator import write_benchmark_report
        from rag_eval.viz.charts import render_dashboard_bundle

        ev = RAGEvaluator(
            embedding_key=body.embedding_key,
            dense_backend=body.dense_backend,  # type: ignore[arg-type]
            top_k=body.top_k,
        )
        try:
            payload = ev.full_benchmark(
                body.pack,
                body.samples,
                body.mode,
                use_llm_judge=body.use_llm_judge,
                use_nli_faithfulness=body.use_nli_faithfulness,
            )
        except JudgeConfigurationError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        tracker = ExperimentTracker()
        run_payload = {
            "config": body.model_dump(),
            **payload,
        }
        if body.persist:
            saved = tracker.save_run(run_payload, name=body.mode)
            rid = saved["run_id"]
            payload_with_id = {**payload, "run_id": rid}
            report_path = write_benchmark_report(payload_with_id)
            dash_paths = render_dashboard_bundle(payload_with_id, rid)
            run_payload["artifacts"] = {
                "run_json": str(saved["path"]),
                "report_md": str(report_path),
                **dash_paths,
            }
            run_payload["run_id"] = rid
        return run_payload

    @app.post("/benchmark/embeddings", dependencies=_PROTECTED)
    def benchmark_embeddings(body: EmbeddingCompareRequest):
        import matplotlib

        matplotlib.use("Agg")

        from rag_eval.pipeline.evaluator import (
            RAGEvaluator,
            compare_embeddings,
            retrieval_overlap_matrix,
        )
        from rag_eval.viz.charts import plot_overlap_heatmap

        comp = compare_embeddings(body.pack, body.mode, body.embedding_keys)  # type: ignore[arg-type]
        keys = body.embedding_keys
        matrices = []
        overlap_png = None
        if len(keys) > 1:
            all_results = []
            for key in keys:
                ev = RAGEvaluator(embedding_key=key)
                res, _ = ev.run_retrieval(body.pack, body.mode)  # type: ignore[arg-type]
                all_results.append(res)
            results_matrix: list[list[float]] = []
            for i in range(len(keys)):
                row = []
                for j in range(len(keys)):
                    row.append(retrieval_overlap_matrix(all_results[i], all_results[j], k=5))
                results_matrix.append(row)
            matrices = results_matrix
            reports_dir = Path(os.environ.get("RAG_EVAL_REPORTS_DIR", "./reports"))
            overlap_png = str(
                plot_overlap_heatmap(matrices, keys, reports_dir / "figures" / "embedding_overlap.png")
            )
        return {**comp, "overlap_matrix": matrices, "overlap_chart": overlap_png}

    @app.post("/benchmark/chunking", dependencies=_PROTECTED)
    def benchmark_chunking(body: ChunkCompareRequest):
        from rag_eval.pipeline.evaluator import compare_chunk_strategies_from_documents

        docs = [(d.doc_id, d.text) for d in body.documents]
        gold = [(e.query_id, e.query, e.gold_document_id) for e in body.examples]
        return compare_chunk_strategies_from_documents(
            docs,
            gold,
            body.mode,
            body.embedding_key,
            dense_backend=body.dense_backend,  # type: ignore[arg-type]
            top_k=body.top_k,
        )

    @app.get("/experiments", dependencies=_PROTECTED)
    def list_experiments():
        from rag_eval.experiments.tracker import ExperimentTracker

        return ExperimentTracker().list_runs()

    @app.get("/experiments/{run_id}", dependencies=_PROTECTED)
    def get_experiment(run_id: str):
        from rag_eval.experiments.tracker import ExperimentTracker

        try:
            return ExperimentTracker().get_run(run_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found") from None

    @app.post("/visualize", dependencies=_PROTECTED)
    def visualize(run_payload: dict):
        import matplotlib

        matplotlib.use("Agg")

        from rag_eval.viz.charts import render_dashboard_bundle

        rid = str(run_payload.get("run_id", "adhoc"))
        paths = render_dashboard_bundle(run_payload, rid)
        return {"figures": paths}

    return app


app = create_app()
