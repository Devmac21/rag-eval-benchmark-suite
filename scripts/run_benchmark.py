"""CLI entrypoint for local benchmarking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_eval.experiments.tracker import ExperimentTracker
from rag_eval.judge import JudgeConfigurationError
from rag_eval.io import load_benchmark_pack
from rag_eval.pipeline.evaluator import RAGEvaluator, compare_embeddings
from rag_eval.reports.generator import write_benchmark_report
from rag_eval.schemas import GenerationSample
from rag_eval.viz.charts import render_dashboard_bundle


def main() -> None:
    p = argparse.ArgumentParser(description="Run RAG retrieval/generation benchmarks.")
    p.add_argument("--pack", type=Path, default=Path("data/sample_benchmark.json"))
    p.add_argument("--mode", default="bm25", choices=["bm25", "dense", "hybrid", "hybrid_rerank"])
    p.add_argument("--embedding", default="minilm")
    p.add_argument("--backend", default="faiss", choices=["faiss", "qdrant"])
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--samples", type=Path, default=None, help="Optional JSON list of GenerationSample")
    p.add_argument("--compare-embeddings", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument(
        "--llm-judge",
        action="store_true",
        help="Call OpenAI-compatible LLM judge (configure RAG_EVAL_OPENAI_COMPATIBLE_* env vars).",
    )
    p.add_argument(
        "--nli-faithfulness",
        action="store_true",
        help="Add faithfulness_nli via cross-encoder NLI (downloads model on first use).",
    )
    args = p.parse_args()

    if args.compare_embeddings and args.mode == "bm25":
        p.error("--compare-embeddings requires --mode dense, hybrid, or hybrid_rerank")

    pack = load_benchmark_pack(args.pack)
    samples: list[GenerationSample] = []
    if args.samples:
        raw = json.loads(args.samples.read_text(encoding="utf-8"))
        samples = [GenerationSample.model_validate(x) for x in raw]
    else:
        for ex in pack.examples:
            samples.append(
                GenerationSample(
                    query_id=ex.query_id,
                    query=ex.query,
                    answer=ex.gold_answer or "Answer grounded in retrieved context.",
                    contexts=[],
                    gold_answer=ex.gold_answer,
                )
            )

    if args.compare_embeddings:
        comp = compare_embeddings(pack, args.mode, ["bge", "e5", "instructor", "minilm"])
        out_path = Path("reports/embedding_comparison.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(comp, indent=2), encoding="utf-8")
        print(json.dumps(comp, indent=2))
        return

    ev = RAGEvaluator(embedding_key=args.embedding, dense_backend=args.backend, top_k=args.top_k)
    try:
        payload = ev.full_benchmark(
            pack,
            samples,
            args.mode,
            use_llm_judge=args.llm_judge,
            use_nli_faithfulness=args.nli_faithfulness,
        )
    except JudgeConfigurationError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    print(json.dumps(payload, indent=2, default=str))

    if not args.no_save:
        tracker = ExperimentTracker()
        saved = tracker.save_run({"cli": vars(args), **payload}, name=args.mode)
        rid = saved["run_id"]
        write_benchmark_report({**payload, "run_id": rid})
        figures = render_dashboard_bundle({**payload, "run_id": rid}, rid)
        print("\nArtifacts:", saved["path"], figures)


if __name__ == "__main__":
    main()
