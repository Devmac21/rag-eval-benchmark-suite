#!/usr/bin/env python3
"""
Run the **same retrieval path as the dashboard** after ingest + chunk, using real `.txt`
fixtures under ``data/fixtures/sample_docs/`` (not ``sample_benchmark.json``).

Matches dashboard flow::

    raw_docs dict  →  ``build_chunks``  →  ``build_benchmark_pack``  → ``RAGEvaluator`` BM25 ``full_benchmark``

Usage from repo root::

    python scripts/smoke_dashboard_documents.py
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "data" / "fixtures" / "sample_docs"


def load_raw_documents() -> dict[str, str]:
    """Plain-text equivalents of sidebar uploads ``raw_docs`` (stem -> full text)."""
    if not DOC_DIR.is_dir():
        raise FileNotFoundError(f"Missing fixture folder: {DOC_DIR}")
    out: dict[str, str] = {}
    for p in sorted(DOC_DIR.glob("*.txt")):
        out[p.stem] = p.read_text(encoding="utf-8").strip()
    if not out:
        raise FileNotFoundError(f"No ``.txt`` files in {DOC_DIR}")
    return out


def main() -> None:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "dashboard"))

    from helpers import build_benchmark_pack, build_chunks

    from rag_eval.pipeline.evaluator import RAGEvaluator
    from rag_eval.schemas import GenerationSample

    raw_docs = load_raw_documents()
    print("# Documents:", ", ".join(sorted(raw_docs.keys())), file=sys.stderr)

    chunks = build_chunks(raw_docs, "fixed_window")
    print(f"# Chunks ({len(chunks)}), strategy=fixed_window", file=sys.stderr)

    corpus_id = f"fixture_{uuid.uuid4().hex[:8]}"
    query_rows = [
        {
            "query": "When is the enrollment deposit due?",
            "gold_doc": "courant_demo",
            "answer": "April 15, 2025 per the demo admission letter.",
        },
        {
            "query": "What GPA triggers academic probation?",
            "gold_doc": "handbook_demo",
            "answer": "GPA below 3.0 for one semester in the handbook demo.",
        },
    ]

    pack = build_benchmark_pack(list(chunks), corpus_id, query_rows)
    if pack is None:
        raise SystemExit("build_benchmark_pack returned None (empty queries?)")

    samples: list[GenerationSample] = []
    for i, row in enumerate(query_rows):
        q_text = str(row.get("query", "")).strip()
        if not q_text:
            continue
        qid = f"uq_{i}"
        samples.append(
            GenerationSample(
                query_id=qid,
                query=q_text,
                answer=str(row.get("answer") or "").strip(),
                contexts=[],
                gold_answer=None,
            ),
        )

    print("# Running BM25 full_benchmark (same as dashboard path)...\n", file=sys.stderr)

    ev = RAGEvaluator(top_k=10)
    payload = ev.full_benchmark(
        pack,
        samples,
        "bm25",
        use_llm_judge=False,
        use_nli_faithfulness=False,
    )

    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
