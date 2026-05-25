"""Helpers for the Streamlit dashboard (document ingest, packs, plotting data)."""

from __future__ import annotations

import io
from collections import defaultdict
from pathlib import PurePath
from typing import Any

import pandas as pd

from rag_eval.chunking import STRATEGIES, get_strategy
from rag_eval.schemas import BenchmarkPack, QAExample


def extract_text(uploaded_file) -> tuple[str, str]:
    """Returns (doc_id_stem, full_text)."""
    raw_name = getattr(uploaded_file, "name", "document")
    stem = PurePath(raw_name).stem or "document"
    name_lower = raw_name.lower()
    blob = uploaded_file.read()

    if name_lower.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(blob))
        parts: list[str] = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        return stem, "\n".join(parts)

    if name_lower.endswith(".docx"):
        try:
            import docx
        except ImportError as e:
            raise RuntimeError(
                ".docx support requires python-docx. Install with: pip install python-docx"
            ) from e
        doc = docx.Document(io.BytesIO(blob))
        paras = [p.text for p in doc.paragraphs if p.text.strip()]
        return stem, "\n".join(paras)

    text = blob.decode("utf-8", errors="replace").strip()
    return stem, text


def ingest_uploads(uploaded_files: list[Any]) -> dict[str, str]:
    docs: dict[str, str] = {}
    seen: dict[str, int] = {}
    for uf in uploaded_files or []:
        stem, text = extract_text(uf)
        if not text.strip():
            continue
        key = stem
        if key in docs:
            seen[key] = seen.get(key, 1) + 1
            key = f"{stem}_{seen[key]}"
        docs[key] = text.strip()
    return docs


def build_chunks(raw_docs: dict[str, str], chunking_strategy: str, **strategy_kwargs: Any):
    strat = get_strategy(chunking_strategy, **strategy_kwargs)
    chunks = []
    for doc_id, text in raw_docs.items():
        chunks.extend(strat.chunk(doc_id, text))
    return chunks


def build_benchmark_pack(
    chunks: list,
    corpus_id: str,
    query_rows: list[dict[str, Any]],
) -> BenchmarkPack | None:
    """
    ``query_rows`` items: ``query``, optional ``gold_doc`` (upload doc id / ``source_doc_id``), optional ``answer``.
    """
    if not chunks:
        return None
    doc_to_chunks: dict[str, list[str]] = defaultdict(list)
    for ch in chunks:
        sid = str(ch.metadata.get("source_doc_id") or "")
        if sid:
            doc_to_chunks[sid].append(ch.id)

    examples: list[QAExample] = []
    for i, row in enumerate(query_rows):
        q_text = str(row.get("query", "")).strip()
        if not q_text:
            continue
        qid = f"uq_{i}"
        gold_doc = str(row.get("gold_doc") or "").strip()
        rel_ids = list(dict.fromkeys(doc_to_chunks.get(gold_doc, [])))
        examples.append(
            QAExample(
                query_id=qid,
                query=q_text,
                relevant_chunk_ids=rel_ids,
                gold_answer=str(row["answer"]).strip() if row.get("answer") else None,
            )
        )
    if not examples:
        return None

    pack = BenchmarkPack(corpus_id=corpus_id, chunks=chunks, examples=examples)
    return pack


def retrieval_metrics_to_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    recalls = [(k.replace("recall@", "Recall@"), v) for k, v in metrics.items() if k.startswith("recall@")]
    ndcg = [(k.replace("ndcg@", "NDCG@"), v) for k, v in metrics.items() if k.startswith("ndcg@")]
    rows = [*recalls, *ndcg, ("MRR", metrics.get("mrr"))]
    return pd.DataFrame([{"metric": k, "value": float(v) if v is not None else None} for k, v in rows if v is not None])


def latency_to_frame(latency_ms: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for k, v in latency_ms.items():
        if isinstance(v, dict):
            rows.append({"stage": k, **{kk: vv for kk, vv in v.items()}})
    return pd.DataFrame(rows)


AVAILABLE_CHUNK_STRATEGIES = sorted(STRATEGIES.keys())
