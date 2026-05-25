"""
RAG Evaluation Benchmark Suite — interactive Streamlit dashboard.

Run from repo root:

    cd dashboard
    streamlit run app.py

Or: streamlit run dashboard/app.py

Requires: pip install -e ".[dashboard]"
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

# Path: ``helpers`` lives next to this file; ``rag_eval`` lives in the parent directory.
_DASH_DIR = Path(__file__).resolve().parent
_ROOT = _DASH_DIR.parent
for p in (_DASH_DIR, _ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rag_eval.experiments.tracker import ExperimentTracker
from rag_eval.latency import LatencySummary
from rag_eval.pipeline.evaluator import RAGEvaluator, compare_embeddings, retrieval_overlap_matrix
from rag_eval.schemas import BenchmarkPack, GenerationSample

from helpers import (
    AVAILABLE_CHUNK_STRATEGIES,
    build_benchmark_pack,
    build_chunks,
    ingest_uploads,
    latency_to_frame,
    retrieval_metrics_to_frame,
)

RetrievalLiteral = ["bm25", "dense", "hybrid"]
EmbeddingChoices = ["minilm", "bge", "e5", "instructor"]


@st.cache_resource
def _cached_torch_health() -> tuple[bool, str | None]:
    from rag_eval.torch_health import pytorch_load_probe

    return pytorch_load_probe()


def _require_pytorch(or_else: str) -> None:
    ok, detail = _cached_torch_health()
    if ok:
        return
    st.error(
        f"**{or_else}** needs **PyTorch** + `sentence-transformers`, but loading failed:\n\n"
        f"`{detail}`\n\n"
        "**Workaround:** choose **BM25** in the sidebar (no PyTorch).\n\n"
        "**Fix (Windows):** use **python.org** Python (avoid Microsoft Store builds for torch), "
        "install **[VC++ Redistributable x64](https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist)**, "
        "then reinstall: `pip install --upgrade torch` (see https://pytorch.org)."
    )
    st.stop()


def _theme_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; max-width: 1400px; }
        h1 { font-variant: small-caps; letter-spacing: 0.04em; font-weight: 600; }
        div[data-testid="stSidebar"] { background-color: #0f172a08; }
        .stMarkdown code { background-color: rgba(125,125,125,0.12); padding: 0.1rem 0.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _runs_dir() -> Path:
    try:
        from rag_eval.config import get_settings

        return Path(get_settings().runs_dir)
    except Exception:
        return Path("runs")


def _run_evaluation(
    pack: BenchmarkPack,
    *,
    embedding_key: str,
    retrieval_mode: str,
    dense_backend: str,
    top_k: int,
    use_llm_judge: bool,
    use_nli_faithfulness: bool,
):
    lat = LatencySummary()
    t0 = time.perf_counter()
    ev = RAGEvaluator(embedding_key=embedding_key, dense_backend=dense_backend, top_k=top_k)

    hits, retr = ev.run_retrieval(pack, retrieval_mode, latency=lat)

    retr_wall_ms = (time.perf_counter() - t0) * 1000
    lat_agg = lat.aggregate()
    samples = [
        GenerationSample(
            query_id=ex.query_id,
            query=ex.query,
            answer=(ex.gold_answer or "").strip() or "(no answer provided)",
            contexts=[],
            gold_answer=ex.gold_answer,
        )
        for ex in pack.examples
    ]
    gm = {}
    judge_meta = {}

    gm_raw = ev.run_generation_benchmark(
        pack,
        samples,
        hits,
        use_llm_judge=use_llm_judge,
        use_nli_faithfulness=use_nli_faithfulness,
    )
    if gm_raw.get("generation_metrics"):
        g = gm_raw["generation_metrics"]
        gm["generation_metrics"] = g
        if "judge_token_usage" in g:
            judge_meta["judge_token_usage"] = g["judge_token_usage"]

    out_eval = {
        "hits": hits,
        "retrieval_metrics": retr.get("retrieval_metrics"),
        "latency_ms": {"wall_clock_total": retr_wall_ms, **lat_agg},
        "embedding_key": embedding_key,
        "retrieval_mode": retrieval_mode,
        **gm,
        **judge_meta,
    }
    if retr.get("token_usage_estimate"):
        out_eval["token_usage_estimate"] = retr["token_usage_estimate"]
    return out_eval


def main() -> None:
    _theme_css()
    st.title("RAG evaluation bench")
    st.caption("Experimentation dashboard — retrieval diagnostics, benchmarks, comparison.")
    st.info(
        "**Corpus uploads live in the left sidebar.** Click the **◀** control (top-left) if the sidebar is "
        "hidden. Under **Corpus**, use the **Browse files** area (drag-and-drop or file picker), then "
        "**Ingest uploads** → **Chunk corpus from ingested docs** before **Run benchmark**."
    )

    ss = st.session_state
    if "raw_docs" not in ss:
        ss.raw_docs = {}
    if "chunks" not in ss:
        ss.chunks = None
    if "query_grid" not in ss:
        ss.query_grid = pd.DataFrame(
            [{"query": "", "gold_document_id": "", "answer": ""} for _ in range(3)],
        )

    sidebar = st.sidebar
    sidebar.header("Corpus")
    sidebar.markdown(
        "Streamlit doesn’t offer a labeled “Upload” button — use **Browse files** "
        "(or drop files onto the dashed box)."
    )

    uploads = sidebar.file_uploader(
        "Step 1 — choose PDF / TXT / DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="visible",
        help="Pick one or many files here, then use **Ingest into session** below.",
    )

    doc_ids_live = sorted(ss.raw_docs.keys())
    sidebar.caption(f"Documents in session: **{len(doc_ids_live)}** loaded.")
    chunking_strategy = sidebar.selectbox("Chunking strategy", AVAILABLE_CHUNK_STRATEGIES)

    sidebar.header("Retrieval pipeline")
    embedding_choice = sidebar.selectbox("Embedding model", EmbeddingChoices)
    retrieval_base = sidebar.selectbox("Retrieval strategy", RetrievalLiteral)

    rerank_toggle = sidebar.checkbox(
        "Apply cross-encoder reranking",
        value=False,
        help="Fusion + MS MARCO cross-encoder (uses hybrid+Rerank path).",
        disabled=(retrieval_base != "hybrid"),
    )
    if retrieval_base != "hybrid":
        rerank_toggle = False

    retrieval_mode = "hybrid_rerank" if rerank_toggle and retrieval_base == "hybrid" else retrieval_base

    torch_ready, torch_err = _cached_torch_health()
    if retrieval_mode != "bm25":
        if not torch_ready:
            sidebar.error(
                "PyTorch is **not** loading on this install (dense/hybrid/rerank will fail). "
                "Use retrieval **BM25**, **or** fix `c10.dll` / WinError 1114 (see error when you Run)."
            )
            sidebar.caption(f"Torch probe: `{torch_err}`")

    top_k = int(sidebar.slider("Top-k", min_value=1, max_value=50, value=10))
    dense_backend = sidebar.radio("Dense vector backend", ["faiss", "qdrant"], horizontal=True)

    sidebar.header("Generation metrics options")
    use_llm_judge = sidebar.checkbox(
        "OpenAI-compatible LLM judge",
        value=False,
        help="Needs RAG_EVAL_OPENAI_COMPATIBLE_BASE_URL (+ API version for Azure)",
    )
    use_nli = sidebar.checkbox("NLI entailment (`faithfulness_nli`)", value=False)

    if sidebar.button(
        "Step 2 — Ingest into session",
        type="primary",
        disabled=not uploads,
        help="Parses uploads into `raw_docs` (required before Chunk corpus).",
    ):
        merged = ingest_uploads(list(uploads))
        if not merged:
            st.sidebar.error("Could not extract text from uploads.")
        else:
            ss.raw_docs = merged
            st.sidebar.success(f"Ingested {len(merged)} document(s).")
            ss.chunks = None
            ss["_bundle_pack"] = None

    sample_pack = sidebar.selectbox(
        "Or load bundled JSON pack",
        ("(none)", "sample_benchmark.json", "benchmark_medium.json"),
    )
    if sidebar.button("Load bundled pack"):
        if sample_pack != "(none)":
            json_path = _ROOT / "data" / sample_pack.replace("(none)", "").lstrip("/")
            if json_path.exists():
                data = json.loads(json_path.read_text(encoding="utf-8"))
                bp = BenchmarkPack.model_validate(data)
                ss["_bundle_pack"] = bp
                st.sidebar.success(f"Loaded `{sample_pack}` — bypasses upload/chunk.")

    if sidebar.button("Step 3 — Chunk corpus from ingested docs") and ss.raw_docs:
        try:
            ss.chunks = build_chunks(dict(ss.raw_docs), chunking_strategy)
            ss["_bundle_pack"] = None
            sidebar.success(f"Built **{len(ss.chunks)}** chunks.")
        except Exception as exc:
            sidebar.exception(exc)

    st.divider()

    tabs = st.tabs(["Run & metrics", "Retrieval debugger", "Side-by-side comparison", "Experiments"])

    bp: BenchmarkPack | None = ss.get("_bundle_pack")

    edited_queries = st.data_editor(
        ss.query_grid,
        column_config={
            "query": st.column_config.TextColumn("Evaluation query"),
            "gold_document_id": st.column_config.SelectboxColumn(
                "Gold document id",
                options=[""] + doc_ids_live,
                help="Supervised Recall/MRR label: all chunks from this doc.",
            ),
            "answer": st.column_config.TextColumn("Hypothetical answer (optional)"),
        },
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="qeditor",
    )
    ss.query_grid = edited_queries

    queries_payload = edited_queries.to_dict("records")

    with tabs[0]:
        st.caption(
            "This page **scores retrieval**, it does **not** auto-write paragraph answers like ChatGPT. "
            "After **Run benchmark**, open the **Retrieval debugger** tab to read the **top retrieved chunks** (snippets). "
            "Use the **Hypothetical answer** column only if you want to measure overlap/hallucination against *your own* wording."
        )
        btn_run = st.button("Run benchmark", type="primary", use_container_width=True)
        if btn_run:
            if bp is None:
                if not ss.chunks:
                    st.error("Chunk corpus from uploads or load a bundled pack first.")
                    st.stop()
                pack_try = build_benchmark_pack(
                    list(ss.chunks),
                    f"sess_{uuid.uuid4().hex[:8]}",
                    [
                        {
                            "query": r.get("query", ""),
                            "gold_doc": r.get("gold_document_id") or "",
                            "answer": r.get("answer"),
                        }
                        for r in queries_payload
                    ],
                )
            else:
                pack_try = bp

            if pack_try is None:
                st.error("Build at least one query with non-empty **query**.")
                st.stop()

            needs_dense = retrieval_mode != "bm25"

            backend_use = dense_backend if needs_dense else "faiss"
            embed_use = embedding_choice if needs_dense else "minilm"

            msg = (
                retrieval_mode.upper()
                + f", **{embed_use}**"
                + (f", `{backend_use}`" if needs_dense else "")
            )
            if needs_dense:
                _require_pytorch("Dense / hybrid / rerank benchmarking")
            with st.spinner(f"Benchmarking ({msg}) — first dense run may download models…"):
                try:
                    out = _run_evaluation(
                        pack_try,
                        embedding_key=embed_use,
                        retrieval_mode=retrieval_mode,
                        dense_backend=backend_use,
                        top_k=top_k,
                        use_llm_judge=use_llm_judge,
                        use_nli_faithfulness=use_nli,
                    )
                    ss["last_pack"] = pack_try.model_dump(mode="json")
                    ss["_last_bundle"] = bool(bp)
                    ss["last_payload"] = {k: v for k, v in out.items() if k != "hits"}
                    ss["hits"] = out["hits"]
                    st.success("Run complete.")
                except Exception as exc:
                    st.exception(exc)

        lp = ss.get("last_payload")
        if lp:
            c1, c2, c3 = st.columns(3)
            rm = lp.get("retrieval_metrics") or {}
            mv = rm.get("mrr")
            with c1:
                st.metric("MRR", f"{float(mv):.4f}" if mv is not None else "—")

            rk = [(k.replace("recall@", ""), float(v)) for k, v in rm.items() if k.startswith("recall@")]
            with c2:
                if rk:
                    best_k = max(rk, key=lambda x: int(x[0]) if x[0].isdigit() else 0)
                    st.metric(f"Recall@{best_k[0]}", f"{best_k[1]:.4f}")
                else:
                    st.metric("Recall@K", "—")

            gm_mean = ((lp.get("generation_metrics") or {}).get("mean") or {})
            lr = gm_mean.get("hallucination_rate_llm") or gm_mean.get("hallucination_rate")
            with c3:
                st.metric("Hallucination (mean)", f"{float(lr):.3f}" if lr is not None else "—")

            st.subheader("Retrieval metrics")
            rf = retrieval_metrics_to_frame(rm)
            col_a, col_b = st.columns(2)
            with col_a:
                if not rf.empty:
                    fc = rf[rf["metric"].str.startswith("Recall")]
                    fn = rf[rf["metric"].str.startswith("NDCG")]
                    if fc.empty:
                        fc = rf
                    fig_r = px.bar(fc, x="metric", y="value", title="Recall — query set")
                    st.plotly_chart(fig_r, use_container_width=True)
                    if not fn.empty:
                        fig_n = px.bar(fn, x="metric", y="value", title="NDCG")
                        st.plotly_chart(fig_n, use_container_width=True)
            with col_b:
                if mv is not None:
                    fig_m = px.bar(pd.DataFrame([{"metric": "MRR", "value": float(mv)}]), x="metric", y="value", title="MRR")
                    st.plotly_chart(fig_m, use_container_width=True)

            st.subheader("Latency")
            lf = latency_to_frame(lp.get("latency_ms") or {})
            if not lf.empty:
                st.dataframe(lf, use_container_width=True)
            _lm = lp.get("latency_ms") or {}
            if "retrieve_query" in _lm and isinstance(_lm["retrieve_query"], dict):
                _fig_mpl, ax = plt.subplots(figsize=(5, 2.8))
                rq = _lm["retrieve_query"]
                ax.bar(["mean_ms", "p95_ms"], [rq["mean_ms"], rq["p95_ms"]], color=["#475569", "#94a3b8"])
                ax.set_ylabel("ms")
                ax.set_title("Retrieval latency (profiling)")
                st.pyplot(_fig_mpl)
                plt.close(_fig_mpl)

            gm = lp.get("generation_metrics") or {}
            st.subheader("Generation / hallucination summary")
            if gm.get("mean"):
                gm_df = pd.DataFrame([gm["mean"]])
                st.dataframe(gm_df, use_container_width=True)
                hq = gm.get("per_query") or []
                if hq and isinstance(hq, list) and hq[0] and "hallucination_rate" in hq[0]:
                    fig_h = px.histogram([float(x.get("hallucination_rate", 0)) for x in hq], nbins=min(15, len(hq) + 2), title="Hallucination rate (lexical), per-query")
                    st.plotly_chart(fig_h, use_container_width=True)
                hq_llm = []
                if hq and isinstance(hq, list):
                    hq_llm = [float(x["hallucination_rate_llm"]) for x in hq if "hallucination_rate_llm" in x]
                if hq_llm:
                    fig_hl = px.histogram(hq_llm, nbins=min(15, len(hq_llm) + 2), title="Hallucination rate (LLM judge), per-query")
                    st.plotly_chart(fig_hl, use_container_width=True)

            st.subheader("Embedding comparison sweep (dense-capable)")
            sweep_mode = retrieval_mode if retrieval_mode != "bm25" else "dense"
            if sweep_mode == "hybrid_rerank":
                sweep_mode = "hybrid"
            lbl = sweep_mode.upper()
            if st.button(
                f"Compute embedding comparison ({lbl})",
                help="Uses dense-capable mode; downloads checkpoints on first sweep.",
                key="btn_emb_sweep",
            ):
                pk = BenchmarkPack.model_validate(ss["last_pack"]) if ss.get("last_pack") else None
                if not pk:
                    st.warning("Run a benchmark first to define the pack.")
                else:
                    _require_pytorch("Embedding comparison sweep")
                    with st.spinner("Sweeping embeddings…"):
                        cmp = compare_embeddings(pk, sweep_mode, list(EmbeddingChoices))
                        rows = cmp.get("embedding_comparison") or []
                        if rows:
                            df_e = pd.DataFrame(rows)
                            st.dataframe(df_e, use_container_width=True)
                            mets = ["mrr", "recall@5", "ndcg@5"]
                            melted = df_e.melt(id_vars=["embedding"], value_vars=[c for c in mets if c in df_e.columns])
                            fig_e = px.bar(
                                melted,
                                x="embedding",
                                y="value",
                                color="variable",
                                barmode="group",
                                title="Embedding comparison",
                            )
                            st.plotly_chart(fig_e, use_container_width=True)
            if st.button("Measure reranking impact vs hybrid fusion"):
                pk = BenchmarkPack.model_validate(ss["last_pack"]) if ss.get("last_pack") else None
                if not pk:
                    st.warning("Run a benchmark first.")
                else:
                    _require_pytorch("Hybrid vs hybrid+rerank comparison")
                    with st.spinner("Hybrid vs Hybrid+Rerank…"):
                        hb = _run_evaluation(
                            pk,
                            embedding_key=embedding_choice,
                            retrieval_mode="hybrid",
                            dense_backend=dense_backend,
                            top_k=top_k,
                            use_llm_judge=False,
                            use_nli_faithfulness=False,
                        )
                        hr = _run_evaluation(
                            pk,
                            embedding_key=embedding_choice,
                            retrieval_mode="hybrid_rerank",
                            dense_backend=dense_backend,
                            top_k=top_k,
                            use_llm_judge=False,
                            use_nli_faithfulness=False,
                        )
                        hbm = hb.get("retrieval_metrics") or {}
                        hrm = hr.get("retrieval_metrics") or {}
                        diff = pd.DataFrame(
                            {
                                "metric": sorted(set(hbm) | set(hrm)),
                                "hybrid": [hbm.get(k) for k in sorted(set(hbm) | set(hrm))],
                                "hybrid_rerank": [hrm.get(k) for k in sorted(set(hbm) | set(hrm))],
                            },
                        ).dropna(how="all")
                        st.dataframe(diff, use_container_width=True)
                        long = diff.melt(id_vars="metric", var_name="strategy", value_name="value")
                        fig_rb = px.bar(long, x="metric", y="value", color="strategy", barmode="group", title="Reranking impact (retrieval metrics)")
                        st.plotly_chart(fig_rb, use_container_width=True)

            if sidebar.button("Save run to experiments/tracker"):
                if ss.get("last_payload"):
                    try:
                        tr = ExperimentTracker()
                        merged = dict(ss["last_pack"] or {})
                        merged.update(dict(ss["last_payload"]))
                        tr.save_run(merged, name="streamlit_ui")
                        st.success("Saved JSON under configured `runs/` dir.")
                    except Exception as exc:
                        st.exception(exc)

    with tabs[1]:
        hits = ss.get("hits")
        lp2 = ss.get("last_payload")
        if hits and lp2:
            for res in hits:
                with st.expander(f"`{res.query_id}`"):
                    dd = [{"rank": i + 1, "chunk_id": h.chunk_id, "score": h.score, "text_preview": (h.text or "")[:800]} for i, h in enumerate(res.hits)]
                    st.dataframe(pd.DataFrame(dd), use_container_width=True)
        else:
            st.info("Run a benchmark — retrieved chunks render here.")

    with tabs[2]:
        st.markdown("Configure **systems A/B** vs the last successful pack.")

        colx, coly = st.columns(2)
        with colx:
            mode_a = st.selectbox("System A retrieval", RetrievalLiteral, key="sx_a_mode")
            emb_a = st.selectbox("System A embedding", EmbeddingChoices, key="sx_a_emb")
        with coly:
            mode_b = st.selectbox("System B retrieval", RetrievalLiteral, key="sx_b_mode")
            emb_b = st.selectbox("System B embedding", EmbeddingChoices, key="sx_b_emb")

        tb = st.checkbox("Overlap uses top-5 retrieved chunk ids.", value=True)

        run_compare = st.button("Run side-by-side retrieval")
        if run_compare and ss.get("last_pack"):
            pk = BenchmarkPack.model_validate(ss["last_pack"])
            needs_emb_a = mode_a != "bm25"
            needs_emb_b = mode_b != "bm25"
            if needs_emb_a or needs_emb_b:
                _require_pytorch("Side-by-side comparison (dense/hybrid)")
            backend_a = dense_backend if needs_emb_a else "faiss"
            backend_b = dense_backend if needs_emb_b else "faiss"
            with st.spinner("Pairwise retrieval comparison…"):
                ev_a = RAGEvaluator(
                    embedding_key=emb_a if needs_emb_a else "minilm",
                    dense_backend=backend_a,
                    top_k=top_k,
                )
                ev_b = RAGEvaluator(
                    embedding_key=emb_b if needs_emb_b else "minilm",
                    dense_backend=backend_b,
                    top_k=top_k,
                )
                out_a = ev_a.run_retrieval(pk, mode_a)
                out_b = ev_b.run_retrieval(pk, mode_b)
                overlap = retrieval_overlap_matrix(out_a[0], out_b[0], k=5 if tb else max(5, top_k))

            st.metric("Mean Jaccard overlap @k", f"{overlap:.3f}")

            qa = pk.examples[0].query_id if pk.examples else None
            if qa:
                ha = next((x.hits[:5] for x in out_a[0] if x.query_id == qa), [])
                hb = next((x.hits[:5] for x in out_b[0] if x.query_id == qa), [])
                zx, zy = st.columns(2)
                with zx:
                    st.markdown("**A hits**")
                    st.dataframe(pd.DataFrame([h.model_dump() for h in ha]))
                with zy:
                    st.markdown("**B hits**")
                    st.dataframe(pd.DataFrame([h.model_dump() for h in hb]))
        elif not ss.get("last_pack"):
            st.warning("Run a benchmark (or reload pack) before comparison.")

    with tabs[3]:
        runs = ExperimentTracker(_runs_dir()).list_runs()
        sel = [r for r in runs if r.get("path")]
        if not sel:
            st.info(f"No experiments in `{_runs_dir()}` yet.")
        else:
            ids = [(r["path"], f"{r.get('created_at')} — …{Path(r['path']).name}") for r in sel]
            pc = [p for p, _ in ids]

            sel_a_path = st.selectbox("Baseline run", pc, format_func=lambda p: Path(p).name)
            sel_b_path = st.selectbox("Compared run", pc, format_func=lambda p: Path(p).name)

            ja = json.loads(Path(sel_a_path).read_text(encoding="utf-8"))
            jb = json.loads(Path(sel_b_path).read_text(encoding="utf-8"))
            ma, mb = ja.get("retrieval_metrics") or {}, jb.get("retrieval_metrics") or {}
            ck = sorted(set(ma) | set(mb))
            cmp_df = pd.DataFrame({"metric": ck, "run_a": [ma.get(k) for k in ck], "run_b": [mb.get(k) for k in ck], "delta": [((mb.get(k) or 0) - (ma.get(k) or 0)) if isinstance(ma.get(k), (int, float)) and isinstance(mb.get(k), (int, float)) else None for k in ck]})
            st.dataframe(cmp_df, use_container_width=True)


if __name__ == "__main__":
    main()