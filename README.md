# RAG Eval Benchmark Suite

Modular toolkit for benchmarking retrieval-augmented generation (RAG) stacks: **retrieval metrics**, **generation metrics**, **latency**, **hallucination-oriented signals**, **chunking comparisons**, and **embedding comparisons**. Ships with **FastAPI**, **Docker**, **FAISS / Qdrant**, **pandas**, matplotlib report figures, and an optional **Streamlit** experimentation UI.

## Architecture

```mermaid
flowchart TB
  subgraph Data
    BP[BenchmarkPack: chunks + QA + relevance ids]
  end
  subgraph Retrieval
    BM25[BM25 Index]
    DEN[Dense Retriever]
    HYB[Hybrid RRF]
    RER[Cross-Encoder Rerank]
  end
  subgraph Stores
    FAISS[FAISS IndexFlatIP]
    QD[Qdrant Client]
  end
  subgraph Metrics
    RM[Recall@K / MRR / NDCG]
    GM[Faithfulness / Relevance / Precision / Hallucination rate]
    LAT[Latency summaries]
  end
  subgraph Outputs
    RUNS[Experiment JSON runs/]
    REP[Reports reports/]
    FIG[Charts reports/figures/]
  end
  BP --> BM25
  BP --> DEN
  BM25 --> HYB
  DEN --> HYB
  HYB --> RER
  DEN --> FAISS
  DEN --> QD
  BM25 --> RM
  RER --> RM
  HYB --> RM
  BP --> GM
  RM --> RUNS
  GM --> RUNS
  LAT --> RUNS
  RUNS --> REP
  RUNS --> FIG
```

### Layout

| Path | Role |
|------|------|
| `dashboard/app.py` | Streamlit experimentation UI (runs, retrieval debugger, A/B, plots) |
| `rag_eval/metrics/nli.py` | Optional `faithfulness_nli` (MNLI cross-encoder entailment vs contexts) |
| `rag_eval/metrics/` | Recall@K, MRR, NDCG; faithfulness, answer relevance, context precision, hallucination rate (+ optional `*_llm` scores) |
| `rag_eval/retrieval/` | BM25, dense (FAISS/Qdrant), hybrid RRF, cross-encoder rerank |
| `rag_eval/embeddings/` | Presets: **BGE**, **E5**, **Instructor**, **MiniLM** (lazy-loaded) |
| `rag_eval/chunking/` | `fixed_window`, `sentence`, `recursive` strategies |
| `rag_eval/pipeline/evaluator.py` | `RAGEvaluator`, embedding/chunk comparisons, overlap matrix |
| `rag_eval/experiments/` | JSON experiment tracker under `runs/` |
| `rag_eval/reports/` | Markdown + CSV summaries |
| `rag_eval/viz/` | Matplotlib charts |
| `app/main.py` | FastAPI service |

## Install

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install -e .
```

**Streamlit dashboard** (additional packages):

```bash
pip install -e ".[dashboard]"
# alternative: pip install -r requirements.txt -r requirements-dashboard.txt
```

Optional: point dense retrieval at Qdrant (see Docker Compose).

## Quick start (CLI)

Additional CLI flags: `--llm-judge`, `--nli-faithfulness` (cross-encoder NLI; downloads weights on first use).

```bash
python scripts/run_benchmark.py --pack data/sample_benchmark.json --mode bm25 --no-save
```

Larger toy corpus: `data/benchmark_medium.json`.

Dense retrieval (downloads the selected embedding checkpoint on first use):

```bash
python scripts/run_benchmark.py --mode dense --embedding bge --backend faiss
```

Embedding comparison (downloads each checkpoint on first use):

```bash
python scripts/run_benchmark.py --compare-embeddings --mode dense
```

## Docker

```bash
docker compose up --build
```

- API: `http://localhost:8000`
- Qdrant: `http://localhost:6333`
- Set `QDRANT_URL=http://qdrant:6333` for dense backend `qdrant` inside containers.

## Benchmark methodology

1. **Corpus**: Build `DocumentChunk` records (stable `id`, `text`, optional `metadata`).
2. **Questions**: Each `QAExample` supplies `query_id`, `query`, and `relevant_chunk_ids` for supervised retrieval metrics.
3. **Retrieval**: Run BM25, dense (cosine on normalized embeddings), hybrid (RRF over BM25 + dense), or hybrid + cross-encoder rerank.
4. **Supervised retrieval scores**: Per-query labels drive Recall@K, MRR, and NDCG@K (binary relevance).
5. **Generation**: Provide `GenerationSample` with `answer` (and optional `contexts`; contexts default to retrieved chunks). Metrics:
   - **Faithfulness**: overlap of informative answer tokens with retrieved context union (fast lexical proxy).
   - **Answer relevance**: cosine similarity between answer and query embeddings (MiniLM-based).
   - **Context precision**: fraction of retrieved hits whose chunk id is in `relevant_chunk_ids`.
   - **Hallucination rate**: fraction of answer sentences with no token overlap with contexts (proxy for unsupported sentences).
   - **`faithfulness_nli`** (optional): entailment probability from a small NLI cross-encoder (`use_nli_faithfulness` / `--nli-faithfulness`).
6. **Latency**: Stage timers wrap retrieval calls (`retrieve_query`, optional `index_dense`) plus wall-clock totals.
7. **Token usage**: Rough counts via `tiktoken` when available, otherwise `len(text)//4`.

Lexical / embedding proxies stay the default. Optionally enable an **LLM-as-judge** (OpenAI-compatible Chat Completions) to add `faithfulness_llm`, `answer_relevance_llm`, and `hallucination_rate_llm` plus aggregated `judge_token_usage` — see [Optional LLM-as-judge](#optional-llm-as-judge).

## Optional LLM-as-judge

What it **requires**:

| Requirement | Notes |
|-------------|--------|
| **Reachable HTTP endpoint** | Must implement OpenAI-style `POST {base_url}/chat/completions` (OpenAI, Azure OpenAI with the right path, vLLM, LM Studio, LiteLLM proxy, etc.). |
| **`RAG_EVAL_OPENAI_COMPATIBLE_BASE_URL`** | Include the `/v1` segment when the server expects it, e.g. `https://api.openai.com/v1` or `http://127.0.0.1:1234/v1`. |
| **`RAG_EVAL_OPENAI_COMPATIBLE_API_KEY`** | Secret for hosted APIs; may be **empty** for many local servers that do not check auth. |
| **`RAG_EVAL_JUDGE_MODEL`** | Model id accepted by that server (default `gpt-4o-mini`). |
| **Network egress** | Allow outbound HTTPS (cloud) or LAN to your inference host. |
| **Latency & cost** | One judge **HTTP request per generation sample** when enabled (`use_llm_judge` / `--llm-judge`). Budget for provider billing and rate limits. |
| **Model behavior** | The judge prompt asks for **strict JSON** with three floats; smaller models may occasionally drift — failures appear as `judge_error` on that row. |

Optional tuning: `RAG_EVAL_JUDGE_TIMEOUT_S` (default `120`), `RAG_EVAL_JUDGE_MAX_TOKENS` (default `450`). For **Azure OpenAI**–style hosts that require `api-version` on the URL, set `RAG_EVAL_OPENAI_COMPATIBLE_API_VERSION` (e.g. `2024-02-15-preview`).

**CLI**: `python scripts/run_benchmark.py ... --llm-judge` after exporting the env vars.

**API**: set `"use_llm_judge": true` on `POST /benchmark/full` or `POST /evaluate/generation`.

**Python**: `ev.full_benchmark(pack, samples, "hybrid", use_llm_judge=True, use_nli_faithfulness=True)` adds **`faithfulness_nli`** (cross-encoder NLI entailment probability vs retrieved contexts).

## Experiment tracking & reports

- **Runs**: `ExperimentTracker.save_run` writes JSON under `runs/` (configure with `RAG_EVAL_RUNS_DIR`).
- **Reports**: `write_benchmark_report` emits `reports/benchmark_<run_id>.md` and `.csv`.
- **Charts**: `render_dashboard_bundle` writes PNGs under `reports/figures/<run_id>/` (retrieval curves, latency bars, token bars, hallucination histogram).

## Interactive dashboard (Streamlit)

From the repo root:

```bash
pip install -e ".[dashboard]"
cd dashboard
streamlit run app.py
```

Adds sidebar controls for corpus upload/chunking, retrieval mode (BM25 / dense / hybrid / hybrid rerank),
embedding preset, judges/NLI, and tabs for benchmark results, retrieval debugger,
side‑by‑side retrieval overlap, and experiment history deltas over `runs/*.json`.

### Screenshots

Screenshots belong in **`docs/dashboard/`**. Add PNGs named e.g. `overview.png`,
`debugger.png`, `experiments.png` after you capture your layout — placeholders are intentional until then.

**CLI smoke test (real files, not bundled JSON):** from repo root,

`python scripts/smoke_dashboard_documents.py`

uses plain-text fixtures in **`data/fixtures/sample_docs/`** and runs the same **chunk → BM25 evaluation** path as the sidebar upload flow.

Import paths avoid loading PyTorch until you run dense retrieval, reranking, or embedding-dependent metrics. **BM25‑only runs** tolerate a broken PyTorch install: `answer_relevance` falls back to a lexical dice score (with one `UserWarning`). **Dense / hybrid / rerank / NLI** still require a working `torch` wheel (on Windows: prefer [python.org](https://www.python.org/downloads/) installs over the Store Python, install the MSVC redistributables, then reinstall CPU PyTorch from [pytorch.org](https://pytorch.org)). **WinError 1114** loading `c10.dll`: the dashboard now blocks dense runs with instructions; fixing the environment remains required for semantic retrieval.

## CI

The workflow `.github/workflows/ci.yml` runs `pytest tests/` on push and pull requests to `main`/`master`.

## Sample experiment results

Using `data/sample_benchmark.json` with `--mode bm25` on the maintainer environment (numbers vary by hardware and caching):

| Metric | Value |
|--------|------:|
| MRR | 1.0 |
| Recall@5 | 1.0 |
| NDCG@5 | 1.0 |
| Mean faithfulness | 1.0 |
| Mean answer relevance | ~0.10 |
| Mean context precision | 0.33 |
| Mean hallucination rate | 0.0 |

Context precision is below 1.0 when `top_k` is larger than the number of gold chunks per query (extra chunks are counted non-relevant).

## REST API

| Endpoint | Description |
|----------|-------------|
| `POST /evaluate/retrieval` | BM25 / dense / hybrid / hybrid_rerank evaluation |
| `POST /evaluate/generation` | Generation metrics; set `use_llm_judge: true` for LLM scores |
| `POST /benchmark/full` | End-to-end run; optional `use_llm_judge`, `use_nli_faithfulness` |
| `POST /benchmark/embeddings` | Table + overlap heatmap across embedding presets |
| `POST /benchmark/chunking` | Compare chunking strategies (document-level gold `gold_document_id`) |
| `GET /experiments` | List saved JSON runs |
| `GET /experiments/{run_id}` | Load one run record by UUID |
| `POST /visualize` | Regenerate matplotlib bundle from a payload |
| `GET /health` | Liveness (**no API key**) |

When `RAG_EVAL_API_KEY` is set, protected routes require header **`X-API-Key`**.

Example body for `/benchmark/full` uses the same schema as `BenchmarkPack` plus `samples: GenerationSample[]`, optional `use_llm_judge`, and optional `use_nli_faithfulness` (adds `faithfulness_nli` in generation metrics).

## Evaluation examples (Python)

```python
from pathlib import Path
from rag_eval.io import load_benchmark_pack
from rag_eval.pipeline.evaluator import RAGEvaluator
from rag_eval.schemas import GenerationSample

pack = load_benchmark_pack(Path("data/sample_benchmark.json"))
samples = [
    GenerationSample(
        query_id="q_python_author",
        query="Who created Python?",
        answer="Guido van Rossum created Python.",
        contexts=[],
    ),
]
ev = RAGEvaluator(embedding_key="minilm", dense_backend="faiss", top_k=5)
payload = ev.full_benchmark(
    pack,
    samples,
    "hybrid",
    use_llm_judge=False,
    use_nli_faithfulness=False,
)
print(payload["retrieval_metrics"], payload["generation_metrics"]["mean"])
```

Compare chunking strategies from raw documents and document-level labels (`compare_chunk_strategies_from_documents`), or build packs manually:

```python
from rag_eval.pipeline.evaluator import compare_chunk_strategies_from_documents

compare_chunk_strategies_from_documents(
    [("doc1", "Long document ...")],
    [("q1", "question?", "doc1")],
    mode="bm25",
    embedding_key="minilm",
)
```

Closure style:

```python
from rag_eval.pipeline.evaluator import compare_chunk_strategies
from rag_eval.schemas import BenchmarkPack, QAExample

raw_docs = [("doc1", "Long document text ... " * 50)]

def pack_builder(chunks):
    return BenchmarkPack(
        corpus_id="ablation",
        chunks=chunks,
        examples=[QAExample(query_id="q1", query="...", relevant_chunk_ids=[chunks[0].id])],
    )

compare_chunk_strategies(raw_docs, pack_builder, mode="dense", embedding_key="minilm")
```

## Configuration

| Env var | Meaning |
|---------|---------|
| `RAG_EVAL_RUNS_DIR` | Experiment JSON directory |
| `RAG_EVAL_REPORTS_DIR` | Markdown / PNG output root |
| `QDRANT_URL` | Optional Qdrant HTTP URL |
| `QDRANT_API_KEY` | Optional Qdrant API key |
| `RAG_EVAL_OPENAI_COMPATIBLE_BASE_URL` | LLM judge API root (e.g. `https://api.openai.com/v1`) |
| `RAG_EVAL_OPENAI_COMPATIBLE_API_KEY` | Bearer token; may be empty locally |
| `RAG_EVAL_JUDGE_MODEL` | Judge model id |
| `RAG_EVAL_JUDGE_TIMEOUT_S` | Judge HTTP timeout seconds |
| `RAG_EVAL_JUDGE_MAX_TOKENS` | Max completion tokens for judge |
| `RAG_EVAL_OPENAI_COMPATIBLE_API_VERSION` | Optional `api-version` query param (e.g. Azure OpenAI) |
| `RAG_EVAL_API_KEY` | If set, clients must send matching `X-API-Key` (except `GET /health`) |

## License

Use and modify according to your organization’s policies.
