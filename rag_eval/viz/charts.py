"""Matplotlib dashboards for offline inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from rag_eval.config import get_settings


def plot_retrieval_quality(metrics_rows: list[dict[str, Any]], out_path: Path) -> Path:
    if not metrics_rows:
        return out_path
    df = pd.DataFrame(metrics_rows)
    numeric = df.select_dtypes(include="number").columns.tolist()
    if not numeric:
        return out_path
    labels_col = None
    for cand in ("mode", "embedding", "chunking"):
        if cand in df.columns:
            labels_col = cand
            break
    x = df[labels_col].tolist() if labels_col else range(len(df))
    plt.figure(figsize=(10, 5))
    for col in numeric[:6]:
        plt.plot(x, df[col], marker="o", label=col)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("score")
    plt.title("Retrieval quality")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_latency_bars(latency_agg: dict[str, dict[str, float]], out_path: Path) -> Path:
    if not latency_agg:
        return out_path
    stages = list(latency_agg.keys())
    means = [latency_agg[s]["mean_ms"] for s in stages]
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(stages)), means, color="#4c72b0")
    plt.xticks(range(len(stages)), stages, rotation=20, ha="right")
    plt.ylabel("mean latency (ms)")
    plt.title("Latency by stage")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_token_usage(tokens: dict[str, int], out_path: Path) -> Path:
    plt.figure(figsize=(6, 4))
    names = list(tokens.keys())
    vals = list(tokens.values())
    plt.bar(names, vals, color=["#4c72b0", "#55a868"])
    plt.ylabel("estimated tokens")
    plt.title("Token usage")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_overlap_heatmap(matrix: list[list[float]], labels: list[str], out_path: Path) -> Path:
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="Jaccard@k overlap")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Retrieval overlap")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_hallucination_hist(values: list[float], out_path: Path) -> Path:
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=min(12, max(4, len(values) // 2)), color="#c44e52", alpha=0.85)
    plt.xlabel("hallucination rate")
    plt.ylabel("count")
    plt.title("Hallucination frequency")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def render_dashboard_bundle(payload: dict[str, Any], run_id: str) -> dict[str, str]:
    reports = Path(get_settings().reports_dir)
    fig_dir = reports / "figures" / run_id
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    retr_row = payload.get("retrieval_metrics")
    mode = payload.get("mode")
    if retr_row and isinstance(retr_row, dict):
        row = {"mode": mode, **retr_row}
        paths["retrieval_png"] = str(plot_retrieval_quality([row], fig_dir / "retrieval.png"))

    lat = payload.get("latency_ms") or {}
    lat_stages = {k: v for k, v in lat.items() if isinstance(v, dict)}
    if lat_stages:
        paths["latency_png"] = str(plot_latency_bars(lat_stages, fig_dir / "latency.png"))

    tok = payload.get("token_usage_estimate")
    if isinstance(tok, dict):
        paths["tokens_png"] = str(plot_token_usage(tok, fig_dir / "tokens.png"))

    gen = (payload.get("generation_metrics") or {}).get("per_query") or []
    halls = [g.get("hallucination_rate") for g in gen if isinstance(g, dict) and "hallucination_rate" in g]
    if halls:
        paths["hallucination_png"] = str(plot_hallucination_hist([float(h) for h in halls], fig_dir / "hallucination.png"))

    return paths
