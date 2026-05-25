"""JSON helpers for benchmark packs."""

from __future__ import annotations

import json
from pathlib import Path

from rag_eval.schemas import BenchmarkPack


def load_benchmark_pack(path: Path | str) -> BenchmarkPack:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return BenchmarkPack.model_validate(raw)


def save_benchmark_pack(pack: BenchmarkPack, path: Path | str) -> None:
    Path(path).write_text(json.dumps(pack.model_dump(), indent=2), encoding="utf-8")
