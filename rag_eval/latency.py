"""Latency instrumentation for retrieval and embedding stages."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class LatencySummary:
    timings_ms: dict[str, list[float]] = field(default_factory=dict)

    def record(self, stage: str, duration_ms: float) -> None:
        self.timings_ms.setdefault(stage, []).append(duration_ms)

    def aggregate(self) -> dict[str, dict[str, float]]:
        import numpy as np

        out: dict[str, dict[str, float]] = {}
        for stage, vals in self.timings_ms.items():
            arr = np.array(vals, dtype=np.float64)
            out[stage] = {
                "mean_ms": float(arr.mean()),
                "p50_ms": float(np.percentile(arr, 50)),
                "p95_ms": float(np.percentile(arr, 95)),
                "max_ms": float(arr.max()),
            }
        return out


def timed(stage: str, summary: LatencySummary, fn: Callable[[], T]) -> T:
    t0 = time.perf_counter()
    try:
        return fn()
    finally:
        summary.record(stage, (time.perf_counter() - t0) * 1000.0)
