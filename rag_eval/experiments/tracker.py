"""Persist structured benchmark artifacts."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_eval.config import get_settings


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class ExperimentTracker:
    def __init__(self, runs_dir: Path | None = None):
        self.runs_dir = Path(runs_dir or get_settings().runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, payload: dict[str, Any], *, name: str | None = None) -> dict[str, Any]:
        run_id = str(uuid.uuid4())
        stamp = utc_now_iso().replace(":", "-")
        fname = f"{stamp}_{name or 'run'}_{run_id[:8]}.json"
        path = self.runs_dir / fname
        record = {"run_id": run_id, "created_at": utc_now_iso(), **payload}
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return {"path": path, "run_id": run_id, "record": record}

    def list_runs(self) -> list[dict[str, Any]]:
        rows = []
        for p in sorted(self.runs_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                rows.append({"path": str(p), "run_id": data.get("run_id"), "created_at": data.get("created_at")})
            except json.JSONDecodeError:
                continue
        return rows

    def get_run(self, run_id: str) -> dict[str, Any]:
        for p in sorted(self.runs_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if data.get("run_id") == run_id:
                return {"path": str(p), **data}
        raise KeyError(run_id)
