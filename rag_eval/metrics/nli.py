"""Optional NLI-style faithfulness via a cross-encoder (lazy-loaded)."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

_encoders: dict[str, Any] = {}


def nli_entailment_faithfulness(
    answer: str,
    contexts: list[str],
    *,
    model_name: str = "cross-encoder/nli-distilroberta-base",
    premise_max_chars: int = 2500,
) -> float:
    """
    Softmax entailment probability for (premise=context union, hypothesis=answer).
    Labels follow MNLI-trained models: contradiction=0, entailment=1, neutral=2.
    """
    try:
        from sentence_transformers import CrossEncoder

        if model_name not in _encoders:
            _encoders[model_name] = CrossEncoder(model_name)

        premise = "\n".join(contexts).strip()[:premise_max_chars]
        if not premise.strip():
            return 0.0
        hypo = answer.strip() or "."
        logits = _encoders[model_name].predict([(premise, hypo)], show_progress_bar=False)
        arr = np.asarray(logits, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        row = arr[0]
        if row.shape[0] < 3:
            return float(np.max(row))
        ex = row - np.max(row)
        probs = np.exp(ex)
        probs = probs / probs.sum()
        return float(probs[1])
    except (ImportError, OSError) as exc:
        warnings.warn(f"NLI cross-encoder unavailable ({exc!r}); faithfulness_nli skipped.", UserWarning, stacklevel=2)
        return float("nan")
