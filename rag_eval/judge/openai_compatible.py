"""OpenAI-compatible Chat Completions API as LLM-as-judge for RAG metrics."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

import httpx

from rag_eval.config import Settings, get_settings


class JudgeConfigurationError(ValueError):
    """Raised when LLM judge is requested but settings are incomplete."""


_SYSTEM_PROMPT = """You are an impartial evaluator for retrieval-augmented generation (RAG).
Score using ONLY the provided CONTEXT passages. Treat anything not supported by CONTEXT as unsupported, even if factually true in the real world.
Respond with a single JSON object and no other text. Use exactly these keys:
- "faithfulness": float 0-1, fraction of the ANSWER's substantive claims that are entailed or clearly supported by CONTEXT.
- "answer_relevance": float 0-1, how completely and directly the ANSWER addresses the QUERY.
- "hallucination_rate": float 0-1, estimated fraction of the ANSWER (by claims or sentences) that is unsupported by or contradicts CONTEXT.
"""


def _extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*\n?", text, re.IGNORECASE)
    if fence:
        text = text[fence.end() :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object found in judge response: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def _clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Expected numeric score, got {x!r}") from e
    return max(0.0, min(1.0, v))


class OpenAICompatibleJudge:
    """Calls POST {base_url}/chat/completions (OpenAI-compatible servers)."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        timeout_s: float = 120.0,
        max_tokens: int = 400,
        api_version: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.api_version = api_version

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> OpenAICompatibleJudge:
        s = settings or get_settings()
        if not s.openai_compatible_base_url:
            raise JudgeConfigurationError(
                "LLM judge requires RAG_EVAL_OPENAI_COMPATIBLE_BASE_URL "
                "(e.g. https://api.openai.com/v1 for OpenAI, or http://localhost:1234/v1 for LM Studio)."
            )
        return cls(
            base_url=s.openai_compatible_base_url,
            api_key=s.openai_compatible_api_key or "",
            model=s.judge_model,
            timeout_s=s.judge_timeout_s,
            max_tokens=s.judge_max_tokens,
            api_version=s.openai_compatible_api_version,
        )

    def score(
        self,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> tuple[dict[str, float], dict[str, int]]:
        """Returns (scores with *_llm keys, token usage from API if present)."""
        numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts)) if contexts else "(no context)"
        user_msg = f"""QUERY:
{query}

CONTEXT:
{numbered}

ANSWER:
{answer}
"""
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        if self.api_version:
            parsed = urlparse(url)
            q = urlencode({"api-version": self.api_version})
            sep = "&" if parsed.query else ""
            new_query = f"{parsed.query}{sep}{q}" if parsed.query else q
            url = urlunparse(parsed._replace(query=new_query))

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        try:
            raw_content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Unexpected judge API response shape: {data!r}") from e

        parsed = _extract_json_object(raw_content)
        scores = {
            "faithfulness_llm": _clamp01(parsed.get("faithfulness")),
            "answer_relevance_llm": _clamp01(parsed.get("answer_relevance")),
            "hallucination_rate_llm": _clamp01(parsed.get("hallucination_rate")),
        }
        usage_raw = data.get("usage") or {}
        usage = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens", 0)),
            "completion_tokens": int(usage_raw.get("completion_tokens", 0)),
            "total_tokens": int(usage_raw.get("total_tokens", 0)),
        }
        return scores, usage
