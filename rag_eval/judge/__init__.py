"""LLM-as-judge backends."""

from rag_eval.judge.openai_compatible import JudgeConfigurationError, OpenAICompatibleJudge

__all__ = ["OpenAICompatibleJudge", "JudgeConfigurationError"]
