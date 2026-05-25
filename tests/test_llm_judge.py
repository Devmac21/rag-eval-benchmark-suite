import json
from unittest.mock import MagicMock, patch

from rag_eval.judge.openai_compatible import OpenAICompatibleJudge, _extract_json_object


def test_extract_json_markdown_fence():
    raw = """```json
{"faithfulness": 0.5, "answer_relevance": 0.6, "hallucination_rate": 0.2}
```"""
    out = _extract_json_object(raw)
    assert out["faithfulness"] == 0.5
    assert out["hallucination_rate"] == 0.2


def test_judge_score_parses_response():
    api_body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "faithfulness": 0.85,
                            "answer_relevance": 0.9,
                            "hallucination_rate": 0.15,
                        }
                    )
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json.return_value = api_body

    client_inst = MagicMock()
    client_inst.post.return_value = fake_resp
    cm = MagicMock()
    cm.__enter__.return_value = client_inst
    cm.__exit__.return_value = None

    judge = OpenAICompatibleJudge(base_url="http://test/v1", api_key="x", model="dummy")

    with patch("rag_eval.judge.openai_compatible.httpx.Client", return_value=cm):
        scores, usage = judge.score("q", "a", ["ctx1"])

    assert scores["faithfulness_llm"] == 0.85
    assert scores["hallucination_rate_llm"] == 0.15
    assert usage["prompt_tokens"] == 10
    client_inst.post.assert_called_once()
