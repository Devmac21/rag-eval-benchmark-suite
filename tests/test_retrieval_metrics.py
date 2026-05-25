from rag_eval.metrics.retrieval import mrr, recall_at_k, retrieval_aggregate
from rag_eval.schemas import QAExample, RetrievalResult, RetrievedHit


def test_recall_and_mrr():
    examples = [
        QAExample(query_id="q1", query="a", relevant_chunk_ids=["c2"]),
        QAExample(query_id="q2", query="b", relevant_chunk_ids=["x"]),
    ]
    rel = {ex.query_id: set(ex.relevant_chunk_ids) for ex in examples}
    results = [
        RetrievalResult(query_id="q1", hits=[RetrievedHit(chunk_id="c1", score=1.0), RetrievedHit(chunk_id="c2", score=0.5)]),
        RetrievalResult(query_id="q2", hits=[RetrievedHit(chunk_id="y", score=1.0)]),
    ]
    assert recall_at_k(results, rel, k=2) == 0.5
    assert abs(mrr(results, rel) - (0.5 + 0.0) / 2) < 1e-6
    agg = retrieval_aggregate(results, rel, ks=[1, 2])
    assert agg["recall@1"] == 0.0
    assert agg["recall@2"] == 0.5
