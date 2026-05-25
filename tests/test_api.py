from fastapi.testclient import TestClient

from app.main import create_app


def test_health_unauthenticated():
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_chunk_compare_bm25():
    client = TestClient(create_app())
    payload = {
        "documents": [
            {"doc_id": "d1", "text": "Python was created by Guido van Rossum for readability."},
            {"doc_id": "d2", "text": "FAISS is used for vector similarity search."},
        ],
        "examples": [
            {"query_id": "q1", "query": "Who created Python?", "gold_document_id": "d1"},
        ],
        "mode": "bm25",
        "top_k": 5,
    }
    r = client.post("/benchmark/chunking", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "chunking_comparison" in body
    assert len(body["chunking_comparison"]) >= 3


def test_get_experiment_missing_returns_404():
    client = TestClient(create_app())
    r = client.get("/experiments/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404
