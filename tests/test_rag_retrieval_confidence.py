from app.services.rag_retrieval_service import compute_retrieval_confidence


def test_compute_retrieval_confidence_returns_zero_for_empty_chunks():
    confidence = compute_retrieval_confidence(
        query="What is Python?",
        chunks=[],
    )

    assert confidence == 0.0


def test_compute_retrieval_confidence_boosts_overlap():
    chunks = [
        {
            "doc_id": "doc1",
            "title": "Python",
            "chunk_id": "c1",
            "text": "Python is a programming language.",
            "embedding": [0.1],
            "score": 0.55,
        }
    ]

    confidence = compute_retrieval_confidence(
        query="What is Python?",
        chunks=chunks,
    )

    assert confidence > 0.55


def test_compute_retrieval_confidence_stays_low_for_irrelevant_query():
    chunks = [
        {
            "doc_id": "doc2",
            "title": "FastAPI",
            "chunk_id": "c2",
            "text": "FastAPI is a modern Python framework for APIs.",
            "embedding": [0.2],
            "score": 0.43,
        }
    ]

    confidence = compute_retrieval_confidence(
        query="What is MoonScript Ultra AI Framework?",
        chunks=chunks,
    )

    assert confidence < 0.5