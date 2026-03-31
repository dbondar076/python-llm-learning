from app.services.retrievers.local_retriever import LocalRetriever


def test_local_retriever_search_returns_chunks() -> None:
    records = [
        {
            "doc_id": "doc1",
            "title": "Python",
            "chunk_id": "c1",
            "text": "Python is a programming language.",
            "embedding": [0.1],
        },
        {
            "doc_id": "doc2",
            "title": "FastAPI",
            "chunk_id": "c2",
            "text": "FastAPI is a Python framework.",
            "embedding": [0.2],
        },
    ]

    retriever = LocalRetriever(records)
    chunks = retriever.search("Python", top_k=2)

    assert len(chunks) > 0
    assert any(chunk["doc_id"] == "doc1" for chunk in chunks)