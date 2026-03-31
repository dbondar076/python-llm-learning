from fastapi.testclient import TestClient

from app.api import app
from app.dependencies import get_retriever


class FakeRetriever:
    def search(
        self,
        query: str,
        top_k: int = 3,
        title_filter: str | None = None,
        doc_id_filter: str | None = None,
    ):
        assert query == "What is Python?"
        assert top_k == 2
        return [
            {
                "doc_id": "doc1",
                "title": "Python",
                "chunk_id": "c1",
                "text": "Python is a programming language.",
                "score": 0.9,
            }
        ]


def test_rag_search_uses_retriever_dependency() -> None:
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()

    try:
        client = TestClient(app)
        response = client.post(
            "/rag/search",
            json={
                "question": "What is Python?",
                "top_k": 2,
            },
        )

        assert response.status_code == 200

        data = response.json()
        assert len(data["chunks"]) == 1
        assert data["chunks"][0]["doc_id"] == "doc1"
        assert data["chunks"][0]["title"] == "Python"
    finally:
        app.dependency_overrides.clear()