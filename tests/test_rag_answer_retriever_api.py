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
        return [
            {
                "doc_id": "doc1",
                "title": "Python",
                "chunk_id": "c1",
                "text": "Python is a programming language.",
                "score": 0.9,
            }
        ]


def test_rag_answer_uses_retriever_dependency(monkeypatch) -> None:
    async def fake_run_rag_agent(
            question: str,
            retriever,
            session_id=None,
            top_k: int = 3,
            min_score: float = 0.3,
            title_filter=None,
            doc_id_filter=None,
    ):
        chunks = retriever.search(question, top_k=top_k)
        return (
            chunks,
            "Python is a programming language.",
            {
                "initial_route": "rag",
                "final_route": "answer",
                "original_question": question,
                "resolved_question": question,
                "top_score": 0.9,
                "chunk_count": len(chunks),
            },
        )

    monkeypatch.setattr(
        "app.routers.rag.run_rag_agent",
        fake_run_rag_agent,
    )

    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()

    try:
        client = TestClient(app)
        response = client.post(
            "/rag/answer",
            json={
                "question": "What is Python?",
                "top_k": 3,
                "min_score": 0.3,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Python is a programming language."
        assert len(data["chunks"]) == 1
        assert data["chunks"][0]["doc_id"] == "doc1"
        assert data["meta"]["final_route"] == "answer"
        assert data["meta"]["chunk_count"] == 1
    finally:
        app.dependency_overrides.clear()