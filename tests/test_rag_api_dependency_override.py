import pytest
import app.settings as settings

from fastapi import status
from fastapi.testclient import TestClient

from app.api import app
from app.dependencies import get_rag_records
from app.services.llm_service import reset_runtime_state

pytestmark = pytest.mark.fast


@pytest.fixture(autouse=True)
def setup_mock_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "USE_REAL_LLM", False)
    monkeypatch.setattr(settings, "LLM_TIMEOUT_SECONDS", 1)
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 2)
    reset_runtime_state()


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def clear_dependency_overrides() -> None:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def test_rag_answer_with_overridden_records(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_rag_records() -> list[dict]:
        return [
            {
                "doc_id": "doc1",
                "title": "Python",
                "chunk_id": "doc1_chunk_1",
                "text": "Python is a high-level programming language.",
                "embedding": [0.1, 0.2, 0.3],
            }
        ]

    async def fake_answer_with_rag(
            question: str,
            records: list[dict],
            top_k: int = 3,
            min_score: float = 0.52,
            title_filter: str | None = None,
            doc_id_filter: str | None = None,
    ):
        assert question == "What can Python be used for?"
        assert len(records) == 1
        assert records[0]["doc_id"] == "doc1"

        return (
            [
                {
                    "doc_id": "doc1",
                    "title": "Python",
                    "chunk_id": "doc1_chunk_1",
                    "text": "Python is a high-level programming language.",
                    "score": 0.99,
                }
            ],
            "Python can be used for many tasks.",
        )

    app.dependency_overrides[get_rag_records] = fake_get_rag_records
    monkeypatch.setattr("app.routers.rag.answer_with_rag", fake_answer_with_rag)

    response = client.post(
        "/rag/answer",
        json={"question": "What can Python be used for?"},
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["answer"] == "Python can be used for many tasks."
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["doc_id"] == "doc1"
    assert data["chunks"][0]["score"] == 0.99