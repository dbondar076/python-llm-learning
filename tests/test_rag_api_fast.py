import pytest
import app.settings as settings

from fastapi import status
from fastapi.testclient import TestClient

from app.api import app
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
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_rag_answer_returns_mocked_response(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_answer_with_rag(
            question: str,
            records: list[dict],
            top_k: int = 3,
            min_score: float = 0.52,
            title_filter: str | None = None,
            doc_id_filter: str | None = None,
    ):
        assert question == "What can Python be used for?"
        assert records is not None
        assert top_k == 3
        assert min_score == 0.52
        assert title_filter is None
        assert doc_id_filter is None

        return (
            [
                {
                    "doc_id": "doc1",
                    "title": "Python",
                    "chunk_id": "doc1_chunk_1",
                    "text": "Python is a high-level programming language.",
                    "score": 0.91,
                },
                {
                    "doc_id": "doc1",
                    "title": "Python",
                    "chunk_id": "doc1_chunk_2",
                    "text": "It is widely used for web development, automation, data analysis, and AI.",
                    "score": 0.88,
                },
            ],
            "Python can be used for web development, automation, data analysis, and AI.",
        )

    monkeypatch.setattr("app.routers.rag.answer_with_rag", mock_answer_with_rag)

    response = client.post(
        "/rag/answer",
        json={"question": "What can Python be used for?"},
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    assert data["answer"] == "Python can be used for web development, automation, data analysis, and AI."
    assert len(data["chunks"]) == 2

    assert data["chunks"][0]["doc_id"] == "doc1"
    assert data["chunks"][0]["title"] == "Python"
    assert data["chunks"][0]["chunk_id"] == "doc1_chunk_1"
    assert data["chunks"][0]["score"] == 0.91


def test_rag_answer_returns_mocked_unknown_answer(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    async def mock_answer_with_rag(
            question: str,
            records: list[dict],
            top_k: int = 3,
            min_score: float = 0.52,
            title_filter: str | None = None,
            doc_id_filter: str | None = None,
    ):
        return (
            [
                {
                    "doc_id": "doc1",
                    "title": "Python",
                    "chunk_id": "doc1_chunk_2",
                    "text": "It is widely used for web development, automation, data analysis, and AI.",
                    "score": 0.42,
                }
            ],
            "I don't know based on the provided context.",
        )

    monkeypatch.setattr("app.routers.rag.answer_with_rag", mock_answer_with_rag)

    response = client.post(
        "/rag/answer",
        json={"question": "What is JavaScript used for?"},
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    assert data["answer"] == "I don't know based on the provided context."
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["chunk_id"] == "doc1_chunk_2"


def test_rag_answer_returns_500_when_index_not_loaded(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(app.state, "rag_records", raising=False)

    response = client.post(
        "/rag/answer",
        json={"question": "What can Python be used for?"},
    )

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR