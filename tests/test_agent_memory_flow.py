import pytest
import app.settings as settings

from fastapi.testclient import TestClient

from app.api import app
from app.services.conversation_memory import reset_memory_store
from app.services.llm_service import reset_runtime_state


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def setup_test_state() -> None:
    reset_memory_store()
    reset_runtime_state()


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_rag_answer_uses_memory_after_clarify(client: TestClient) -> None:
    session_id = "memory-flow-1"

    first_response = client.post(
        "/rag/answer",
        json={
            "question": "what about that?",
            "session_id": session_id,
        },
    )

    assert first_response.status_code == 200

    first_data = first_response.json()
    assert first_data["chunks"] == []
    assert first_data["answer"]

    second_response = client.post(
        "/rag/answer",
        json={
            "question": "Python",
            "session_id": session_id,
            "min_score": 0.35,
        },
    )

    assert second_response.status_code == 200

    second_data = second_response.json()

    assert len(second_data["chunks"]) > 0

    answer = second_data["answer"].lower()
    assert "python" in answer
    assert (
        "web" in answer
        or "automation" in answer
        or "data" in answer
        or "ai" in answer
    )


def test_rag_answer_returns_meta_for_memory_flow(client: TestClient) -> None:
    session_id = "memory-meta-1"

    first_response = client.post(
        "/rag/answer",
        json={
            "question": "what about that?",
            "session_id": session_id,
        },
    )

    assert first_response.status_code == 200

    second_response = client.post(
        "/rag/answer",
        json={
            "question": "Python",
            "session_id": session_id,
            "min_score": 0.35,
        },
    )

    assert second_response.status_code == 200

    data = second_response.json()

    assert "meta" in data
    meta = data["meta"]

    assert meta["initial_route"] is not None
    assert meta["final_route"] == "answer"
    assert meta["original_question"] == "Python"
    assert meta["resolved_question"] is not None
    assert meta["resolved_question"] != ""
    assert meta["top_score"] is not None
    assert meta["chunk_count"] > 0

    assert len(data["chunks"]) == meta["chunk_count"]