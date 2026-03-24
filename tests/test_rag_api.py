import pytest
from fastapi import status
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


def test_rag_answer_python(client: TestClient) -> None:
    response = client.post(
        "/rag/answer",
        json={"question": "What can Python be used for?"},
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    assert "answer" in data
    assert "chunks" in data
    assert len(data["chunks"]) == 3

    answer = data["answer"].lower()
    assert "web" in answer
    assert "automation" in answer
    assert "data" in answer
    assert "ai" in answer


def test_rag_answer_fastapi(client: TestClient) -> None:
    response = client.post(
        "/rag/answer",
        json={"question": "API framework in Python"},
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    answer = data["answer"].lower()
    assert "fastapi" in answer
    assert "framework" in answer
    assert "api" in answer


def test_rag_answer_unknown(client: TestClient) -> None:
    response = client.post(
        "/rag/answer",
        json={"question": "What is JavaScript used for?"},
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    assert data["answer"] == "I don't know based on the provided context."
    assert len(data["chunks"]) == 3


def test_rag_answer_with_title_filter(client: TestClient) -> None:
    response = client.post(
        "/rag/answer",
        json={
            "question": "API framework in Python",
            "title_filter": "FastAPI",
        },
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    assert len(data["chunks"]) == 2
    assert all(chunk["title"] == "FastAPI" for chunk in data["chunks"])

    answer = data["answer"].lower()
    assert "fastapi" in answer
    assert "framework" in answer
    assert "api" in answer


def test_rag_answer_with_doc_id_filter(client: TestClient) -> None:
    response = client.post(
        "/rag/answer",
        json={
            "question": "What can Python be used for?",
            "doc_id_filter": "doc1",
        },
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    assert len(data["chunks"]) == 2
    assert all(chunk["doc_id"] == "doc1" for chunk in data["chunks"])

    answer = data["answer"].lower()
    assert "web" in answer
    assert "automation" in answer
    assert "data" in answer
    assert "ai" in answer


def test_rag_answer_with_top_k(client: TestClient) -> None:
    response = client.post(
        "/rag/answer",
        json={
            "question": "API framework in Python",
            "top_k": 2,
        },
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    assert len(data["chunks"]) == 2


def test_rag_answer_with_high_min_score_returns_fallback(client: TestClient) -> None:
    response = client.post(
        "/rag/answer",
        json={
            "question": "What can Python be used for?",
            "min_score": 0.95,
        },
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    assert data["answer"] == "I don't know based on the provided context."
    assert len(data["chunks"]) == 3