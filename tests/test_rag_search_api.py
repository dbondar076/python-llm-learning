import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.api import app


pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_rag_search_returns_ranked_chunks(client: TestClient) -> None:
    response = client.post(
        "/rag/search",
        json={"question": "API framework in Python"},
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    assert "chunks" in data
    assert len(data["chunks"]) == 3

    first_chunk = data["chunks"][0]
    assert first_chunk["doc_id"] == "doc2"
    assert first_chunk["title"] == "FastAPI"
    assert first_chunk["chunk_id"] == "doc2_chunk_1"

    scores = [chunk["score"] for chunk in data["chunks"]]
    assert scores == sorted(scores, reverse=True)


def test_rag_search_python_query(client: TestClient) -> None:
    response = client.post(
        "/rag/search",
        json={"question": "What can Python be used for?"},
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    assert "chunks" in data
    assert len(data["chunks"]) == 3

    chunk_ids = [chunk["chunk_id"] for chunk in data["chunks"]]
    assert "doc1_chunk_1" in chunk_ids
    assert "doc1_chunk_2" in chunk_ids


def test_rag_search_unknown_still_returns_chunks(client: TestClient) -> None:
    response = client.post(
        "/rag/search",
        json={"question": "What is JavaScript used for?"},
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()

    assert "chunks" in data
    assert len(data["chunks"]) == 3


def test_rag_search_with_title_filter(client: TestClient) -> None:
    response = client.post(
        "/rag/search",
        json={
            "question": "API framework in Python",
            "title_filter": "FastAPI",
        },
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    chunks = data["chunks"]

    assert len(chunks) == 2
    assert all(chunk["title"] == "FastAPI" for chunk in chunks)


def test_rag_search_with_doc_id_filter(client: TestClient) -> None:
    response = client.post(
        "/rag/search",
        json={
            "question": "API framework in Python",
            "doc_id_filter": "doc1",
        },
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    chunks = data["chunks"]

    assert len(chunks) == 2
    assert all(chunk["doc_id"] == "doc1" for chunk in chunks)


def test_rag_search_with_top_k(client: TestClient) -> None:
    response = client.post(
        "/rag/search",
        json={
            "question": "API framework in Python",
            "top_k": 2,
        },
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    chunks = data["chunks"]

    assert len(chunks) == 2