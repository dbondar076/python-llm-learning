import pytest
from fastapi import status
from fastapi.testclient import TestClient

from app.api import app

pytestmark = pytest.mark.integration

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "ok"}
    assert "X-Request-ID" in response.headers


def test_analyze() -> None:
    response = client.post("/analyze", json={"text": "What is Python?"})

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["text"] == "What is Python?"
    assert data["category"] == "question"
    assert isinstance(data["summary"], str)
    assert data["summary"].strip() != ""
    assert "X-Request-ID" in response.headers


def test_summarize() -> None:
    response = client.post(
        "/summarize",
        json={"text": "Python is a programming language."},
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert isinstance(data["summary"], str)
    assert data["summary"].strip() != ""
    assert "X-Request-ID" in response.headers


def test_classify() -> None:
    response = client.post("/classify", json={"text": "Hi"})

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["category"] == "short"
    assert "X-Request-ID" in response.headers


def test_extract_user_success() -> None:
    response = client.post("/extract-user", json={"text": "Name: Dima, Age: 30"})

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data["name"] == "Dima"
    assert data["age"] == 30
    assert "X-Request-ID" in response.headers


def test_analyze_many() -> None:
    response = client.post(
        "/analyze-many",
        json={
            "texts": [
                "What is Python?",
                "Hi",
            ]
        },
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["category"] == "question"
    assert data["results"][1]["category"] == "short"
    assert "X-Request-ID" in response.headers


def test_analyze_many_safe() -> None:
    response = client.post(
        "/analyze-many-safe",
        json={
            "texts": [
                "What is Python?",
                "Hi",
            ]
        },
    )

    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["degraded"] is False
    assert data["results"][1]["degraded"] is False
    assert "X-Request-ID" in response.headers