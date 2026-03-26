import json

import pytest
from fastapi.testclient import TestClient

from app.api import app


pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_langgraph_stream_returns_meta_chunks_and_done(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/rag/answer/langgraph/stream",
        json={
            "question": "What can Python be used for?",
            "min_score": 0.35,
        },
    ) as response:
        assert response.status_code == 200

        lines = [
            json.loads(line)
            for line in response.iter_lines()
            if line
        ]

    assert len(lines) >= 3

    first = lines[0]
    last = lines[-1]
    middle = lines[1:-1]

    assert first["type"] == "meta"
    assert "meta" in first
    assert "chunks" in first

    assert first["meta"]["initial_route"] is not None
    assert first["meta"]["final_route"] == "answer"
    assert first["meta"]["chunk_count"] > 0
    assert len(first["chunks"]) > 0

    assert any(item["type"] == "chunk" for item in middle)

    chunk_text = "".join(
        item["content"]
        for item in middle
        if item["type"] == "chunk"
    )
    assert chunk_text != ""

    chunk_text_lower = chunk_text.lower()
    assert (
        "web" in chunk_text_lower
        or "automation" in chunk_text_lower
        or "data" in chunk_text_lower
        or "ai" in chunk_text_lower
    )

    assert last["type"] == "done"