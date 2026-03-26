import pytest

from app.agents.rag.response import build_langgraph_meta

pytestmark = pytest.mark.fast


def test_build_langgraph_meta_with_chunks() -> None:
    state = {
        "initial_route": "clarify",
        "route": "answer",
        "original_question": "Python",
        "question": "Python programming language",
        "top_chunks": [
            {
                "doc_id": "doc1",
                "title": "Python",
                "chunk_id": "doc1_chunk_1",
                "text": "Python is a high-level programming language.",
                "embedding": [0.1, 0.2, 0.3],
                "score": 0.67,
            },
            {
                "doc_id": "doc1",
                "title": "Python",
                "chunk_id": "doc1_chunk_2",
                "text": "It is widely used for web development.",
                "embedding": [0.2, 0.3, 0.4],
                "score": 0.31,
            },
        ],
    }

    meta = build_langgraph_meta(state)

    assert meta["initial_route"] == "clarify"
    assert meta["final_route"] == "answer"
    assert meta["original_question"] == "Python"
    assert meta["resolved_question"] == "Python programming language"
    assert meta["top_score"] == 0.67
    assert meta["chunk_count"] == 2


def test_build_langgraph_meta_without_chunks() -> None:
    state = {
        "initial_route": "clarify",
        "route": "fallback",
        "original_question": "Python",
        "question": "Information about Python",
        "top_chunks": [],
    }

    meta = build_langgraph_meta(state)

    assert meta["initial_route"] == "clarify"
    assert meta["final_route"] == "fallback"
    assert meta["original_question"] == "Python"
    assert meta["resolved_question"] == "Information about Python"
    assert meta["top_score"] is None
    assert meta["chunk_count"] == 0