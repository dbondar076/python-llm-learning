import pytest

from app.services.manual_agent_service import decide_after_retrieval


pytestmark = pytest.mark.fast


@pytest.mark.asyncio
async def test_decide_after_retrieval_returns_answer_for_good_score() -> None:
    state = {
        "top_chunks": [
            {
                "doc_id": "doc1",
                "title": "Python",
                "chunk_id": "doc1_chunk_2",
                "text": "It is widely used for web development, automation, data analysis, and AI.",
                "embedding": [0.1, 0.2, 0.3],
                "score": 0.91,
            }
        ]
    }

    result = await decide_after_retrieval(state, min_score=0.52)

    assert result["route"] == "answer"


@pytest.mark.asyncio
async def test_decide_after_retrieval_returns_fallback_for_low_score() -> None:
    state = {
        "top_chunks": [
            {
                "doc_id": "doc1",
                "title": "Python",
                "chunk_id": "doc1_chunk_2",
                "text": "It is widely used for web development, automation, data analysis, and AI.",
                "embedding": [0.1, 0.2, 0.3],
                "score": 0.42,
            }
        ]
    }

    result = await decide_after_retrieval(state, min_score=0.52)

    assert result["route"] == "fallback"


@pytest.mark.asyncio
async def test_decide_after_retrieval_returns_fallback_when_no_chunks() -> None:
    state = {
        "top_chunks": []
    }

    result = await decide_after_retrieval(state, min_score=0.52)

    assert result["route"] == "fallback"