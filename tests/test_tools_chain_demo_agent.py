import pytest

from app.agents.tools_chain_demo.runtime import run_tools_chain_demo_agent


@pytest.mark.asyncio
async def test_tools_chain_demo_agent_retrieves_and_answers():
    records = [
        {
            "doc_id": "doc1",
            "title": "Python",
            "chunk_id": "c1",
            "text": "Python is a programming language.",
            "score": 0.9,
            "embedding": [0.1],
        }
    ]

    result = await run_tools_chain_demo_agent(
        "What is Python?",
        records=records,
    )

    assert "answer" in result
    assert "top_chunks" in result
    assert result["route"] == "answered"
    assert len(result["top_chunks"]) > 0
    assert "python" in result["answer"].lower()


@pytest.mark.asyncio
async def test_tools_chain_demo_agent_falls_back_when_no_chunks():
    result = await run_tools_chain_demo_agent(
        "Unknown topic",
        records=[],
    )

    assert result["route"] == "fallback"
    assert result["answer"] == "I don't know based on the provided context."