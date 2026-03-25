import pytest

from app.services.agent_graph import run_langgraph_agent
from app.services.rag_index_service import load_chunk_embeddings


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_langgraph_agent_answers_python_question() -> None:
    records = load_chunk_embeddings()

    result = await run_langgraph_agent(
        question="What can Python be used for?",
        records=records,
        top_k=3,
        min_score=0.35,
    )

    assert "answer" in result
    assert "top_chunks" in result

    answer = result["answer"].lower()
    assert "python" in answer
    assert (
        "web" in answer
        or "automation" in answer
        or "data" in answer
        or "ai" in answer
    )

    assert len(result["top_chunks"]) > 0