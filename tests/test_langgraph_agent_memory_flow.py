import pytest
from openai import APIConnectionError

from app.services.agent_graph import run_langgraph_agent
from app.services.conversation_memory import reset_memory_store
from app.services.llm_service import reset_runtime_state
from app.services.rag_index_service import load_chunk_embeddings


pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def setup_test_state() -> None:
    reset_memory_store()
    reset_runtime_state()


@pytest.mark.asyncio
async def test_langgraph_agent_uses_memory_after_clarify() -> None:
    records = load_chunk_embeddings()
    session_id = "lg-memory-1"

    try:
        first_result = await run_langgraph_agent(
            question="what about that?",
            records=records,
            session_id=session_id,
            top_k=3,
            min_score=0.35,
        )

        assert "answer" in first_result
        assert "top_chunks" in first_result
        assert first_result["top_chunks"] == []

        second_result = await run_langgraph_agent(
            question="Python",
            records=records,
            session_id=session_id,
            top_k=3,
            min_score=0.35,
        )
    except APIConnectionError as exc:
        pytest.skip(f"OpenAI API unavailable: {exc}")

    assert "answer" in second_result
    assert "top_chunks" in second_result

    assert second_result["original_question"] == "Python"
    assert second_result["question"] != "Python"
    assert second_result["initial_route"] is not None
    assert "route" in second_result
    assert second_result["route"] in {"answer", "fallback"}
    assert len(second_result["top_chunks"]) > 0

    answer = second_result["answer"].lower()
    if second_result["route"] == "answer":
        assert "python" in answer
    else:
        assert answer == "i don't know based on the provided context."