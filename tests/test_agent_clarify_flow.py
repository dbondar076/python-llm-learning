import pytest

from app.services.manual_agent_service import run_rag_agent


pytestmark = pytest.mark.fast


class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, query: str, top_k: int = 3, title_filter=None, doc_id_filter=None):
        return self._chunks


retriever = FakeRetriever([...])


@pytest.mark.asyncio
async def test_run_rag_agent_uses_clarify_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_route_question(state):
        state["route"] = "clarify"
        return state

    async def fake_clarify_question_tool(question: str) -> str:
        assert question == "what about that?"
        return "Could you clarify what you mean?"

    monkeypatch.setattr("app.services.manual_agent_service.route_question", fake_route_question)
    monkeypatch.setattr(
        "app.services.manual_agent_service.clarify_question_tool",
        fake_clarify_question_tool,
    )

    chunks, answer, _ = await run_rag_agent(
        question="what about that?",
        retriever=retriever,
    )

    assert chunks == []
    assert answer == "Could you clarify what you mean?"