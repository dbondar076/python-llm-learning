import pytest

from app.services.manual_agent_service import route_question


pytestmark = pytest.mark.fast


@pytest.mark.asyncio
async def test_route_question_uses_llm_router_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_route_question_with_llm(question: str) -> str:
        assert question == "hi"
        return "direct"

    monkeypatch.setattr(
        "app.services.manual_agent_service.route_question_with_llm",
        fake_route_question_with_llm,
    )

    state = {"question": "hi"}

    result = await route_question(state)

    assert result["route"] == "direct"


@pytest.mark.asyncio
async def test_route_question_uses_llm_router_rag(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_route_question_with_llm(question: str) -> str:
        assert question == "What can Python be used for?"
        return "rag"

    monkeypatch.setattr(
        "app.services.manual_agent_service.route_question_with_llm",
        fake_route_question_with_llm,
    )

    state = {"question": "What can Python be used for?"}

    result = await route_question(state)

    assert result["route"] == "rag"


@pytest.mark.asyncio
async def test_route_question_falls_back_to_heuristic_direct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_route_question_with_llm(question: str) -> str:
        raise RuntimeError("router failed")

    monkeypatch.setattr(
        "app.services.manual_agent_service.route_question_with_llm",
        fake_route_question_with_llm,
    )

    state = {"question": "hi"}

    result = await route_question(state)

    assert result["route"] == "direct"


@pytest.mark.asyncio
async def test_route_question_falls_back_to_heuristic_rag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_route_question_with_llm(question: str) -> str:
        raise RuntimeError("router failed")

    monkeypatch.setattr(
        "app.services.manual_agent_service.route_question_with_llm",
        fake_route_question_with_llm,
    )

    state = {"question": "What can Python be used for?"}

    result = await route_question(state)

    assert result["route"] == "rag"


@pytest.mark.asyncio
async def test_route_question_uses_llm_router_clarify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_route_question_with_llm(question: str) -> str:
        assert question == "what about that?"
        return "clarify"

    monkeypatch.setattr(
        "app.services.manual_agent_service.route_question_with_llm",
        fake_route_question_with_llm,
    )

    state = {"question": "what about that?"}

    result = await route_question(state)

    assert result["route"] == "clarify"