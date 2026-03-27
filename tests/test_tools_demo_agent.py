import pytest

from app.agents.tools_demo.runtime import run_tools_demo_agent


@pytest.mark.asyncio
async def test_tools_demo_agent_uses_calculator_for_expression() -> None:
    result = await run_tools_demo_agent("2 + 2 * 5")

    assert result["route"] == "tool"
    assert result["selected_tool"] == "calculator"
    assert result["tool_input"] == "2 + 2 * 5"
    assert result["tool_output"] == "12"
    assert result["answer"] == "2 + 2 * 5 = 12"


@pytest.mark.asyncio
async def test_tools_demo_agent_extracts_expression_from_text() -> None:
    result = await run_tools_demo_agent("what is 10 * 5?")

    assert result["route"] == "tool"
    assert result["tool_input"] == "10 * 5"
    assert result["tool_output"] == "50"
    assert result["answer"] == "10 * 5 = 50"


@pytest.mark.asyncio
async def test_tools_demo_agent_uses_direct_for_non_math_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_route_tool_with_llm(question: str) -> str:
        assert question == "hello"
        return "direct"

    monkeypatch.setattr(
        "app.agents.tools_demo.nodes.route_tool_with_llm",
        fake_route_tool_with_llm,
    )

    result = await run_tools_demo_agent("hello")

    assert result["route"] == "direct"
    assert "answer" in result
    assert result["answer"] != ""


@pytest.mark.asyncio
async def test_tools_demo_agent_lists_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_route_tool_with_llm(question: str) -> str:
        assert question == "what documents are available?"
        return "list_docs"

    monkeypatch.setattr(
        "app.agents.tools_demo.nodes.route_tool_with_llm",
        fake_route_tool_with_llm,
    )

    records = [
        {"doc_id": "doc1", "title": "Python"},
        {"doc_id": "doc1", "title": "Python"},
        {"doc_id": "doc2", "title": "FastAPI"},
    ]

    result = await run_tools_demo_agent(
        "what documents are available?",
        records=records,
    )

    assert result["route"] == "tool"
    assert result["selected_tool"] == "list_docs"
    assert "Python (doc1)" in result["tool_output"]
    assert "FastAPI (doc2)" in result["tool_output"]
    assert "Available documents:" in result["answer"]


@pytest.mark.asyncio
async def test_tools_demo_agent_uses_search_chunks_with_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_route_tool_with_llm(question: str) -> str:
        return "search_chunks"

    async def fake_llm(prompt: str) -> str:
        return "Python is a programming language."

    monkeypatch.setattr(
        "app.agents.tools_demo.nodes.route_tool_with_llm",
        fake_route_tool_with_llm,
    )

    monkeypatch.setattr(
        "app.agents.tools_demo.nodes.run_text_prompt_with_retry_async",
        fake_llm,
    )

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

    result = await run_tools_demo_agent(
        "What is Python?",
        records=records,
    )

    assert result["route"] == "tool"
    assert result["selected_tool"] == "search_chunks"
    assert "Python" in result["answer"]