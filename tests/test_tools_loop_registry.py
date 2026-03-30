import pytest

from app.agents.tools_loop_demo.registry import (
    TOOLS_REGISTRY,
    get_tool_names,
    get_tool_node_name,
    is_known_tool,
)
from app.agents.tools_loop_demo.schemas import ToolDecision
from app.agents.tools_loop_demo.tools import decide_next_tool_with_llm


def test_tools_registry_contains_expected_tools() -> None:
    names = get_tool_names()

    assert "calculator" in names
    assert "search_chunks" in names
    assert "list_docs" in names


def test_get_tool_node_name_returns_expected_node() -> None:
    assert get_tool_node_name("calculator") == "tool"
    assert get_tool_node_name("search_chunks") == "tool"
    assert get_tool_node_name("list_docs") == "tool"


def test_is_known_tool() -> None:
    assert is_known_tool("calculator") is True
    assert is_known_tool("search_chunks") is True
    assert is_known_tool("list_docs") is True
    assert is_known_tool("unknown_tool") is False


def test_registry_entries_have_required_keys() -> None:
    for name, config in TOOLS_REGISTRY.items():
        assert "node" in config
        assert "kind" in config
        assert "input_mode" in config
        assert "description" in config
        assert "callable" in config
        assert isinstance(name, str)


@pytest.mark.asyncio
async def test_decide_next_tool_prompt_uses_registry_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    async def fake_llm(prompt: str) -> str:
        captured["prompt"] = prompt
        return '{"tool": "finish", "arguments": {}, "reason": "Enough information"}'

    monkeypatch.setattr(
        "app.agents.tools_loop_demo.tools.run_text_prompt_with_retry_async",
        fake_llm,
    )

    result = await decide_next_tool_with_llm(
        question="What is Python?",
        steps_taken=0,
        max_steps=3,
        previous_tool_output=None,
    )

    assert result.tool == "finish"
    assert result.arguments == {}
    assert result.reason == "Enough information"

    assert "- calculator ->" in captured["prompt"]
    assert "- search_chunks ->" in captured["prompt"]
    assert "- list_docs ->" in captured["prompt"]
    assert "- finish" in captured["prompt"]

    assert "Use for arithmetic calculations" in captured["prompt"]

    assert '"tool": "calculator"' in captured["prompt"]
    assert '"expression": "2 + 2 * 5"' in captured["prompt"]
    assert '"tool": "search_chunks"' in captured["prompt"]
    assert '"tool": "list_docs"' in captured["prompt"]
    assert '"tool": "finish"' in captured["prompt"]


@pytest.mark.asyncio
async def test_decide_next_tool_returns_structured_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_llm(prompt: str) -> str:
        return '{"tool": "search_chunks", "arguments": {}, "reason": "Need factual lookup"}'

    monkeypatch.setattr(
        "app.agents.tools_loop_demo.tools.run_text_prompt_with_retry_async",
        fake_llm,
    )

    result = await decide_next_tool_with_llm(
        question="What is Python?",
        steps_taken=0,
        max_steps=3,
        previous_tool_output=None,
    )

    assert isinstance(result, ToolDecision)
    assert result.tool == "search_chunks"
    assert result.arguments == {}
    assert result.reason == "Need factual lookup"


@pytest.mark.asyncio
async def test_decide_next_tool_returns_structured_decision_with_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_llm(prompt: str) -> str:
        return (
            '{"tool": "calculator", '
            '"arguments": {"expression": "2 + 2 * 5"}, '
            '"reason": "Math question"}'
        )

    monkeypatch.setattr(
        "app.agents.tools_loop_demo.tools.run_text_prompt_with_retry_async",
        fake_llm,
    )

    result = await decide_next_tool_with_llm(
        question="What is 2 + 2 * 5?",
        steps_taken=0,
        max_steps=3,
        previous_tool_output=None,
    )

    assert result.tool == "calculator"
    assert result.arguments["expression"] == "2 + 2 * 5"
    assert result.reason == "Math question"