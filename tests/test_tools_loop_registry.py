import pytest

from app.agents.tools_loop_demo.registry import (
    TOOLS_REGISTRY,
    get_tool_names,
    get_tool_node_name,
    is_known_tool,
)
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
        return "finish"

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

    assert result == "finish"
    assert "- calculator ->" in captured["prompt"]
    assert "- search_chunks ->" in captured["prompt"]
    assert "- list_docs ->" in captured["prompt"]
    assert "- finish" in captured["prompt"]
    assert "Use for arithmetic calculations" in captured["prompt"]