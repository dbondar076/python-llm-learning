from app.agents.tools_loop_demo.registry import (
    TOOLS_REGISTRY,
    get_tool_names,
    get_tool_node_name,
    is_known_tool,
)


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
        assert "callable" in config
        assert isinstance(name, str)