from app.agents.tools_loop_demo.tools import (
    calculator_tool,
    list_documents_tool,
    search_chunks_tool,
)


TOOLS_REGISTRY = {
    "calculator": {
        "node": "tool",
        "kind": "math",
        "input_mode": "tool_input",
        "callable": calculator_tool,
    },
    "search_chunks": {
        "node": "tool",
        "kind": "retrieval",
        "input_mode": "question+records",
        "callable": search_chunks_tool,
    },
    "list_docs": {
        "node": "tool",
        "kind": "metadata",
        "input_mode": "records",
        "callable": list_documents_tool,
    },
}


def get_tool_names() -> list[str]:
    return list(TOOLS_REGISTRY.keys())


def get_tool_node_name(tool_name: str) -> str:
    tool = TOOLS_REGISTRY.get(tool_name)
    if not tool:
        return "finish"
    return tool["node"]


def is_known_tool(tool_name: str) -> bool:
    return tool_name in TOOLS_REGISTRY


def get_tool_config(tool_name: str) -> dict | None:
    return TOOLS_REGISTRY.get(tool_name)