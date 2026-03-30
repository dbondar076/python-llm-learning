from app.agents.tools_loop_demo.tools import (
    calculator_tool,
    list_documents_tool,
    search_chunks_tool,
)


TOOLS_REGISTRY = {
    "calculator": {
        "node": "calculator",
        "kind": "math",
        "callable": calculator_tool,
    },
    "search_chunks": {
        "node": "search",
        "kind": "retrieval",
        "callable": search_chunks_tool,
    },
    "list_docs": {
        "node": "list_docs",
        "kind": "metadata",
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