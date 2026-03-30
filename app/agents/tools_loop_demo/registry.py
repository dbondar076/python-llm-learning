from app.agents.tools_loop_demo.tool_specs import TOOLS_SPECS
from app.agents.tools_loop_demo.tools import (
    calculator_tool,
    list_documents_tool,
    search_chunks_tool,
)


TOOLS_REGISTRY = {
    "calculator": {
        **TOOLS_SPECS["calculator"],
        "callable": calculator_tool,
    },
    "search_chunks": {
        **TOOLS_SPECS["search_chunks"],
        "callable": search_chunks_tool,
    },
    "list_docs": {
        **TOOLS_SPECS["list_docs"],
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