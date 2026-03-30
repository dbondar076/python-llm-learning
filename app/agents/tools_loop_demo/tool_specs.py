TOOLS_SPECS = {
    "calculator": {
        "node": "tool",
        "kind": "math",
        "input_mode": "tool_input",
        "description": "Use for arithmetic calculations and numeric expressions.",
    },
    "search_chunks": {
        "node": "tool",
        "kind": "retrieval",
        "input_mode": "question+records",
        "description": "Use for factual knowledge-base questions that require searching retrieved content.",
    },
    "list_docs": {
        "node": "tool",
        "kind": "metadata",
        "input_mode": "records",
        "description": "Use when the user asks what documents or sources are available.",
    },
}