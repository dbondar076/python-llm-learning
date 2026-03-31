TOOL_DEFINITIONS = [
    {
        "name": "calculator",
        "description": "Use for arithmetic calculations and numeric expressions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate.",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_chunks",
        "description": "Use for factual knowledge-base questions that require searching retrieved content.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_docs",
        "description": "Use when the user asks what documents or sources are available.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
]