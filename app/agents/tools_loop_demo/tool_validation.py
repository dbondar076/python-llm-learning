from typing import Any, Tuple

from app.agents.tools_loop_demo.tool_arguments import (
    CalculatorArguments,
    SearchChunksArguments,
    ListDocsArguments,
)


def validate_tool_arguments(tool: str, arguments: dict) -> Tuple[bool, dict]:
    try:
        if tool == "calculator":
            validated = CalculatorArguments.model_validate(arguments)
            return True, validated.model_dump()

        if tool == "search_chunks":
            validated = SearchChunksArguments.model_validate(arguments or {})
            return True, validated.model_dump()

        if tool == "list_docs":
            validated = ListDocsArguments.model_validate(arguments or {})
            return True, validated.model_dump()

        if tool == "finish":
            return True, {}

        return False, {}

    except Exception:
        return False, {}