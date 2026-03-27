from typing import TypedDict


class ToolDemoState(TypedDict, total=False):
    question: str
    route: str
    selected_tool: str
    tool_input: str
    tool_output: str
    answer: str
    records: list[dict]
    top_k: int