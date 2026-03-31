from typing import TypedDict


class ToolsLoopState(TypedDict, total=False):
    question: str
    steps_taken: int
    max_steps: int
    action_type: str
    selected_tool: str
    decision_reason: str
    tool_input: str
    tool_arguments: dict
    tool_output: str
    answer: str
    records: list[dict]
    top_k: int
    history: list[dict]
    next_action: str