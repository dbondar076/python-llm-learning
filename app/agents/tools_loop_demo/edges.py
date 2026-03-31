from app.agents.tools_loop_demo.state import ToolsLoopState
from app.agents.tools_loop_demo.registry import is_known_tool


def route_after_decide(state: ToolsLoopState) -> str:
    action_type = state.get("action_type", "finish")
    selected_tool = state.get("selected_tool", "finish")

    if action_type == "finish":
        return "finish"

    if action_type == "tool_call" and is_known_tool(selected_tool):
        return "tool"

    return "finish"


def route_after_tool(state: ToolsLoopState) -> str:
    return "assess"


def route_after_assess(state: ToolsLoopState) -> str:
    next_action = state.get("next_action", "finish")

    if next_action == "continue":
        return "decide"

    return "finish"