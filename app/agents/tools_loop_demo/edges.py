from app.agents.tools_loop_demo.registry import get_tool_node_name
from app.agents.tools_loop_demo.state import ToolsLoopState


def route_after_decide(state: ToolsLoopState) -> str:
    selected_tool = state.get("selected_tool", "finish")

    if selected_tool == "finish":
        return "finish"

    return get_tool_node_name(selected_tool)


def route_after_tool(state: ToolsLoopState) -> str:
    return "assess"


def route_after_assess(state: ToolsLoopState) -> str:
    next_action = state.get("next_action", "finish")

    if next_action == "continue":
        return "decide"

    return "finish"