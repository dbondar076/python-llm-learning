from app.agents.tools_loop_demo.registry import get_tool_node_name
from app.agents.tools_loop_demo.state import ToolsLoopState


def route_after_decide(state: ToolsLoopState) -> str:
    selected_tool = state.get("selected_tool", "finish")

    if selected_tool == "finish":
        return "finish"

    return get_tool_node_name(selected_tool)


def route_after_tool(state: ToolsLoopState) -> str:
    steps_taken = state.get("steps_taken", 0)
    max_steps = state.get("max_steps", 2)
    tool_output = state.get("tool_output", "")

    if steps_taken >= max_steps:
        return "finish"

    if not tool_output or tool_output == "No relevant chunks found.":
        return "finish"

    return "decide"