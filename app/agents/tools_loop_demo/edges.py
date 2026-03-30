import logging
from app.agents.tools_loop_demo.state import ToolsLoopState

logger = logging.getLogger(__name__)


def route_after_decide(state: ToolsLoopState) -> str:
    selected_tool = state.get("selected_tool", "finish")

    if selected_tool == "calculator":
        return "calculator"

    if selected_tool == "search_chunks":
        return "search"

    return "finish"


def route_after_tool(state: ToolsLoopState) -> str:
    steps_taken = state.get("steps_taken", 0)
    max_steps = state.get("max_steps", 2)
    tool_output = state.get("tool_output", "")
    selected_tool = state.get("selected_tool", "search_chunks")

    logger.info(
        "LOOP step=%s tool=%s input=%s",
        steps_taken,
        selected_tool,
        tool_output,
    )

    if steps_taken >= max_steps:
        return "finish"

    if not tool_output or tool_output == "No relevant chunks found.":
        return "finish"

    if selected_tool == "search_chunks" and steps_taken > 0:
        return "finish"

    return "decide"