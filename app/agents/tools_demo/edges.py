from app.agents.tools_demo.state import ToolDemoState


def route_after_decide(state: ToolDemoState) -> str:
    if state.get("route") == "tool":
        if state.get("selected_tool") == "calculator":
            return "calculator"

        if state.get("selected_tool") == "list_docs":
            return "list_docs"

        if state.get("selected_tool") == "search_chunks":
            return "search_chunks"

    return "direct"