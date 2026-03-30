from langgraph.graph import END, START, StateGraph

from app.agents.tools_loop_demo.edges import (
    route_after_assess,
    route_after_decide,
    route_after_tool,
)
from app.agents.tools_loop_demo.nodes import (
    assess_node,
    decide_node,
    finish_node,
    tool_node,
)
from app.agents.tools_loop_demo.state import ToolsLoopState


def build_tools_loop_graph():
    graph = StateGraph(ToolsLoopState)

    graph.add_node("decide", decide_node)
    graph.add_node("tool", tool_node)
    graph.add_node("assess", assess_node)
    graph.add_node("finish", finish_node)

    graph.add_edge(START, "decide")

    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {
            "tool": "tool",
            "finish": "finish",
        },
    )

    graph.add_conditional_edges(
        "tool",
        route_after_tool,
        {
            "assess": "assess",
        },
    )

    graph.add_conditional_edges(
        "assess",
        route_after_assess,
        {
            "decide": "decide",
            "finish": "finish",
        },
    )

    graph.add_edge("finish", END)

    return graph.compile()