from langgraph.graph import END, START, StateGraph

from app.agents.tools_loop_demo.edges import route_after_decide, route_after_tool
from app.agents.tools_loop_demo.nodes import (
    calculator_node,
    decide_node,
    finish_node,
    list_docs_node,
    search_node,
)
from app.agents.tools_loop_demo.state import ToolsLoopState


def build_tools_loop_graph():
    graph = StateGraph(ToolsLoopState)

    graph.add_node("decide", decide_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("search", search_node)
    graph.add_node("list_docs", list_docs_node)
    graph.add_node("finish", finish_node)

    graph.add_edge(START, "decide")

    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {
            "calculator": "calculator",
            "search": "search",
            "list_docs": "list_docs",
            "finish": "finish",
        },
    )

    graph.add_conditional_edges(
        "calculator",
        route_after_tool,
        {
            "decide": "decide",
            "finish": "finish",
        },
    )

    graph.add_conditional_edges(
        "search",
        route_after_tool,
        {
            "decide": "decide",
            "finish": "finish",
        },
    )

    graph.add_conditional_edges(
        "list_docs",
        route_after_tool,
        {
            "decide": "decide",
            "finish": "finish",
        },
    )

    graph.add_edge("finish", END)

    return graph.compile()