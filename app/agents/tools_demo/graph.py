from langgraph.graph import END, START, StateGraph

from app.agents.tools_demo.edges import route_after_decide
from app.agents.tools_demo.nodes import (
    calculator_node,
    decide_tool_node,
    direct_answer_node,
    list_docs_node,
    respond_with_docs_node,
    respond_with_search_node,
    respond_with_tool_node,
    search_chunks_node,
)
from app.agents.tools_demo.state import ToolDemoState


_GRAPH = None


def build_tools_demo_graph():
    graph = StateGraph(ToolDemoState)

    graph.add_node("decide", decide_tool_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("respond_with_tool", respond_with_tool_node)
    graph.add_node("direct", direct_answer_node)
    graph.add_node("list_docs", list_docs_node)
    graph.add_node("respond_with_docs", respond_with_docs_node)
    graph.add_node("search_chunks", search_chunks_node)
    graph.add_node("respond_with_search", respond_with_search_node)

    graph.add_edge(START, "decide")

    graph.add_conditional_edges(
        "decide",
        route_after_decide,
        {
            "calculator": "calculator",
            "list_docs": "list_docs",
            "search_chunks": "search_chunks",
            "direct": "direct",
        },
    )

    graph.add_edge("calculator", "respond_with_tool")
    graph.add_edge("respond_with_tool", END)

    graph.add_edge("list_docs", "respond_with_docs")
    graph.add_edge("respond_with_docs", END)

    graph.add_edge("direct", END)

    graph.add_edge("search_chunks", "respond_with_search")
    graph.add_edge("respond_with_search", END)

    return graph.compile()


def get_tools_demo_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_tools_demo_graph()
    return _GRAPH