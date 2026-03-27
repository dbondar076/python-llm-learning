from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from app.agents.rag.edges import route_after_router, route_after_retrieval
from app.agents.rag.nodes import route_node, direct_node, clarify_node, retrieve_node, answer_node, fallback_node
from app.agents.rag.state import GraphState


_CHECKPOINTER = InMemorySaver()
_GRAPH = None


def build_agent_graph():
    graph = StateGraph(GraphState)

    graph.add_node("route", route_node)
    graph.add_node("direct", direct_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "route")

    graph.add_conditional_edges(
        "route",
        route_after_router,
        {
            "direct": "direct",
            "clarify": "clarify",
            "retrieve": "retrieve",
        },
    )

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "answer": "answer",
            "fallback": "fallback",
        },
    )

    graph.add_edge("direct", END)
    graph.add_edge("clarify", END)
    graph.add_edge("answer", END)
    graph.add_edge("fallback", END)

    return graph.compile(checkpointer=_CHECKPOINTER)


def get_agent_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_agent_graph()
    return _GRAPH


def reset_langgraph_runtime_state() -> None:
    global _GRAPH, _CHECKPOINTER
    _CHECKPOINTER = InMemorySaver()
    _GRAPH = None