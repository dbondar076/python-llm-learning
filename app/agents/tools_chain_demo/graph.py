from langgraph.graph import END, START, StateGraph

from app.agents.tools_chain_demo.edges import after_retrieve
from app.agents.tools_chain_demo.nodes import answer_node, fallback_node, retrieve_node
from app.agents.tools_chain_demo.state import ChainState


def build_chain_graph(records, top_k: int):
    graph = StateGraph(ChainState)

    async def retrieve_bound(state: ChainState):
        return await retrieve_node(state, records, top_k)

    graph.add_node("retrieve", retrieve_bound)
    graph.add_node("answer", answer_node)
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "retrieve")

    graph.add_conditional_edges(
        "retrieve",
        after_retrieve,
        {
            "answer": "answer",
            "fallback": "fallback",
        },
    )

    graph.add_edge("answer", END)
    graph.add_edge("fallback", END)

    return graph.compile()