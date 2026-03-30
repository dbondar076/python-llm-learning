from langchain_core.messages import HumanMessage

from app.agents.tools_chain_demo.graph import build_chain_graph


async def run_tools_chain_demo_agent(
    question: str,
    records,
    top_k: int = 3,
):
    graph = build_chain_graph(records, top_k)

    result = await graph.ainvoke(
        {
            "question": question,
            "messages": [HumanMessage(content=question)],
        }
    )

    return result