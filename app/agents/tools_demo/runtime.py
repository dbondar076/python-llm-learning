from app.agents.tools_demo.graph import get_tools_demo_graph
from app.agents.tools_demo.state import ToolDemoState


async def run_tools_demo_agent(
    question: str,
    records: list[dict] | None = None,
    top_k: int = 3,
) -> ToolDemoState:
    graph = get_tools_demo_graph()

    result = await graph.ainvoke(
        {
            "question": question,
            "records": records or [],
            "top_k": top_k,
        }
    )

    return result