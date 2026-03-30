from app.agents.tools_loop_demo.graph import build_tools_loop_graph
from app.agents.tools_loop_demo.state import ToolsLoopState


_GRAPH = None


def get_tools_loop_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_tools_loop_graph()
    return _GRAPH


async def run_tools_loop_demo_agent(
    question: str,
    records: list[dict] | None = None,
    top_k: int = 3,
    max_steps: int = 2,
) -> ToolsLoopState:
    graph = get_tools_loop_graph()

    result = await graph.ainvoke(
        {
            "question": question,
            "records": records or [],
            "top_k": top_k,
            "max_steps": max_steps,
            "steps_taken": 0,
        }
    )

    return result