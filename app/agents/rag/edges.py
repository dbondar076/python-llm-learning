from app.agents.rag.state import GraphState
from app.services.rag_retrieval_service import should_answer
from app.settings import RAG_MIN_SCORE


def route_after_router(state: GraphState) -> str:
    route = state.get("route", "clarify")

    if route == "direct":
        return "direct"

    if route == "clarify":
        return "clarify"

    return "retrieve"


def route_after_retrieval(state: GraphState) -> str:
    chunks = state.get("top_chunks", [])
    min_score = state.get("min_score", RAG_MIN_SCORE)

    if should_answer(chunks, min_score=min_score):
        return "answer"

    return "fallback"