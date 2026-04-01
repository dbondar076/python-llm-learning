import logging

from app.agents.rag.state import GraphState
from app.services.rag_answer_service import NO_ANSWER
from app.services.rag_retrieval_service import should_answer, ScoredChunk
from app.settings import RAG_MIN_SCORE


logger = logging.getLogger(__name__)


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
    query = state.get("question", "")
    confidence = state.get("retrieval_confidence", 0.0)
    top_score = chunks[0]["score"] if chunks else None

    decision = should_answer(query, chunks, min_score=min_score)

    logger.info(
        "RETRIEVAL route decision: chunk_count=%s top_score=%s confidence=%s min_score=%s decision=%s",
        len(chunks),
        top_score,
        confidence,
        min_score,
        "answer" if decision else "fallback",
    )

    if decision:
        return "answer"

    return "fallback"


def decide_rag_route(
    question: str,
    top_chunks: list[ScoredChunk],
    min_score: float,
    answer: str | None = None,
) -> str:
    if not top_chunks:
        return "fallback"

    can_answer = should_answer(question, top_chunks, min_score=min_score)

    if not can_answer:
        return "fallback"

    normalized_answer = (answer or "").strip()

    if not normalized_answer:
        return "fallback"

    if normalized_answer == NO_ANSWER:
        return "fallback"

    return "answer"