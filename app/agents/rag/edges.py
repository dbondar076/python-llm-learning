import logging

from app.agents.rag.state import GraphState
from app.services.rag_answer_service import NO_ANSWER
from app.services.rag_retrieval_service import (
    ScoredChunk,
    compute_retrieval_confidence,
    retrieve_top_chunks_multi_query,
    should_answer,
)
from app.settings import RAG_MIN_SCORE


logger = logging.getLogger(__name__)


def route_after_router(state: GraphState) -> str:
    route = state.get("route", "retrieve")

    if route == "direct":
        return "direct"

    return "retrieve"


def route_after_retrieval(state: GraphState) -> str:
    chunks = state.get("top_chunks", [])
    min_score = state.get("min_score", RAG_MIN_SCORE)
    query = state.get("question", "")
    confidence = state.get("retrieval_confidence", 0.0)
    retrieval_can_answer = state.get("retrieval_can_answer", False)
    top_score = chunks[0]["score"] if chunks else None

    retrieval_passed = should_answer(query, chunks, min_score=min_score)
    decision = retrieval_passed and retrieval_can_answer

    logger.info(
        "RETRIEVAL route decision: chunk_count=%s top_score=%s confidence=%s min_score=%s retrieval_passed=%s retrieval_can_answer=%s decision=%s",
        len(chunks),
        top_score,
        confidence,
        min_score,
        retrieval_passed,
        retrieval_can_answer,
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


def decide_retrieval_route(
    question: str,
    records: list[dict],
    top_k: int,
    min_score: float,
    title_filter: str | None = None,
    doc_id_filter: str | None = None,
) -> tuple[str, list[dict], float]:
    top_chunks = retrieve_top_chunks_multi_query(
        query=question,
        records=records,
        top_k=top_k,
        title_filter=title_filter,
        doc_id_filter=doc_id_filter,
        per_query_k=5,
    )

    confidence = compute_retrieval_confidence(question, top_chunks)
    top_score = top_chunks[0]["score"] if top_chunks else 0.0

    decision = "answer" if should_answer(question, top_chunks, min_score=min_score) else "fallback"

    logger.info(
        "RETRIEVAL route decision: chunk_count=%s top_score=%s confidence=%s min_score=%s decision=%s",
        len(top_chunks),
        top_score,
        confidence,
        min_score,
        decision,
    )

    return decision, top_chunks, confidence