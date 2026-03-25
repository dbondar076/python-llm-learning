import logging
from typing import TypedDict

from app.settings import RAG_MIN_SCORE, RAG_TOP_K
from app.services.rag_answer_service import NO_ANSWER
from app.services.rag_index_service import ChunkEmbeddingRecord
from app.services.rag_retrieval_service import ScoredChunk, should_answer
from app.services.rag_tools import (
    search_chunks_tool,
    generate_grounded_answer_tool,
    direct_answer_tool,
    clarify_question_tool,
    route_question_with_llm,
)
from app.services.conversation_memory import (
    get_conversation_state,
)
from app.services.agent_runtime import (
    resolve_question_with_memory,
    should_force_rag_for_resolved_question,
    save_memory_if_needed,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    original_question: str
    route: str
    initial_route: str
    top_chunks: list[ScoredChunk]
    answer: str


def should_use_rag(question: str) -> bool:
    normalized = question.strip().lower()

    small_talk = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "bye",
    }

    if normalized in small_talk:
        return False

    if len(normalized.split()) <= 1:
        return False

    return True


async def route_question(state: AgentState) -> AgentState:
    question = state["question"]

    try:
        route = await route_question_with_llm(question)
        state["route"] = route
        logger.info("Agent router selected route=%s for question=%r", route, question)
    except Exception as exc:
        if should_use_rag(question):
            state["route"] = "rag"
        else:
            state["route"] = "direct"

        logger.warning(
            "Agent router fallback route=%s for question=%r due to error: %s",
            state["route"],
            question,
            exc,
        )

    return state


async def retrieve_context(
    state: AgentState,
    records: list[ChunkEmbeddingRecord],
    top_k: int = RAG_TOP_K,
    title_filter: str | None = None,
    doc_id_filter: str | None = None,
) -> AgentState:
    top_chunks = await search_chunks_tool(
        question=state["question"],
        records=records,
        top_k=top_k,
        title_filter=title_filter,
        doc_id_filter=doc_id_filter,
    )

    state["top_chunks"] = top_chunks
    return state


async def decide_after_retrieval(
    state: AgentState,
    min_score: float = RAG_MIN_SCORE,
) -> AgentState:
    top_chunks = state.get("top_chunks", [])

    if should_answer(top_chunks, min_score=min_score):
        state["route"] = "answer"
    else:
        state["route"] = "fallback"

    top_score = top_chunks[0]["score"] if top_chunks else None
    logger.info(
        "Agent retrieval decision route=%s top_score=%s chunk_count=%s",
        state["route"],
        top_score,
        len(top_chunks),
    )

    return state


async def generate_answer_node(state: AgentState) -> AgentState:
    top_chunks = state.get("top_chunks", [])

    answer = await generate_grounded_answer_tool(
        question=state["question"],
        chunks=top_chunks,
    )

    state["answer"] = answer
    return state


async def fallback_no_answer(state: AgentState) -> AgentState:
    state["answer"] = NO_ANSWER
    return state


def build_agent_meta(state: AgentState) -> dict:
    top_chunks = state.get("top_chunks", [])
    top_score = top_chunks[0]["score"] if top_chunks else None

    return {
        "initial_route": state.get("initial_route"),
        "final_route": state.get("route"),
        "original_question": state.get("original_question"),
        "resolved_question": state.get("question"),
        "top_score": top_score,
        "chunk_count": len(top_chunks),
    }


async def run_rag_agent(
    question: str,
    records: list[ChunkEmbeddingRecord],
    session_id: str | None = None,
    top_k: int = RAG_TOP_K,
    min_score: float = RAG_MIN_SCORE,
    title_filter: str | None = None,
    doc_id_filter: str | None = None,
) -> tuple[list[ScoredChunk], str, dict]:
    state: AgentState = {
        "question": question,
        "original_question": question,
    }

    logger.info("Agent started for question=%r session_id=%r", question, session_id)

    memory = None
    original_question = question

    if session_id:
        memory = get_conversation_state(session_id)
        logger.info("Loaded memory: %s", memory)

    question = await resolve_question_with_memory(question, memory)
    state["question"] = question

    logger.info(
        "Resolved question: original=%r resolved=%r",
        original_question,
        question,
    )

    if question != original_question:
        logger.info(
            "Resolved question with memory: original=%r resolved=%r",
            original_question,
            question,
        )

    state = await route_question(state)

    state["initial_route"] = state["route"]

    if state["route"] == "clarify" and should_force_rag_for_resolved_question(state["question"]):
        logger.info(
            "Overriding clarify -> rag for resolved technical question=%r",
            state["question"],
        )
        state["route"] = "rag"

    if state["route"] == "direct":
        state["top_chunks"] = []
        state["answer"] = await direct_answer_tool(state["question"])

        logger.info(
            "Agent finished route=%s answer_len=%s chunk_count=%s",
            state.get("route"),
            len(state.get("answer", "")),
            len(state.get("top_chunks", [])),
        )

        save_memory_if_needed(session_id, original_question, state)
        return state["top_chunks"], state["answer"], build_agent_meta(state)

    if state["route"] == "clarify":
        state["top_chunks"] = []
        state["answer"] = await clarify_question_tool(state["question"])

        logger.info(
            "Agent finished route=%s answer_len=%s chunk_count=%s",
            state.get("route"),
            len(state.get("answer", "")),
            len(state.get("top_chunks", [])),
        )

        save_memory_if_needed(session_id, original_question, state)
        return state["top_chunks"], state["answer"], build_agent_meta(state)

    state = await retrieve_context(
        state=state,
        records=records,
        top_k=top_k,
        title_filter=title_filter,
        doc_id_filter=doc_id_filter,
    )

    state = await decide_after_retrieval(
        state=state,
        min_score=min_score,
    )

    if state["route"] == "answer":
        state = await generate_answer_node(state)
    else:
        state = await fallback_no_answer(state)

    logger.info(
        "Agent finished route=%s answer_len=%s chunk_count=%s",
        state.get("route"),
        len(state.get("answer", "")),
        len(state.get("top_chunks", [])),
    )

    save_memory_if_needed(session_id, original_question, state)
    return state.get("top_chunks", []), state["answer"], build_agent_meta(state)