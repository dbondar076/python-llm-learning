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
    rewrite_question_with_memory_tool,
)
from app.services.conversation_memory import (
    get_conversation_state,
    save_conversation_state,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    question: str
    original_question: str
    route: str
    initial_route: str
    top_chunks: list[ScoredChunk]
    answer: str


async def resolve_question_with_memory(
    question: str,
    memory: dict | None,
) -> str:
    if not memory:
        print("RESOLVE: no memory")
        return question

    last_route = memory.get("last_agent_route")
    last_user_message = memory.get("last_user_message")
    last_agent_answer = memory.get("last_agent_answer")

    print("RESOLVE memory:", memory)

    if last_route == "clarify" and last_user_message:
        rewritten = await rewrite_question_with_memory_tool(
            previous_user_message=last_user_message,
            current_user_message=question,
            previous_agent_answer=last_agent_answer,
        )
        logger.info("Rewrite candidate: %r", rewritten)

        if not rewritten or is_bad_rewrite(rewritten):
            fallback = build_fallback_rewrite(question)
            logger.info("Rejected bad rewrite, fallback rewrite=%r", fallback)
            return fallback

        return rewritten

    return question


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


def should_force_rag_for_resolved_question(question: str) -> bool:
    normalized = question.strip().lower()

    technical_markers = {
        "python",
        "fastapi",
        "api",
        "programming",
        "language",
        "ai",
        "llm",
    }

    words = set(normalized.split())

    if len(words) >= 2 and words & technical_markers:
        return True

    return False


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


def save_memory_if_needed(
    session_id: str | None,
    question: str,
    state: AgentState,
) -> None:
    if not session_id:
        return

    save_conversation_state(
        session_id,
        {
            "last_user_message": question,
            "last_agent_route": state.get("route"),
            "last_agent_answer": state.get("answer"),
        },
    )


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


def is_bad_rewrite(text: str) -> bool:
    lowered = text.lower().strip()

    bad_patterns = [
        "clarify",
        "refers to",
        "refer to",
        "reference to",
        "previous message",
        "user message",
        "current user message",
        "when the user",
        "the user replies",
        "the user replied",
        "meaning of",
        "in the context of",
        "what does",
        "what do you mean",
    ]

    return any(pattern in lowered for pattern in bad_patterns)


def build_fallback_rewrite(current_user_message: str) -> str:
    text = current_user_message.strip()

    if not text:
        return current_user_message

    return f"Information about {text}"


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

        save_memory_if_needed(session_id, question, state)
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

        save_memory_if_needed(session_id, question, state)
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

    save_memory_if_needed(session_id, question, state)
    return state.get("top_chunks", []), state["answer"], build_agent_meta(state)