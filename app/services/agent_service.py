from typing import TypedDict

from app.settings import RAG_MIN_SCORE, RAG_TOP_K
from app.services.llm_service import run_text_prompt_with_retry_async
from app.services.rag_answer_service import (
    NO_ANSWER,
    build_context,
    build_rag_prompt,
    merge_adjacent_chunks,
)
from app.services.rag_index_service import ChunkEmbeddingRecord
from app.services.rag_retrieval_service import (
    ScoredChunk,
    retrieve_top_chunks,
    should_answer,
)


class AgentState(TypedDict, total=False):
    question: str
    route: str
    top_chunks: list[ScoredChunk]
    answer: str


async def route_question(state: AgentState) -> AgentState:
    if should_use_rag(state["question"]):
        state["route"] = "rag"
    else:
        state["route"] = "direct"

    return state


async def retrieve_context(
    state: AgentState,
    records: list[ChunkEmbeddingRecord],
    top_k: int = RAG_TOP_K,
    title_filter: str | None = None,
    doc_id_filter: str | None = None,
) -> AgentState:
    top_chunks = retrieve_top_chunks(
        query=state["question"],
        records=records,
        top_k=top_k,
        title_filter=title_filter,
        doc_id_filter=doc_id_filter,
    )

    state["top_chunks"] = top_chunks
    return state


async def generate_answer_node(state: AgentState) -> AgentState:
    top_chunks = state.get("top_chunks", [])

    merged_chunks = merge_adjacent_chunks(top_chunks)
    context = build_context(merged_chunks)
    prompt = build_rag_prompt(state["question"], context)
    answer = await run_text_prompt_with_retry_async(prompt)

    state["answer"] = answer
    return state


async def fallback_no_answer(state: AgentState) -> AgentState:
    state["answer"] = NO_ANSWER
    return state


async def run_rag_agent(
    question: str,
    records: list[ChunkEmbeddingRecord],
    top_k: int = RAG_TOP_K,
    min_score: float = RAG_MIN_SCORE,
    title_filter: str | None = None,
    doc_id_filter: str | None = None,
) -> tuple[list[ScoredChunk], str]:
    state: AgentState = {
        "question": question,
    }

    state = await route_question(state)

    if state["route"] == "direct":
        state["top_chunks"] = []
        state = await fallback_no_answer(state)
        return state["top_chunks"], state["answer"]

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

    return state.get("top_chunks", []), state["answer"]


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


async def decide_after_retrieval(
    state: AgentState,
    min_score: float = RAG_MIN_SCORE,
) -> AgentState:
    top_chunks = state.get("top_chunks", [])

    if should_answer(top_chunks, min_score=min_score):
        state["route"] = "answer"
    else:
        state["route"] = "fallback"

    return state