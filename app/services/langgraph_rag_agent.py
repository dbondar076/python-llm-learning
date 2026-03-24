from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

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
    records: list[ChunkEmbeddingRecord]
    top_k: int
    min_score: float
    title_filter: str | None
    doc_id_filter: str | None


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


async def route_question(state: AgentState) -> dict:
    if should_use_rag(state["question"]):
        return {"route": "rag"}
    return {"route": "direct"}


async def retrieve_context(state: AgentState) -> dict:
    top_chunks = retrieve_top_chunks(
        query=state["question"],
        records=state["records"],
        top_k=state.get("top_k", RAG_TOP_K),
        title_filter=state.get("title_filter"),
        doc_id_filter=state.get("doc_id_filter"),
    )
    return {"top_chunks": top_chunks}


async def decide_after_retrieval(state: AgentState) -> dict:
    top_chunks = state.get("top_chunks", [])

    if should_answer(top_chunks, min_score=state.get("min_score", RAG_MIN_SCORE)):
        return {"route": "answer"}

    return {"route": "fallback"}


async def generate_answer_node(state: AgentState) -> dict:
    top_chunks = state.get("top_chunks", [])
    merged_chunks = merge_adjacent_chunks(top_chunks)
    context = build_context(merged_chunks)
    prompt = build_rag_prompt(state["question"], context)
    answer = await run_text_prompt_with_retry_async(prompt)

    return {"answer": answer}


async def fallback_no_answer(state: AgentState) -> dict:
    return {"answer": NO_ANSWER}


def route_after_question(state: AgentState) -> Literal["retrieve_context", "fallback_no_answer"]:
    if state["route"] == "rag":
        return "retrieve_context"
    return "fallback_no_answer"


def route_after_retrieval(state: AgentState) -> Literal["generate_answer_node", "fallback_no_answer"]:
    if state["route"] == "answer":
        return "generate_answer_node"
    return "fallback_no_answer"


def build_rag_graph():
    graph = StateGraph(AgentState)

    graph.add_node("route_question", route_question)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("decide_after_retrieval", decide_after_retrieval)
    graph.add_node("generate_answer_node", generate_answer_node)
    graph.add_node("fallback_no_answer", fallback_no_answer)

    graph.add_edge(START, "route_question")

    graph.add_conditional_edges(
        "route_question",
        route_after_question,
    )

    graph.add_edge("retrieve_context", "decide_after_retrieval")

    graph.add_conditional_edges(
        "decide_after_retrieval",
        route_after_retrieval,
    )

    graph.add_edge("generate_answer_node", END)
    graph.add_edge("fallback_no_answer", END)

    return graph.compile()


rag_graph = build_rag_graph()


async def run_rag_agent_langgraph(
    question: str,
    records: list[ChunkEmbeddingRecord],
    top_k: int = RAG_TOP_K,
    min_score: float = RAG_MIN_SCORE,
    title_filter: str | None = None,
    doc_id_filter: str | None = None,
) -> tuple[list[ScoredChunk], str]:
    result = await rag_graph.ainvoke(
        {
            "question": question,
            "records": records,
            "top_k": top_k,
            "min_score": min_score,
            "title_filter": title_filter,
            "doc_id_filter": doc_id_filter,
        }
    )

    return result.get("top_chunks", []), result["answer"]