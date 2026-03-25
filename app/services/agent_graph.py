from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.settings import RAG_MIN_SCORE, RAG_TOP_K
from app.services.rag_answer_service import NO_ANSWER
from app.services.rag_index_service import ChunkEmbeddingRecord
from app.services.rag_retrieval_service import ScoredChunk, should_answer
from app.services.rag_tools import (
    direct_answer_tool,
    clarify_question_tool,
    generate_grounded_answer_tool,
    route_question_with_llm,
    search_chunks_tool,
)


class GraphState(TypedDict, total=False):
    question: str
    route: str
    top_chunks: list[ScoredChunk]
    answer: str
    records: list[ChunkEmbeddingRecord]
    top_k: int
    min_score: float
    title_filter: str | None
    doc_id_filter: str | None


async def route_node(state: GraphState) -> GraphState:
    route = await route_question_with_llm(state["question"])
    return {"route": route}


async def direct_node(state: GraphState) -> GraphState:
    answer = await direct_answer_tool(state["question"])
    return {
        "answer": answer,
        "top_chunks": [],
    }


async def clarify_node(state: GraphState) -> GraphState:
    answer = await clarify_question_tool(state["question"])
    return {
        "answer": answer,
        "top_chunks": [],
    }


async def retrieve_node(state: GraphState) -> GraphState:
    chunks = await search_chunks_tool(
        question=state["question"],
        records=state["records"],
        top_k=state.get("top_k", RAG_TOP_K),
        title_filter=state.get("title_filter"),
        doc_id_filter=state.get("doc_id_filter"),
    )
    return {"top_chunks": chunks}


async def answer_node(state: GraphState) -> GraphState:
    answer = await generate_grounded_answer_tool(
        question=state["question"],
        chunks=state.get("top_chunks", []),
    )
    return {"answer": answer}


async def fallback_node(state: GraphState) -> GraphState:
    return {"answer": NO_ANSWER}


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


def build_agent_graph():
    graph = StateGraph(GraphState)

    graph.add_node("route", route_node)
    graph.add_node("direct", direct_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "route")

    graph.add_conditional_edges(
        "route",
        route_after_router,
        {
            "direct": "direct",
            "clarify": "clarify",
            "retrieve": "retrieve",
        },
    )

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "answer": "answer",
            "fallback": "fallback",
        },
    )

    graph.add_edge("direct", END)
    graph.add_edge("clarify", END)
    graph.add_edge("answer", END)
    graph.add_edge("fallback", END)

    return graph.compile()


async def run_langgraph_agent(
    question: str,
    records: list[ChunkEmbeddingRecord],
    top_k: int = RAG_TOP_K,
    min_score: float = RAG_MIN_SCORE,
    title_filter: str | None = None,
    doc_id_filter: str | None = None,
) -> GraphState:
    graph = build_agent_graph()

    result = await graph.ainvoke(
        {
            "question": question,
            "records": records,
            "top_k": top_k,
            "min_score": min_score,
            "title_filter": title_filter,
            "doc_id_filter": doc_id_filter,
        }
    )

    return result