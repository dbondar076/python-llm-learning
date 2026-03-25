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
    rewrite_question_with_memory_tool,
)
from app.services.conversation_memory import (
    get_conversation_state,
    save_conversation_state,
)


class GraphState(TypedDict, total=False):
    question: str
    original_question: str
    initial_route: str
    route: str
    top_chunks: list[ScoredChunk]
    answer: str
    records: list[ChunkEmbeddingRecord]
    top_k: int
    min_score: float
    title_filter: str | None
    doc_id_filter: str | None


async def route_node(state: GraphState) -> GraphState:
    question = state["question"]
    route = await route_question_with_llm(question)
    initial_route = route

    if route == "clarify" and should_force_rag_for_resolved_question(question):
        route = "rag"

    return {
        "route": route,
        "initial_route": initial_route,
    }


async def direct_node(state: GraphState) -> GraphState:
    answer = await direct_answer_tool(state["question"])
    return {
        "answer": answer,
        "top_chunks": [],
        "route": "direct",
    }


async def clarify_node(state: GraphState) -> GraphState:
    answer = await clarify_question_tool(state["question"])
    return {
        "answer": answer,
        "top_chunks": [],
        "route": "clarify",
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
    return {
        "answer": answer,
        "route": "answer",
    }


async def fallback_node(state: GraphState) -> GraphState:
    return {
        "answer": NO_ANSWER,
        "route": "fallback",
    }


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
    session_id: str | None = None,
    top_k: int = RAG_TOP_K,
    min_score: float = RAG_MIN_SCORE,
    title_filter: str | None = None,
    doc_id_filter: str | None = None,
) -> GraphState:
    graph = build_agent_graph()

    original_question = question
    memory = None

    if session_id:
        memory = get_conversation_state(session_id)

    question = await resolve_question_with_memory(question, memory)

    result = await graph.ainvoke(
        {
            "question": question,
            "original_question": original_question,
            "records": records,
            "top_k": top_k,
            "min_score": min_score,
            "title_filter": title_filter,
            "doc_id_filter": doc_id_filter,
        }
    )

    save_memory_if_needed(session_id, original_question, result)
    return result


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


async def resolve_question_with_memory(
    question: str,
    memory: dict | None,
) -> str:
    if not memory:
        return question

    last_route = memory.get("last_agent_route")
    last_user_message = memory.get("last_user_message")
    last_agent_answer = memory.get("last_agent_answer")

    if last_route == "clarify" and last_user_message:
        return await rewrite_question_with_memory_tool(
            previous_user_message=last_user_message,
            previous_agent_answer=last_agent_answer,
            current_user_message=question,
        )

    return question


def save_memory_if_needed(
    session_id: str | None,
    question: str,
    result: GraphState,
) -> None:
    if not session_id:
        return

    save_conversation_state(
        session_id,
        {
            "last_user_message": question,
            "last_agent_route": result.get("route"),
            "last_agent_answer": result.get("answer"),
        },
    )


def build_langgraph_meta(state: GraphState) -> dict:
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