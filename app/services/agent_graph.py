import uuid
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
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
from app.services.agent_runtime import (
    resolve_question_with_memory,
    should_force_rag_for_resolved_question,
)


_CHECKPOINTER = InMemorySaver()
_GRAPH = None


def get_agent_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_agent_graph()
    return _GRAPH


class GraphState(TypedDict, total=False):
    question: str
    original_question: str
    messages: list
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
        "question": state["question"],
        "original_question": state.get("original_question"),
        "messages": state.get("messages", []),
    }


async def direct_node(state: GraphState) -> GraphState:
    answer = await direct_answer_tool(state["question"])
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=answer))

    return {
        "answer": answer,
        "top_chunks": [],
        "route": "direct",
        "messages": messages,
    }


async def clarify_node(state: GraphState) -> GraphState:
    answer = await clarify_question_tool(state["question"])
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=answer))

    return {
        "answer": answer,
        "top_chunks": [],
        "route": "clarify",
        "messages": messages,
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
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=answer))

    return {
        "answer": answer,
        "route": "answer",
        "messages": messages,
    }


async def fallback_node(state: GraphState) -> GraphState:
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=NO_ANSWER))

    return {
        "answer": NO_ANSWER,
        "route": "fallback",
        "messages": messages,
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

    return graph.compile(checkpointer=_CHECKPOINTER)


async def run_langgraph_agent(
    question: str,
    records: list[ChunkEmbeddingRecord],
    session_id: str | None = None,
    top_k: int = RAG_TOP_K,
    min_score: float = RAG_MIN_SCORE,
    title_filter: str | None = None,
    doc_id_filter: str | None = None,
) -> GraphState:
    graph = get_agent_graph()

    original_question = question
    memory = None

    if session_id:
        thread_id = session_id
    else:
        thread_id = f"tmp-{uuid.uuid4()}"

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await graph.aget_state(config)

    if snapshot and snapshot.values:
        values = snapshot.values
        memory = {
            "last_user_message": values.get("original_question") or values.get("question") or "",
            "last_agent_route": values.get("route"),
            "last_agent_answer": values.get("answer"),
        }

    question = await resolve_question_with_memory(question, memory)

    result = await graph.ainvoke(
        {
            "question": question,
            "original_question": original_question,
            "messages": [HumanMessage(content=original_question)],
            "records": records,
            "top_k": top_k,
            "min_score": min_score,
            "title_filter": title_filter,
            "doc_id_filter": doc_id_filter,
        },
        config=config,
    )

    return result


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


def build_langgraph_response(state: GraphState) -> dict:
    raw_chunks = state.get("top_chunks", [])

    public_chunks = [
        {
            "doc_id": c["doc_id"],
            "title": c["title"],
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "score": c["score"],
        }
        for c in raw_chunks
    ]

    return {
        "answer": state.get("answer", ""),
        "chunks": public_chunks,
        "meta": build_langgraph_meta(state),
    }